"""第一次调用：定位事件在视频里的时间，并给不需要专家的项下首轮判定。"""

from __future__ import annotations

import json
from pathlib import Path

from ....app.config import Config
from ....app.types import Plan
from ..tools.llm import parse, vote
from ..tools.llm.frames import extract, frame_count, locate_indices, sheet
from ..tools.llm.prompt import render

EVENT_SCOPE = ("Judge this step by its result: does it happen, and does its end result appear in "
               "the video? Hand usage, direction, object identity and physics are scored by their own "
               "items; a hidden hand must not turn this event into needs_review.")


def _brief(item) -> dict:
    return {"id": item.id, "standard": item.standard}


def _shown(frames: list[dict], workdir: Path, vlm: dict) -> list[dict]:
    """粗层送什么图：开了拼接就一图多帧，否则一帧一图。"""
    cfg = vlm["frames"]["sheet"]
    if "locate" not in cfg["apply_to"]:
        return frames
    return sheet(frames, workdir / "frames_locate_sheets", cfg["cols"], cfg["rows"],
                 cfg["side"], cfg["gap"], vlm["frames"]["jpeg_quality"])


def _sheet(out: dict) -> dict:
    """一张答卷：看不见帧的标成无分，事件缺省 occurred 补齐，按编号收成字典。"""
    events = {row["id"]: row for row in out["events"]}
    blind = parse.looks_blind(out["events"])
    for row in events.values():
        if blind or parse.is_blind_reason(row.get("reason")):
            row.update(score=None,
                       reason="judge saw no frames: " + str(row.get("reason", ""))[:120])
        row.setdefault("occurred", False)
    return {"events": events, "items": {row["id"]: row for row in out["items"]},
            "overall": out.get("overall", "")}


def _voted(sheets: list[dict], ids: list[str], kind: str, field: str, tie: object) -> dict:
    """同一编号跨答卷取多数；并列到顶就用第一张的原文、把该字段记成 tie。"""
    out = {}
    for ident in ids:
        pairs = [(sheet[kind][ident].get(field), sheet[kind][ident]) for sheet in sheets]
        row, tally, unanimous = vote.winner(pairs)
        if row is None:
            row = {**sheets[0][kind][ident], field: tie,
                   "reason": str(sheets[0][kind][ident].get("reason", "")) + f" [票数并列 {tally}]"}
        out[ident] = {**row, "votes": {str(k): v for k, v in tally.items()},
                      "unanimous": unanimous}
    return out


def run(client, plan: Plan, video: Path, reference: Path, workdir: Path, config: Config) -> dict:
    vlm = config.vlm
    indices = locate_indices(frame_count(video), vlm["frames"]["locate"])
    frames, meta = extract(video, workdir / "frames_locate", indices,
                           vlm["frames"]["jpeg_quality"])
    frames = _shown(frames, workdir, vlm)
    frame_ids = {tid for block in frames for tid in block.get("tiles", [block["id"]])}
    duration = meta["last_frame_seconds"]
    ref = {"id": "reference", "path": str(reference), "role": "reference_not_generated_video"}
    expected_items = {i.id: i for i in plan.all_items() if i.step == "first"}
    expected_events = [e.id for e in plan.events]

    payload = {
        "task": plan.prompt,
        "video": meta,
        "events": [{"id": e.id, "what": e.what, "completion": e.completion, "scope": EVENT_SCOPE}
                   for e in plan.events],
        "items": [_brief(i) for i in expected_items.values()],
    }
    prompt = render("judge/first", input_json=json.dumps(payload, ensure_ascii=False))

    def validate(out: dict) -> None:
        parse.check_events(out.get("events", []), expected_events, frame_ids, duration, "first")
        out["items"] = parse.check_items(out.get("items", []), expected_items, frame_ids,
                                         duration, "first")

    def draw(index: int) -> dict:
        marker = "" if index == 0 else f"\n\n（独立重采样 {index + 1}，与此前采样相互独立）"
        return _sheet(client.ask_validated("first", prompt + marker, [ref] + frames,
                                           validate, "first"))

    def still_tied(answers: list[dict]) -> bool:
        events = [vote.tied([a["events"][i].get("score") for a in answers]) for i in expected_events]
        items = [vote.tied([a["items"][i]["status"] for a in answers]) for i in expected_items]
        return any(events + items)

    answers = vote.collect(draw, int(vlm["vote"]["first"]), int(vlm["vote"]["max_votes"]),
                           still_tied)
    events = _voted(answers, expected_events, "events", "score", None)
    for row in events.values():
        # 模型自己写了核查请求，就是它说"这十张稀疏帧不够"：这一票不能当定案。
        unsettled = row.get("score") is None or row.get("review")
        row["status"] = "needs_review" if unsettled else "resolved"
        row["step"] = "first_call"
    items = _voted(answers, list(expected_items), "items", "status", "unjudgeable")
    return {"events": events, "items": items, "overall": answers[0]["overall"],
            "meta": meta, "reference": ref, "frames": len(frames)}