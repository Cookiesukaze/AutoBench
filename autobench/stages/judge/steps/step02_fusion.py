"""第二次调用：带着专家证据，重判需要证据的项和首轮没定的项。"""

from __future__ import annotations

import json
from pathlib import Path

from ....app.config import Config
from ....app.data.types import Plan
from ..tools.llm import level, parse, vote
from ..tools.llm.frames import extract, fusion_indices, window_indices
from ..tools.llm.prompt import render

EVENT_SCOPE = ("Judge this step by its result: does it happen, and does its end result appear in "
               "the video? Hand usage, direction, object identity and physics are scored by their own "
               "items; a hidden hand must not turn this event into needs_review.")


def _span(row: dict, following: float | None, duration: float, margin: float,
          tail: str) -> tuple[float, float]:
    """一个事件在融合这一步看哪一段时间。

    "margin" 只看动作本身（旧口径）；"next" 一直看到下一个事件开始；"end" 看到片尾。
    判断"这一步有没有发生/有没有放下"时，决定性证据常常在遮挡解除之后，所以后两种才有意义。
    """
    start = max(0.0, row["start"] - margin)
    if tail == "end":
        return start, duration
    if tail == "next" and following is not None:
        return start, min(duration, following + margin)
    return start, min(duration, row["end"] + margin)


def pending(plan: Plan, first: dict, rule_results: dict,
            size_outlier: float) -> dict[str, str]:
    """哪些项必须在这轮再问一次，以及为什么。"""
    out: dict[str, str] = {}
    for item in plan.all_items():
        if item.step == "fusion":
            out[item.id] = "expert_evidence"
            continue
        if item.judged_by != "vlm":
            continue
        if item.id not in first["items"]:
            out[item.id] = "expert_evidence"
        elif first["items"][item.id]["status"] == "needs_review":
            out[item.id] = "first_call_unsettled"
    for item in plan.all_items():
        if item.judged_by != "rule":
            continue
        if item.id not in rule_results:
            out[item.id] = "rule_arrays_missing"
            continue
        row = rule_results[item.id]
        if not row.get("review"):
            continue
        # 带数字的 fail 可能是题面自己要求的位移或大小，交 VLM 看数字复核语义。
        # 体型离群不在此列：题面写"更大"不能给任何倍数背书。
        outliers = [f for f in (row.get("measured") or {}).get("size_group") or []
                    if f.get("ratio", 0) >= size_outlier]
        if not outliers:
            out[item.id] = "rule_semantics_check"
    return out


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


def run(client, plan: Plan, first: dict, rule_results: dict, evidence: dict,
        video: Path, workdir: Path, config: Config) -> dict:
    vlm = config.vlm
    todo = pending(plan, first, rule_results,
                   float(config.checks["thresholds"]["judge"]["size_outlier_ratio"]))
    review_events = [eid for eid, row in first["events"].items() if row["status"] == "needs_review"]
    if not todo and not review_events:
        return {"items": {}, "events": {}}

    meta = first["meta"]
    duration = meta["last_frame_seconds"]
    starts = sorted(row["start"] for row in first["events"].values() if row.get("start") is not None)
    tail = vlm["frames"]["event_tail_to"]
    windows = {}
    for eid, row in first["events"].items():
        if row.get("start") is None:
            continue
        following = next((s for s in starts if s > row["start"]), None)
        first_second, last_second = _span(row, following, duration,
                                          vlm["frames"]["event_margin_s"], tail)
        windows[eid] = window_indices(first_second, last_second, meta["fps"], meta["frames"],
                                      vlm["frames"]["window"])
    indices = (fusion_indices(windows, meta["frames"], meta["fps"], vlm["frames"]["window"])
               if windows else list(range(min(meta["frames"], vlm["frames"]["locate"]))))
    frames, _ = extract(video, workdir / "frames_fusion", indices, vlm["frames"]["jpeg_quality"])
    frame_ids = {f["id"] for f in frames}
    duration = meta["last_frame_seconds"]
    items = {i.id: i for i in plan.all_items()}

    shown = []
    for item_id, why in todo.items():
        item = items[item_id]
        row = {"id": item.id, "standard": item.standard,
               "expert_evidence": evidence.get(item.id), "reason_asked": why}
        if why == "rule_arrays_missing":
            row["provisional"] = True
        elif why == "rule_semantics_check":
            row["rule_measurements"] = (rule_results[item_id] or {}).get("measured")
        shown.append(row)
    located = [{"id": eid, "start": row["start"], "end": row["end"]}
               for eid, row in first["events"].items()]
    reconfirm = [{"id": eid, "what": next(e.what for e in plan.events if e.id == eid),
                  "completion": next(e.completion for e in plan.events if e.id == eid),
                  "scope": EVENT_SCOPE, "first_call": first["events"][eid]["reason"],
                  "expert_evidence": evidence.get(eid)} for eid in review_events]
    prompt = render("judge/fusion", input_json=json.dumps(
        {"video": meta, "located": located, "events": reconfirm, "items": shown}, ensure_ascii=False))
    expected_items = {i: items[i] for i in todo}

    def validate(out: dict) -> None:
        parse.check_events(out.get("events", []), review_events, frame_ids, duration, "fusion")
        out["items"] = parse.check_items(out.get("items", []), expected_items, frame_ids,
                                         duration, "fusion")

    def draw(index: int) -> dict:
        marker = "" if index == 0 else f"\n\n（独立重采样 {index + 1}，与此前采样相互独立）"
        out = client.ask_validated("fusion", prompt + marker, [first["reference"]] + frames,
                                   validate, "fusion")
        return {"events": {row["id"]: row for row in out["events"]},
                "items": {row["id"]: row for row in out["items"]}}

    def still_tied(sheets: list[dict]) -> bool:
        items_tied = [vote.tied([s["items"][i]["status"] for s in sheets]) for i in todo]
        events_tied = [vote.tied([s["events"][i].get("score") for s in sheets])
                       for i in review_events]
        return any(items_tied + events_tied)

    sheets = vote.collect(draw, int(vlm["vote"]["fusion"]), int(vlm["vote"]["max_votes"]),
                          still_tied)
    verdicts = {item_id: {**row, "step": "fusion", "asked_because": todo[item_id]}
                for item_id, row in _voted(sheets, list(todo), "items", "status",
                                           "unjudgeable").items()}
    for item_id, row in verdicts.items():
        if items[item_id].type == "end_state" and not isinstance(row.get("status"), float):
            forced = level.ask(client, items[item_id], first["reference"], frames)
            if forced is not None:
                row["status"] = forced[0]
                row["reason"] = (row.get("reason") or "") + " [三档追问: " + forced[1] + "]"
    events = {}
    for eid, row in _voted(sheets, review_events, "events", "score", None).items():
        events[eid] = {"score": row.get("score"),
                       "status": "resolved" if row.get("score") is not None else "needs_review",
                       "reason": row.get("reason", ""), "evidence": row.get("evidence", []),
                       "votes": row["votes"], "unanimous": row["unanimous"], "step": "fusion"}
    return {"items": verdicts, "events": events}
