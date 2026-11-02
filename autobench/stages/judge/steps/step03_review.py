"""第三次调用：把还没定的项分组问掉。

前两步已经挑出每一段该看哪几帧，所以这里不再一项一次请求：
待判项按窗口分组，一组一次请求（同组里每项带自己的时间段和放大框），
按票数重复这张卷子。组数 × 票数 + 前两问，就是这条视频的全部调用。
"""

from __future__ import annotations

import json
from pathlib import Path

from ....app.config import Config
from ....app.data.types import Plan
from ..tools.llm import level, parse, vote
from ..tools.llm.frames import extract, review_indices, sheet
from ..tools.llm.prompt import render

EVENT_SCOPE = ("Judge this step by its result: does it happen, and does its end result appear in "
               "the video? Hand usage, direction, object identity and physics are scored by their own "
               "items; a hidden hand must not turn this event into needs_review.")

MAX_ZOOMS = 2


def _open(state: dict) -> list[tuple[str, str, dict]]:
    out = []
    for kind in ("items", "events"):
        for item_id, row in state[kind].items():
            if row["status"] == "needs_review":
                out.append((kind, item_id, row))
    return out


def _default_request(item_id: str, duration: float) -> dict:
    if item_id.endswith("end_state"):
        return {"window": [max(0.0, duration - 1.0), duration], "kind": "frames",
                "question": "请按标准判定这一项（结尾状态按 1 / 0.9 / 0.5 三档）"}
    return {"window": [0.0, duration], "kind": "frames", "question": "请按标准判定这一项"}


def _reveal_window(window: list[float], item_id: str, events: dict, duration: float,
                   margin: float) -> list[float]:
    """事件内的项：右边界至少延到下一个事件开始，因为遮挡往往到那时才解除。"""
    owner = item_id.split(".")[0] if item_id[:1] == "E" and item_id[1:2].isdigit() else None
    if owner is None or owner not in events:
        return window
    starts = sorted(row["start"] for row in events.values() if row.get("start") is not None)
    following = next((s for s in starts if s > window[0]), None)
    stop = duration if following is None else min(duration, following + margin)
    return [window[0], max(window[1], stop)]


def _vanish_windows(state: dict, fps: float) -> dict[str, list[float]]:
    """专家说"在画面中间原地消失"的对象，它从最后可见到第一次归零的那一小段。

    出错的瞬间就落在这一段里；复核必须逐帧看它，而不是在一个大窗口里均匀抽几张。
    走出画面边缘的不算：那是正常离场。
    """
    out: dict[str, list[float]] = {}
    for row in state["items"].values():
        for found in (row.get("measured") or {}).get("vanish") or []:
            last, zero = found.get("last_seen_frame"), found.get("first_zero_frame")
            if last is None or zero is None or found.get("last_seen_touches_border"):
                continue
            out[found["concept"]] = [last / fps, zero / fps]
    return out


def _item_row(answered: dict, item_id: str) -> dict:
    """一张答卷里这一项的行；没答或没看见帧都算未定，不猜。"""
    found = next((r for r in answered["items"] if r["id"] == item_id), None)
    if found is None:
        return {"status": "unjudgeable", "reason": "这张答卷没有这一项"}
    if parse.is_blind_reason(found.get("reason")):
        return {**found, "status": "unjudgeable"}
    return found


def _event_reason(answered: dict, event_id: str) -> str:
    found = next((r for r in answered["events"] if r["id"] == event_id), None)
    return str((found or {}).get("reason", ""))


def _event_score(answered: dict, event_id: str) -> float | None:
    found = next((r for r in answered["events"] if r["id"] == event_id), None)
    if found is None or parse.is_blind_reason(found.get("reason")):
        return None
    return found.get("score")


def _tally(entries: list[dict], answers: list[dict], plan_items: dict, reference: str,
           client) -> tuple[dict, dict]:
    """一组问卷的答案按票数收成结果。"""
    verdicts: dict[str, dict] = {}
    events: dict[str, dict] = {}
    for entry in entries:
        ident = entry["id"]
        if entry["kind"] == "items":
            pairs = [(_item_row(s, ident)["status"], _item_row(s, ident)) for s in answers]
            picked, tally, unanimous = vote.winner(pairs)
            if picked is None:
                picked = {**pairs[0][1], "status": "unjudgeable",
                          "reason": str(pairs[0][1].get("reason", "")) + f" [票数并列 {tally}]"}
            chosen = picked["status"]
            if plan_items[ident].type == "end_state" and not isinstance(chosen, float):
                forced = level.ask(client, plan_items[ident], reference, answers[0]["frames"])
                if forced is not None:
                    chosen = forced[0]
            verdicts[ident] = {**picked, "status": chosen,
                               "votes": {str(k): v for k, v in tally.items()},
                               "unanimous": unanimous, "step": "review"}
        else:
            pairs = [(_event_score(s, ident), {"score": _event_score(s, ident),
                                                "reason": _event_reason(s, ident)}) for s in answers]
            picked, tally, unanimous = vote.winner(pairs)
            chosen = None if picked is None else picked["score"]
            events[ident] = {"score": chosen,
                             "status": "unresolved" if chosen is None else "resolved",
                             # 复核改了分，就要把复核的理由写回去，不能留着初标那句旧话
                             **({"reason": picked["reason"]} if picked else {}),
                             "evidence": [],
                             "votes": {str(k): v for k, v in tally.items()},
                             "unanimous": unanimous, "step": "review"}
    return verdicts, events


def run(client, plan: Plan, state: dict, video: Path, workdir: Path, config: Config,
        evidence: dict, conflicts: dict[str, str] | None = None) -> dict:
    vlm = config.vlm
    items = {i.id: i for i in plan.all_items()}
    events = {e.id: e for e in plan.events}
    meta = state["meta"]
    duration = meta["last_frame_seconds"]
    samples = int(vlm["vote"]["review"])
    stagger = bool(vlm["frames"]["stagger_votes"])
    sheet_cfg = vlm["frames"]["sheet"]
    mosaicked = "review" in sheet_cfg["apply_to"]
    budget = int(vlm["budget"]["max_calls_per_video"])
    todo = _open(state)
    conflicts = conflicts or {}
    if not todo and not conflicts:
        return {"items": {}, "events": {}}

    entries: list[dict] = []
    open_events = {ident for kind, ident, _ in todo if kind == "events"}
    vanished = _vanish_windows(state, meta["fps"])
    for ident, question in conflicts.items():
        if ident in open_events:
            continue
        located = state["events"].get(ident) or {}
        window = [located.get("start") or 0.0, located.get("end") or duration]
        request = {"window": window, "kind": "frames", "question": question}
        hits = [vanished[c] for c in events[ident].concepts if c in vanished]
        if hits:
            request = {**request, "window": [min(h[0] for h in hits), max(h[1] for h in hits)],
                       "exact": True}
        entries.append({"kind": "events", "id": ident, "request": request})
    for kind, ident, row in todo:
        request = row.get("review") if isinstance(row.get("review"), dict) else _default_request(ident, duration)
        window = list(request["window"])
        if kind == "items":
            window = _reveal_window(window, ident, state["events"], duration,
                                    vlm["frames"]["event_margin_s"])
        if kind == "items" and ident.endswith(".support"):
            # 支撑是"放稳之后"的状态，窗口必须延到片尾才看得到。
            window = [window[0], max(window[1], duration)]
        # 事件不在 items 里，别拿它去查 items 的概念表。
        hits = ([vanished[c] for c in items[ident].concepts if c in vanished]
                if kind == "items" else [])
        if hits:
            window = [min(h[0] for h in hits), max(h[1] for h in hits)]
            request = {**request, "exact": True}
        entries.append({"kind": kind, "id": ident, "request": {**request, "window": window}})

    planned: dict[tuple, list[dict]] = {}
    for entry in entries:
        planned.setdefault(tuple(entry["request"]["window"]), []).append(entry)

    # 一请求最多问几个窗口由预算决定：组数 × 票数不能超过 remaining。
    max_requests = max(1, int(vlm["budget"]["max_review_requests"]))
    size = max(1, -(-len(planned) // max_requests))
    all_windows = list(planned.items())
    groups = [dict(all_windows[i:i + size]) for i in range(0, len(all_windows), size)]

    def shot(view: int, group: dict) -> list[dict]:
        per_window = (int(sheet_cfg["cols"]) * int(sheet_cfg["rows"])
                      * int(vlm["frames"]["review_sheets"]) if mosaicked
                      else max(2, int(vlm["frames"]["review"]) // max(1, len(group))))
        frames: list[dict] = []
        for order, (window, rows) in enumerate(group.items()):
            exact = any(e["request"].get("exact") for e in rows)
            wanted = per_window
            if exact and mosaicked:
                # 出错瞬间所在的短窗口：不加边距，最多给两张拼图，能装下就逐帧全给。
                frames_in = round(window[1] * meta["fps"]) - round(window[0] * meta["fps"]) + 1
                wanted = max(per_window, min(frames_in, per_window * 2))
            indices = review_indices(list(window), meta["fps"], meta["frames"], wanted,
                                     0.0 if exact else vlm["frames"]["review_margin_s"],
                                     view if stagger else 0, samples if stagger else 1)
            if not indices:
                continue
            plain, _ = extract(video, workdir / ("frames_review_w%d_s%d" % (order, view)),
                               indices, vlm["frames"]["jpeg_quality"])
            for page, block in enumerate([plain[i:i + int(sheet_cfg["cols"]) * int(sheet_cfg["rows"])]
                                          for i in range(0, len(plain),
                                                         int(sheet_cfg["cols"]) * int(sheet_cfg["rows"]))]):
                frames += (sheet(block, workdir / ("sheets_review_w%d_p%d_s%d" % (order, page, view)),
                                 int(sheet_cfg["cols"]), int(sheet_cfg["rows"]), int(sheet_cfg["side"]),
                                 int(sheet_cfg["gap"]), vlm["frames"]["jpeg_quality"])
                           if mosaicked else block)
            boxes = sorted({tuple(e["request"]["box"]) for e in rows if e["request"].get("box")})
            for box_order, box in enumerate(boxes[:MAX_ZOOMS]):
                zoom, _ = extract(video, workdir / ("frames_zoom_w%d_b%d_s%d" % (order, box_order, view)),
                                  indices, vlm["frames"]["jpeg_quality"], crop=box, prefix="Z")
                frames += (sheet(zoom, workdir / ("sheets_zoom_w%d_b%d_s%d" % (order, box_order, view)),
                                 int(sheet_cfg["cols"]), int(sheet_cfg["rows"]), int(sheet_cfg["side"]),
                                 int(sheet_cfg["gap"]), vlm["frames"]["jpeg_quality"])
                           if mosaicked else zoom)
        return frames

    def judge_group(group: dict, group_index: int) -> tuple[dict, dict]:
        group_entries = [e for rows in group.values() for e in rows]
        shown_items = [{"id": e["id"], "standard": items[e["id"]].standard, "request": e["request"]}
                       for e in group_entries if e["kind"] == "items"]
        shown_events = [{"id": e["id"], "what": events[e["id"]].what,
                         "completion": events[e["id"]].completion, "scope": EVENT_SCOPE,
                         "request": e["request"]} for e in group_entries if e["kind"] == "events"]
        prompt = render("judge/review", input_json=json.dumps(
            {"video": meta, "events": shown_events, "items": shown_items}, ensure_ascii=False))
        expected = {i["id"]: items[i["id"]] for i in shown_items}
        event_ids = [e["id"] for e in shown_events]

        def draw(index: int) -> dict:
            frames = shot(index % samples if stagger else 0, group)
            frame_ids = {tid for block in frames for tid in block.get("tiles", [block["id"]])}

            def validate(out: dict) -> None:
                parse.check_events(out.get("events", []), event_ids, frame_ids, duration, "review")
                out["items"] = parse.check_items(out.get("items", []), expected, frame_ids,
                                                 duration, "review") if expected else out.get("items", [])

            marker = "" if index == 0 else f"\n\n（独立重采样 {index + 1}，与此前采样相互独立）"
            out = client.ask_validated(f"review_g{group_index}_s{index}", prompt + marker,
                                       [state["reference"]] + frames, validate, "review")
            return {"items": out.get("items", []), "events": out.get("events", []), "frames": frames}

        def any_tied(sheets: list[dict]) -> bool:
            for entry in group_entries:
                values = [(_item_row(s, entry["id"])["status"] if entry["kind"] == "items"
                           else _event_score(s, entry["id"])) for s in sheets]
                if vote.tied(values):
                    return True
            return False

        left = budget - 2 - group_index * samples
        allowance = max(1, min(int(vlm["vote"]["max_votes"]), left))
        answers = vote.collect(draw, samples, allowance, any_tied)
        return _tally(group_entries, answers, items, state["reference"], client)

    verdicts: dict[str, dict] = {}
    resolved_events: dict[str, dict] = {}
    for group_index, group in enumerate(groups):
        part_items, part_events = judge_group(group, group_index)
        verdicts.update(part_items)
        resolved_events.update(part_events)
    return {"items": verdicts, "events": resolved_events}
