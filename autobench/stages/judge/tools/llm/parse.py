"""回答校验：格式不对就是错，不修补。看不见图的判断单独识别，不当成 0 分。"""

from __future__ import annotations

import re
from typing import Any

EVENT_SCORES = (0, 0.5, 1)
PHYSICAL = ("body", "persistence", "contact", "support", "interaction", "material")
REVIEW_KINDS = ("frames", "track", "mask", "flow")
STAGES = {
    "first": {"pass", "fail", "not_applicable", "needs_review"},
    "fusion": {"pass", "fail", "not_applicable", "needs_review"},
    "review": {"pass", "fail", "not_applicable", "unjudgeable"},
}
NO_VISION_MARKERS = ("没有可供观察", "没有提供可实际查看", "没有可供直接观察", "没有可见的实际帧",
                     "没有可查看的帧", "没有提供可供观察", "没有可直接观察", "无法直接观察视频帧",
                     "no frame images", "no images were provided")
BLIND_PATTERN = re.compile(
    r"(没有|未|无法)[^。]{0,12}(提供|包含|接收)[^。]{0,20}(图像|画面|帧图)"
    r"|仅包含[^。]{0,20}(编号|校验|哈希)"
    r"|没有(可供|可见|可查看|可直接观察)的?(帧|图像|画面)"
    r"|no (frame )?images? (were |was )?(provided|included)")


def _review(review: Any, duration: float) -> dict | None:
    if review is None:
        return None
    window = review.get("window")
    if (not isinstance(window, list) or len(window) != 2
            or not all(isinstance(v, (int, float)) for v in window)
            or not 0 <= window[0] <= window[1] <= duration + 1e-6):
        raise ValueError("Review window must be [start, end] seconds inside the video")
    if review.get("kind") not in REVIEW_KINDS:
        raise ValueError("Review kind must be one of " + ", ".join(REVIEW_KINDS))
    box = review.get("box")
    if box is not None:
        if (not isinstance(box, list) or len(box) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError("Review box must be [x0, y0, x1, y1] normalized inside the frame")
    if not str(review.get("question", "")).strip():
        raise ValueError("Review needs a question")
    return review


def _evidence(row: dict, frame_ids: set[str], required: bool) -> None:
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        row["status"] = "unjudgeable"
        row["evidence"] = []
        row["reason"] = (str(row.get("reason", "")) + " [malformed evidence, kept open]").strip()
        return
    row["evidence"] = [e for e in evidence if e in frame_ids]
    if required and not row["evidence"]:
        row["status"] = "unjudgeable"


def check_events(rows: list[dict], expected_ids: list[str], frame_ids: set[str],
                 duration: float, stage: str) -> None:
    if sorted(r.get("id") for r in rows) != sorted(expected_ids):
        raise ValueError("Events must be exactly: " + ", ".join(sorted(expected_ids)))
    for row in rows:
        if not str(row.get("reason", "")).strip():
            raise ValueError("%s: reason is required" % row["id"])
        score = row.get("score")
        if stage == "first":
            start, end = row.get("start"), row.get("end")
            if (start is None) != (end is None):
                raise ValueError("%s: start and end are both null or both numbers" % row["id"])
            if start is not None and not 0 <= start <= end <= duration + 1e-6:
                raise ValueError("%s: event times outside the video" % row["id"])
            if start is not None and start == end:
                raise ValueError("%s: start 与 end 不能是同一个数；帧是稀疏抽的，只在一张帧里看到时，"
                                 "start 写它前一张帧的时间，end 写它后一张帧的时间" % row["id"])
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
                raise ValueError("%s: score must be a number, got %r" % (row["id"], score))
            if score is not None and score > 0 and start is None:
                raise ValueError("%s: a scored event needs times" % row["id"])
            occurred = row.get("occurred")
            if isinstance(occurred, str):
                low = occurred.strip().lower()
                if low in ("true", "yes", "1"):
                    row["occurred"] = True
                elif low in ("false", "no", "0"):
                    row["occurred"] = False
            if not isinstance(row.get("occurred"), bool):
                raise ValueError("%s: occurred must be true or false" % row["id"])
            if score and not row["occurred"]:
                raise ValueError("%s: a step that did not happen cannot score above 0" % row["id"])
        review = _review(row.get("review"), duration) if stage != "review" else None
        if score is None:
            unresolved = review is not None or (stage == "review" and row.get("status") == "unjudgeable")
            if not unresolved:
                raise ValueError("%s: score may be null only with a review request or unjudgeable" % row["id"])
        elif score not in EVENT_SCORES:
            raise ValueError("%s: score must be 0, 0.5 or 1" % row["id"])
        if stage == "fusion" and row.get("status") not in ("resolved", "needs_review"):
            raise ValueError("%s: status must be resolved or needs_review" % row["id"])
        _evidence(row, frame_ids, required=bool(score))


def check_items(rows: list[dict], expected: dict[str, Any], frame_ids: set[str],
                duration: float, stage: str) -> list[dict]:
    """每个被问到的项恰好答一次。fusion/review 里多出来的行丢掉；首轮必须精确。"""
    asked = {r.get("id"): r for r in rows}
    missing = [i for i in expected if i not in asked]
    if missing and stage == "first":
        raise ValueError("Items missing: " + ", ".join(sorted(missing)))
    for item_id in missing:
        rows.append({"id": item_id, "status": "unjudgeable", "reason": "not answered after retries"})
        asked[item_id] = rows[-1]
    ordered = [asked[i] for i in expected]
    for row in ordered:
        item = expected[row["id"]]
        status = row.get("status")
        if item.type == "end_state" and status in (1, 0.9, 0.5, "1", "0.9", "0.5"):
            row["status"] = status = float(status)
        elif item.type == "end_state":
            # 结尾状态只有三档。写 "resolved"/"unjudgeable" 是把结论藏进状态字里，
            # 会让这一项悄悄变成"未定"，整条视频作废——退回重答，而不是替它降级。
            raise ValueError("%s: end_state 的 status 只能是 1 / 0.9 / 0.5，不能写 %r"
                             % (row["id"], status))
        elif status in ("resolved", "open"):
            row["status"] = "unjudgeable"
        elif status not in STAGES[stage]:
            raise ValueError("%s: status %r not allowed in the %s step" % (row["id"], status, stage))
        if item.type == "identity" and status == "not_applicable":
            raise ValueError("Identity applicability is decided from the reference before video judging")
        if (status == "not_applicable" and stage != "fusion" and item.type not in PHYSICAL
                and not (stage == "review" and item.type == "direction")):
            raise ValueError("%s: only physics items can be not_applicable" % row["id"])
        if not str(row.get("reason", "")).strip():
            raise ValueError("%s: reason is required" % row["id"])
        review = _review(row.get("review"), duration) if stage != "review" else None
        if status == "needs_review" and review is None:
            raise ValueError("%s: needs_review requires a review request" % row["id"])
        _evidence(row, frame_ids, required=status in ("pass", "fail"))
    return ordered


def is_blind_reason(text: Any) -> bool:
    text = str(text or "")
    return any(m in text for m in NO_VISION_MARKERS) or bool(BLIND_PATTERN.search(text))


def looks_blind(rows: list[dict]) -> bool:
    texts = [str(r.get("reason", "")) for r in rows]
    return bool(texts) and all(is_blind_reason(t) for t in texts)
