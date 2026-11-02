"""一条视频算三个分，并留下每一行的账。

规则与旧仓 scoring.py 一致：
- 事件分 0 / 0.5 / 1
- 约束 pass=1 fail=0；动作没发生时约束记 0
- 物理 pass=1 fail=0；动作没发生时物理不适用，退出该项
- 测不了的项不进分母，只降 coverage
- 按维度先平均，再合内容/物理；有任何一项未定，整条视频的视频分就是未定
"""

from __future__ import annotations

from collections import defaultdict

from ....app.types import ContractError, Outcome, Plan, Score, Verdict

VALUE = {"pass": 1.0, "partial": 0.5, "fail": 0.0}
PHYSICS_PREFIX = "MP"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _verdict(raw: dict) -> Verdict:
    """把 result.json 里的一行读成 Verdict；状态不认识的直接报错。"""
    status = raw["status"]
    if status in VALUE:
        return Verdict(id=raw["id"], outcome=Outcome.MEASURED, value=VALUE[status],
                       reason=raw.get("reason", ""), step=raw.get("step", ""))
    if isinstance(status, (int, float)) and not isinstance(status, bool):
        return Verdict(id=raw["id"], outcome=Outcome.MEASURED, value=float(status),
                       reason=raw.get("reason", ""), step=raw.get("step", ""))
    if status in ("unjudgeable", "needs_review"):
        return Verdict(id=raw["id"], outcome=Outcome.UNMEASURED,
                       reason=raw.get("reason", ""), step=raw.get("step", ""))
    if status in ("not_applicable", "not_triggered"):
        return Verdict(id=raw["id"], outcome=Outcome.NOT_APPLICABLE,
                       reason=raw.get("reason", ""), step=raw.get("step", ""))
    raise ValueError(f'{raw["id"]}: 不认识的状态 {status!r}')


def rows(plan: Plan, result: dict) -> list[dict]:
    """每个受评项一行：id / dimension / value / kind / why。value=None 表示退出计分。"""
    out: list[dict] = []
    for event in plan.events:
        if event.id not in result.get("events", {}):
            raise ContractError(f"{plan.question_id}: 事件 {event.id} 没有判定结果")
        state = result["events"][event.id]
        value = state["score"] if state["status"] == "resolved" else None
        out.append({"id": event.id, "dimension": event.dimension, "value": value,
                    "kind": "event", "why": None if value is not None else state["status"]})
    for item in plan.all_items():
        if item.id not in result.get("items", {}):
            raise ContractError(f"{plan.question_id}: 检查项 {item.id} 没有判定结果")
        state = result["items"][item.id]
        verdict = _verdict({**state, "id": item.id})
        value = verdict.value if verdict.outcome is Outcome.MEASURED else None
        why = None if verdict.outcome is Outcome.MEASURED else verdict.outcome.value
        physics = item.dimension.startswith(PHYSICS_PREFIX)
        owner = item.id.split(".")[0] if item.id.startswith("E") else None
        if owner and result["events"][owner].get("occurred") is False:
            if physics:
                value, why = None, "event_not_done"
            elif verdict.outcome is not Outcome.UNMEASURED:
                value, why = 0.0, "event_not_done"
        out.append({"id": item.id, "dimension": item.dimension, "value": value,
                    "kind": "physics" if physics else "constraint" if owner else "whole",
                    "why": why})
    return out


def undetermined(result: dict) -> list[str]:
    open_ids = []
    for key in ("items", "events"):
        for item_id, row in (result.get(key) or {}).items():
            if row.get("status") in ("unjudgeable", "needs_review"):
                open_ids.append(item_id)
    return open_ids


def video(plan: Plan, result: dict, aggregation: str = "dimension") -> tuple[Score, list[dict]]:
    """一条视频的 Score 与逐行账。有任何一项未定，三个分都给 None。"""
    scored = rows(plan, result)
    by_dimension: dict[str, list[float]] = defaultdict(list)
    for row in scored:
        if row["value"] is not None:
            by_dimension[row["dimension"]].append(row["value"])

    def group(prefix: str) -> float | None:
        dims = {d: v for d, v in by_dimension.items() if d.startswith(prefix)}
        if aggregation == "item":
            return _mean([x for v in dims.values() for x in v])
        return _mean([_mean(v) for v in dims.values()])

    judged = [r for r in scored if r["value"] is not None]
    coverage = len(judged) / len(scored) if scored else 0.0
    open_ids = undetermined(result)
    if open_ids:
        return Score(instruction=None, physics=None, quality=None, coverage=coverage,
                     by_dimension={}), scored
    return Score(instruction=group("CA"), physics=group("MP"),
                 quality=None, coverage=coverage,
                 by_dimension={d: _mean(v) for d, v in sorted(by_dimension.items())}), scored
