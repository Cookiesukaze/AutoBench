"""每个小项走哪一步。

规则判定要求两件事同时成立：代码里真有这条规则，专家真的装着。
少任何一条都算计划有问题——宁可在这里炸，也不悄悄改成让 VLM 判。
"""

from __future__ import annotations

from dataclasses import replace

from ....app.types import ContractError, Item, Plan

RULES_IMPLEMENTED = ("camera", "continuity", "persistence", "contact", "hand_side")


def _route_item(item: Item, checks: dict) -> Item:
    if item.judged_by != "rule":
        return item
    if item.type not in RULES_IMPLEMENTED:
        raise ContractError(f"{item.id}: 计划要求规则判定，但代码里没有 {item.type!r} 这条规则")
    if not item.expert:
        raise ContractError(f"{item.id}: 规则判定没有指定专家")
    if not checks["experts"][item.expert]["installed"]:
        # 专家没装是配置里写明的事实，不是错误：这一项改由 VLM 判，但没有证据可等，首轮就问。
        return replace(item, judged_by="vlm", step="first")
    return item


def route(plan: Plan, checks: dict) -> Plan:
    events = tuple(
        replace(event,
                constraints=tuple(_route_item(i, checks) for i in event.constraints),
                physics=tuple(_route_item(i, checks) for i in event.physics))
        for event in plan.events)
    return replace(plan, events=events, items=tuple(_route_item(i, checks) for i in plan.items))
