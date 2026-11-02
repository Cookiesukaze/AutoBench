"""阶段入口：三问 VLM + 五条规则，合成一条视频的判定结果。

规则先跑：它不要额度、可复现。规则判不了的项交给 VLM，VLM 首轮判不了的进融合，
融合后还没定的进复核；复核完还没定的记成未测量，不当 0 也不当 1。
"""

from __future__ import annotations

from pathlib import Path

from ...app.config import Config
from ...app.data.types import Plan
from .experts import evidence as evidence_builder
from .steps import step00_reference, step01_first, step02_fusion, step03_review
from .tools.expert import camera, contact, continuity, hand_side, persistence

RULES = {"camera": camera.decide, "continuity": continuity.decide,
         "persistence": persistence.decide, "contact": contact.decide,
         "hand_side": hand_side.decide}


def _rule_verdicts(plan: Plan, experts_dir: Path, meta: dict, config: Config) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in plan.all_items():
        if item.judged_by != "rule":
            continue
        result = RULES[item.type](item, experts_dir, meta, config.checks)
        if result is None:
            continue
        out[item.id] = {"status": result.status, "measured": result.measured,
                        "reason": result.reason, "failed_components": list(result.failed),
                        "review": result.review, "step": "expert"}
    return out


def _conflicts(plan: Plan, state: dict) -> dict[str, str]:
    """同一条视频里自相矛盾的地方：事件给了满分，它下面的物理项却说动作没发生。

    不在这里压分——把两句话一起交给复核，让模型自己选一句成立。
    """
    out: dict[str, str] = {}
    vanished = step03_review._vanish_windows(state, state["meta"]["fps"])
    for event in plan.events:
        row = state["events"].get(event.id) or {}
        gone = [c for c in event.concepts if c in vanished and row.get("start") is not None
                and row["start"] <= vanished[c][0] <= row["end"]]
        if row.get("score") and gone:
            out[event.id] = ("输入里有互相矛盾的结论：本事件此前依据稀疏帧判 score=%s（%s），"
                             "但专家数值显示 %s 在 %.2f 秒之后于画面内部消失（没有走出画面）。"
                             "请只依据窗口内逐帧画面，按本事件的完成标准重新判定完成度（0 / 0.5 / 1），"
                             "并在 reason 里说明哪一句成立、另一句为什么不成立。"
                             % (row.get("score"), str(row.get("reason", ""))[:80], gone[0],
                                vanished[gone[0]][0]))
            continue
        if row.get("score") != 1:
            continue
        blocked = [item for item in event.physics
                   if (state["items"].get(item.id) or {}).get("status") == "not_applicable"]
        if not blocked:
            continue
        cause = "；".join(
            "%s 判 not_applicable（%s）" % (item.id,
                                           str((state["items"].get(item.id) or {}).get("reason", ""))[:80])
            for item in blocked)
        out[event.id] = ("输入里有互相矛盾的结论：本事件此前判 score=1（%s），"
                         "但 %s。请按要求判定本事件完成度（0 / 0.5 / 1），"
                         "并在 reason 里说明哪一句成立、另一句为什么不成立。"
                         % (str(row.get("reason", ""))[:80], cause))
    return out


def run(client, plan: Plan, video: Path, reference: Path, experts_dir: Path,
        workdir: Path, config: Config, reference_audit: dict | None = None) -> dict:
    if reference_audit is None:
        plan, reference_audit = step00_reference.run(client, plan, reference)
    else:
        plan = step00_reference.apply(plan, reference_audit)
    first = step01_first.run(client, plan, video, reference, workdir, config)
    located = {eid: {"start": row.get("start"), "end": row.get("end")}
               for eid, row in first["events"].items()}
    evidence = evidence_builder.build(plan, located, experts_dir, first["meta"], config)
    rules = _rule_verdicts(plan, experts_dir, first["meta"], config)

    state = {"meta": first["meta"], "reference": first["reference"], "overall": first["overall"],
             "events": {eid: dict(row) for eid, row in first["events"].items()},
             "items": {}}
    for item in plan.all_items():
        if item.id in rules:
            state["items"][item.id] = rules[item.id]
        elif item.id in first["items"]:
            state["items"][item.id] = {**first["items"][item.id], "step": "first_call"}
        else:
            # 规则要的数组不在：这一项没有判定，交融合。
            state["items"][item.id] = {"status": "needs_review", "reason": "rule arrays missing",
                                       "step": "pending"}

    fused = step02_fusion.run(client, plan, first, rules, evidence, video, workdir, config)
    for item_id, row in fused["items"].items():
        state["items"][item_id] = {**state["items"].get(item_id, {}), **row}
    for eid, row in fused["events"].items():
        state["events"][eid] = {**state["events"][eid], **row}

    # 专家说某个对象在画面里消失了：它名下的方向、持续性、数量项，融合只给初步意见，
    # 一律交第三步，用"最后可见到第一次归零"那一小段逐帧重判。稀疏帧上的一票不能定案。
    vanished = step03_review._vanish_windows(state, first["meta"]["fps"])
    for item in plan.all_items():
        row = state["items"][item.id]
        if (item.type in ("direction", "persistence", "count") and row.get("step") != "first_call"
                and row.get("status") != "needs_review"
                and any(c in vanished for c in item.concepts)):
            state["items"][item.id] = {**row, "fusion_status": row.get("status"),
                                       "status": "needs_review"}

    reviewed = step03_review.run(client, plan, state, video, workdir, config, evidence,
                                 conflicts=_conflicts(plan, state))
    for item_id, row in reviewed["items"].items():
        state["items"][item_id] = {**state["items"][item_id], **row}
    for eid, row in reviewed["events"].items():
        state["events"][eid] = {**state["events"][eid], **row}

    for item_id, row in state["items"].items():
        if row.get("status") in ("needs_review", "unresolved"):
            state["items"][item_id] = {**row, "status": "unjudgeable",
                                       "reason": row.get("reason", "") + " [仍未能判定]"}
    for eid, row in state["events"].items():
        if row.get("status") in ("needs_review", "unresolved"):
            state["events"][eid] = {**row, "status": "unjudgeable"}
        elif row.get("status") == "resolved":
            state["events"][eid] = {**row, "occurred": bool(row.get("occurred", True))}

    for item_id, row in reference_audit.items():
        if row["status"] == "not_applicable":
            state["items"][item_id] = {**row, "evidence": [],
                                       "excluded_because": "reference_subject_absent"}
        elif row["status"] == "unjudgeable":
            state["items"][item_id] = {**row, "status": "unjudgeable", "evidence": []}
    state["reference_applicability"] = reference_audit
    state["expert_requests_not_run"] = []
    return state
