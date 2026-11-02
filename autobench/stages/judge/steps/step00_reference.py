"""只看参考图，先确认身份比对是否有可用的原始对象。"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ....app.data.types import Plan
from ..tools.llm.prompt import render


def apply(plan: Plan, audit: dict) -> Plan:
    excluded = {k for k, r in audit.items() if r["status"] == "not_applicable"}
    return replace(plan, items=tuple(i for i in plan.items if i.id not in excluded),
                   events=tuple(replace(e,
                       constraints=tuple(i for i in e.constraints if i.id not in excluded),
                       physics=tuple(i for i in e.physics if i.id not in excluded))
                       for e in plan.events))


def run(client, plan: Plan, reference: Path) -> tuple[Plan, dict]:
    candidates = {i.id: i for i in plan.all_items() if i.type == "identity"}
    if not candidates:
        return plan, {}
    payload = {"items": [{"id": i.id, "subject": i.params.get("subject"),
                           "standard": i.standard} for i in candidates.values()]}
    prompt = render("judge/reference", input_json=json.dumps(payload, ensure_ascii=False))

    def validate(out):
        rows = out.get("items")
        if not isinstance(rows, list) or sorted(r.get("id", "") for r in rows) != sorted(candidates):
            raise ValueError("Reference items must match the requested identity items exactly")
        for row in rows:
            if row.get("status") not in ("applicable", "not_applicable", "unjudgeable"):
                raise ValueError("Invalid reference applicability status")
            if not str(row.get("reason", "")).strip():
                raise ValueError("Reference applicability requires visual evidence in reason")

    answer = client.ask_validated("reference", prompt,
                                  [{"id": "reference", "path": str(reference),
                                    "role": "reference_not_generated_video"}], validate, "first")
    audit = {r["id"]: {**r, "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                          "step": "reference_applicability"} for r in answer["items"]}
    # 只排除明确没有参考对象的身份项；遮挡、模糊或不确定仍须照常判定。
    return apply(plan, audit), audit
