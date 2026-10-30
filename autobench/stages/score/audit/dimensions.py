"""维度盘点：每个评分维度到底有没有数据、分布是否均匀。

它只报数，不判对错。用来发现"某个维度只在 1 道题里出现，却和其他维度等权"。
"""

from __future__ import annotations

from collections import defaultdict

from ....app.types import split_unit


def summarize(by_config: dict[str, dict[str, list[dict]]]) -> dict:
    """by_config: {config: {unit: rows}}。rows 来自 step01。"""
    per_question: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"items": 0, "measured": 0}))
    for config, by_unit in by_config.items():
        for unit, rows in by_unit.items():
            question, _ = split_unit(unit)
            for row in rows:
                cell = per_question[question][row["dimension"]]
                cell["items"] += 1
                if row["value"] is not None:
                    cell["measured"] += 1

    questions_out = {
        q: {d: {"items": c["items"], "measured": c["measured"],
                "coverage": c["measured"] / c["items"] if c["items"] else None}
            for d, c in sorted(dims.items())}
        for q, dims in sorted(per_question.items())
    }

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"questions": 0, "items": 0, "measured": 0})
    for q, dims in per_question.items():
        for d, c in dims.items():
            totals[d]["questions"] += 1
            totals[d]["items"] += c["items"]
            totals[d]["measured"] += c["measured"]
    dimensions_out = {
        d: {"questions": c["questions"], "items": c["items"], "measured": c["measured"],
            "coverage": c["measured"] / c["items"] if c["items"] else None}
        for d, c in sorted(totals.items())
    }
    thin = sorted(d for d, c in dimensions_out.items() if c["questions"] == 1)
    return {"questions": questions_out, "dimensions": dimensions_out, "single_question": thin}
