"""正式画质评分：保留原始专家值、换算方案和比较组。"""

from .steps import step01_measure, step02_percent
from ...app.data.types import ContractError


def run(units, quality_dir, config, expected=None):
    recipe = config.checks["quality_score"]
    scores, raw, groups = {}, {}, {}
    for unit in units:
        summary = step01_measure.measure(quality_dir, unit)
        rows = step01_measure.metric_rows(summary, recipe["metric"]) or []
        values = {r["model"]: r["value"] for r in rows}
        if len(values) != len(rows):
            raise ContractError(f"{unit}: 专家测量中存在重复配置")
        if expected is not None:
            wanted = {c for c, by in expected.items() if unit in by}
            values = {c: values.get(c) for c in sorted(wanted)}
        groups[unit] = sorted(values)
        raw[unit] = values
        for c, value in step02_percent.convert(values, recipe).items():
            scores.setdefault(c, {})[unit] = value
    return {
        "schema_version": 1,
        "recipe": recipe,
        "scores": scores,
        "raw": raw,
        "comparison_groups": groups,
    }
