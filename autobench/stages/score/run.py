"""阶段入口：从判定结果算出分数与排行榜。

不重新判定任何东西：只读 result.json 和 plan，做算术。
质量维来自上一关的配置级 rate（1 - rate，越高越干净）。
"""

from __future__ import annotations

from dataclasses import replace

from ...app.config import Config
from ...app.types import Plan, split_unit
from .audit import dimensions as audit_dimensions
from .steps import step01_video, step02_aggregate, step03_leaderboard


def _weights(config: Config) -> dict[str, float]:
    w = config.checks["weights"]
    return {"instruction": float(w["content"]), "physics": float(w["physics"]),
            "quality": float(w["quality"])}


def _options(config: Config) -> step03_leaderboard.Options:
    w = config.checks["weights"]
    b = w["bootstrap"]
    return step03_leaderboard.Options(
        category_weights={"content": float(w["content"]), "physics": float(w["physics"])},
        min_coverage=float(w["min_coverage"]),
        n_boot=int(b["n"]), seed=int(b["seed"]), alpha=float(b["alpha"]),
        separable_at=float(b["separable_at"]),
    )


def run(results: dict[str, dict[str, dict]], plans: dict[str, Plan],
        quality_rates: dict[str, float | None], config: Config) -> dict:
    """results: {config: {unit: result.json 内容}}；plans: {question: Plan}。"""
    weights = _weights(config)
    aggregation = config.checks["weights"]["aggregation"]
    per_video: dict[str, dict[str, dict]] = {}
    rows_for_board: dict[str, dict[str, list[dict]]] = {}
    scores_by_config: dict[str, dict[str, object]] = {}

    for config_name, by_unit in sorted(results.items()):
        rate = quality_rates.get(config_name)
        quality = None if rate is None else 1.0 - rate
        per_video[config_name] = {}
        rows_for_board[config_name] = {}
        scores_by_config[config_name] = {}
        for unit, result in sorted(by_unit.items()):
            question, _ = split_unit(unit)
            if question not in plans:
                raise KeyError(f"{unit}: 没有这道题的计划")
            score, rows = step01_video.video(plans[question], result, aggregation)
            score = replace(score, quality=quality)
            scores_by_config[config_name][unit] = score
            rows_for_board[config_name][unit] = rows
            row = {
                "instruction": score.instruction, "physics": score.physics,
                "quality": score.quality, "coverage": score.coverage,
            }
            try:
                row["total"] = score.total(weights)
            except Exception:
                row["total"] = None
            per_video[config_name][unit] = row

    per_config = {c: step02_aggregate.aggregate(s) for c, s in sorted(scores_by_config.items())}
    for config_name, rows in per_config.items():
        parts = {"instruction": rows["instruction"], "physics": rows["physics"],
                 "quality": quality_rates.get(config_name)}
        quality = None if parts["quality"] is None else 1.0 - parts["quality"]
        parts["quality"] = quality
        if any(v is None for v in parts.values()):
            rows["total"] = None
            rows["total_undetermined"] = "unresolved_or_unmeasured"
        else:
            rows["total"] = (parts["instruction"] * weights["instruction"]
                             + parts["physics"] * weights["physics"]
                             + quality * weights["quality"]) / sum(weights.values())
        rows["quality"] = quality

    board = step03_leaderboard.rank(rows_for_board, _options(config),
                                    config.checks["weights"]["categories"])
    audit = audit_dimensions.summarize(rows_for_board)
    return {"per_video": per_video, "per_config": per_config, "board": board, "dimensions": audit}
