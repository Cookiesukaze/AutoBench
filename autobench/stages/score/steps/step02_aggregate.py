"""配置级聚合：把多条视频的分数合成每个配置的一组数。"""

from __future__ import annotations

from collections import defaultdict

from ....app.types import Score, split_unit


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(by_unit: dict[str, Score]) -> dict:
    """一个配置的多条视频 → 各维度均值、覆盖率、按 seed 的稳定性。"""
    scores = list(by_unit.values())
    dims: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        for dim, value in score.by_dimension.items():
            if value is not None:
                dims[dim].append(value)
    judged = sum(1 for s in scores if s.coverage is not None)
    coverage = _mean([s.coverage for s in scores])
    seeds: dict[str, list[Score]] = defaultdict(list)
    for unit, score in by_unit.items():
        seeds[split_unit(unit)[1]].append(score)
    return {
        "videos": len(scores),
        "judged_videos": judged,
        "instruction": _mean([s.instruction for s in scores if s.instruction is not None]),
        "physics": _mean([s.physics for s in scores if s.physics is not None]),
        "by_dimension": {d: _mean(v) for d, v in sorted(dims.items())},
        "coverage": coverage,
        "by_seed": {
            seed: {
                "videos": len(rows),
                "instruction": _mean([r.instruction for r in rows if r.instruction is not None]),
                "physics": _mean([r.physics for r in rows if r.physics is not None]),
            }
            for seed, rows in sorted(seeds.items())
        },
    }
