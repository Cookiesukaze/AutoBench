"""配置级汇总：一条视频被几个检测器标了，合成一个问题概率；一个配置取均值就是 rate。

用 1-∏(1-p)：被两个检测器同时标，比只被标一次更可疑。
只统计 counts 不为 false 的检测器；brightness_step 只贴标签，不进这个概率。
"""

from __future__ import annotations

from ....app.data.types import Flag


def _p(detector: dict) -> float:
    return float(detector["p"])


def joint(flags: dict[str, Flag], detectors: dict) -> float | None:
    """一条视频的联合概率；一个检测器都没测到就是 None。"""
    measured = {
        name: flag
        for name, flag in flags.items()
        if flag.flagged is not None and detectors[name].get("counts") is not False
    }
    if not measured:
        return None
    clean = 1.0
    for name, flag in measured.items():
        if flag.flagged:
            clean *= 1 - _p(detectors[name])
    return 1 - clean


def run(per_unit: dict[str, dict[str, dict[str, Flag]]], detectors: dict) -> dict:
    """per_unit: {unit: {config: {detector: Flag}}} → {config: {rate, videos, ...}}。"""
    per_config: dict[str, list[float]] = {}
    for unit, by_config in per_unit.items():
        for config, flags in by_config.items():
            value = joint(flags, detectors)
            if value is not None:
                per_config.setdefault(config, []).append(value)
    rates = {}
    for config in sorted({c for by in per_unit.values() for c in by}):
        values = per_config.get(config, [])
        rates[config] = {
            "rate": sum(values) / len(values) if values else None,
            "videos": len(
                per_unit and {u for u, by in per_unit.items() if config in by}
            ),
            "measured": len(values),
        }
    return rates
