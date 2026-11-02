"""阶段入口：读测量 → 三路判定 → 配置级 rate。

rate 的含义是"这个配置有问题的概率"，质量分 = 1 - rate。
"""

from __future__ import annotations

from pathlib import Path

from ...app.config import Config
from ...app.types import ContractError
from .steps import step01_measure, step02_routes, step03_gate


def _configs_in(summary: dict) -> list[str]:
    names = set()
    for block in (summary.get("metrics") or {}).values():
        for row in block.get("rows", []):
            names.add(row["model"])
    return sorted(names)


def run(units: list[str], quality_dir: Path, config: Config,
        parents: dict | None = None) -> dict:
    detectors = config.checks["thresholds"]["quality"]["detectors"]
    parent_z = float(config.checks["thresholds"]["quality"]["scale"]["parent_z"])
    parents = parents or {}
    per_unit: dict[str, dict[str, dict]] = {}
    for unit in units:
        summary = step01_measure.measure(quality_dir, unit)
        by_config = {}
        for config_name in _configs_in(summary):
            flags = {name: step02_routes.evaluate(name, detector, config_name, summary,
                                                  parents, parent_z)
                     for name, detector in detectors.items()}
            by_config[config_name] = flags
        if not by_config:
            raise ContractError(f"{unit}: summary 里没有任何配置的测量值")
        per_unit[unit] = by_config
    rates = step03_gate.run(per_unit, detectors)
    return {
        "units": {unit: {config_name: {name: _flag_dict(flag) for name, flag in flags.items()}
                         for config_name, flags in by.items()}
                  for unit, by in per_unit.items()},
        "rates": {config_name: row["rate"] for config_name, row in rates.items()},
        "detail": rates,
    }


def _flag_dict(flag) -> dict:
    return {"flagged": flag.flagged, "suspect": flag.suspect, "reason": flag.reason,
            "routes": flag.routes}
