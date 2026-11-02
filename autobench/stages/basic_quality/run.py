"""阶段入口：读测量 → 三路判定 → 配置级 rate。

rate 仅汇总第一关告警，不作为正式画质分。
"""

from __future__ import annotations

from pathlib import Path

from ...app.config import Config
from ...app.data.types import ContractError
from .steps import step01_measure, step02_three_routes, step03_gate


def _configs_in(summary: dict) -> list[str]:
    names = set()
    for block in (summary.get("metrics") or {}).values():
        for row in block.get("rows", []):
            names.add(row["model"])
    return sorted(names)


def run(
    units: list[str],
    quality_dir: Path,
    config: Config,
    parents: dict | None = None,
    expected: dict | None = None,
) -> dict:
    detectors = config.checks["thresholds"]["quality"]["detectors"]
    parent_z = float(config.checks["thresholds"]["quality"]["scale"]["parent_z"])
    parents = parents or {}
    per_unit: dict[str, dict[str, dict]] = {}
    for unit in units:
        summary = step01_measure.measure(quality_dir, unit)
        by_config = {}
        configs = (
            [c for c, us in expected.items() if unit in us]
            if expected is not None
            else _configs_in(summary)
        )
        for config_name in configs:
            flags = {
                name: step02_three_routes.evaluate(
                    name, detector, config_name, summary, parents, parent_z
                )
                for name, detector in detectors.items()
            }
            by_config[config_name] = flags
        # 绝对检测已确认一半配置异常时，组内分布不再是可靠的参照。
        for name, detector in detectors.items():
            known = [
                by_config[c][name].routes["absolute"]["flagged"]
                for c in configs
                if by_config[c][name].routes["absolute"]["flagged"] is not None
            ]
            if (
                len(known) * 2 >= len(configs)
                and known
                and sum(known)
                >= config.checks["thresholds"]["quality"]["scale"]["bad_share"]
                * len(known)
            ):
                for c in configs:
                    routes = dict(by_config[c][name].routes)
                    routes["group"] = {
                        "flagged": None,
                        "evidence": {},
                        "reason": "组内过半被绝对路标为异常",
                    }
                    by_config[c][name] = step02_three_routes.decide(
                        name, detector, routes
                    )
        if not by_config:
            raise ContractError(f"{unit}: summary 里没有任何配置的测量值")
        per_unit[unit] = by_config
    rates = step03_gate.run(per_unit, detectors)
    return {
        "units": {
            unit: {
                config_name: {name: _flag_dict(flag) for name, flag in flags.items()}
                for config_name, flags in by.items()
            }
            for unit, by in per_unit.items()
        },
        "rates": {config_name: row["rate"] for config_name, row in rates.items()},
        "detail": rates,
        "admission": {
            c: {
                "allowed": row["rate"] is not None,
                "reason": "measured" if row["rate"] is not None else "unmeasured",
            }
            for c, row in rates.items()
        },
        "per_video_admission": {
            c: {
                u: {
                    "allowed": step03_gate.joint(by[c], detectors) is not None,
                    "reason": "measured"
                    if step03_gate.joint(by[c], detectors) is not None
                    else "unmeasured",
                }
                for u, by in per_unit.items()
                if c in by
            }
            for c in rates
        },
    }


def _flag_dict(flag) -> dict:
    return {
        "flagged": flag.flagged,
        "suspect": flag.suspect,
        "reason": flag.reason,
        "routes": flag.routes,
    }
