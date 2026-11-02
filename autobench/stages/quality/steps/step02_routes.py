"""三路判定：绝对线 / 与原版比 / 与同题同 seed 的组比。

一个检测器只有在「三条路至少一路测到」时才可能有结论；全都测不到就是未测量，
不是干净。方向（越高越差 / 越低越差）和所有线都来自 config。
"""

from __future__ import annotations

from typing import Any

from ....app.types import Flag
from .step01_measure import metric_block, metric_rows

GEN_ARTIFACT = "gen_artifact"


def _direction(detector: dict) -> int:
    return 1 if detector["worse_when"] == "higher" else -1


def _route(flagged: bool | None, evidence: dict | None = None, reason: str = "") -> dict:
    return {"flagged": flagged, "evidence": evidence or {}, "reason": reason}


def _absolute(name: str, detector: dict, config: str, summary: dict) -> dict:
    limit = detector.get("absolute")
    if limit is None:
        return _route(None, reason="这个检测器没有标定的绝对线")
    if name == GEN_ARTIFACT:
        return _route(None, reason="gen_artifact 的线在组内尺度上，没有绝对线")
    if name == "human_structure":
        return _route(None, reason="人体结构的绝对判定还没实现（要读 HADM 逐帧输出）")
    rows = metric_rows(summary, detector.get("metric", name))
    row = next((r for r in rows or [] if r["model"] == config), None)
    if row is None:
        return _route(None, reason="这个配置没有测量值")
    direction = _direction(detector)
    return _route(direction * row["value"] > direction * limit,
                  {"value": row["value"], "limit": limit})


def _gen_artifact(name: str, detector: dict, config: str, summary: dict, z: float | None,
                  route: str) -> dict:
    if z is None:
        return _route(None, reason="gen_artifact 的 warp 没有组内 z")
    burst = metric_rows(summary, "gen_artifact_burst_share")
    burst_row = next((r for r in burst or [] if r["model"] == config), None)
    if burst_row is None:
        return _route(None, reason="gen_artifact 的 burst_share 没有测到")
    rule = detector["absolute"]
    strong = z > rule["strong_warp"]
    weak = z > rule["weak_warp"] and burst_row["value"] > rule["burst_share"]
    return _route(bool(strong or weak),
                  {"warp_robust_z": round(z, 3), "burst_share": burst_row["value"],
                   "strong_line": strong, "weak_line": weak, "rule": rule, "route": route})


def _parent(name: str, detector: dict, config: str, summary: dict, parents: dict,
            parent_z: float) -> dict:
    parent = parents.get(config)
    if parent is None:
        return _route(None, reason="这个配置没有声明原版")
    if parent == config:
        return _route(None, reason="这个配置就是原版")
    block = metric_block(summary, detector.get("metric", name))
    if block is None:
        return _route(None, reason="这道题没有组内测量")
    if block.get("status") == "unavailable":
        return _route(None, reason=f'组内刻度不可用：{block.get("reason")}')
    scale, transform = block.get("log_scale"), block.get("transform", "log")
    values = {r["model"]: r["value"] for r in block["rows"]}
    if config not in values or parent not in values:
        return _route(None, reason=f"{config} 或原版 {parent} 没有测量值")
    import math
    fn = {"log": math.log, "log1p": math.log1p}.get(transform, lambda x: x)
    direction = _direction(detector)
    z = direction * (fn(values[config]) - fn(values[parent])) / scale
    if name == GEN_ARTIFACT:
        return _gen_artifact(name, detector, config, summary, z, "parent")
    # 与原版比较同样用稳健 z 的 3 倍线：和组内一路保持同一个尺度。
    return _route(z >= parent_z, {"robust_z_vs_parent": z, "original": parent,
                             "value": values[config], "original_value": values[parent]})


def _group(name: str, detector: dict, config: str, summary: dict) -> dict:
    block = metric_block(summary, detector.get("metric", name))
    if block is None:
        return _route(None, reason="这道题没有组内测量")
    if block.get("status") == "unavailable":
        return _route(None, reason=f'组内刻度不可用：{block.get("reason")}')
    row = next((r for r in block["rows"] if r["model"] == config), None)
    if row is None:
        return _route(None, reason="这个配置没有测量值")
    if name == GEN_ARTIFACT:
        return _gen_artifact(name, detector, config, summary, row.get("robust_log_z"), "group")
    if "flagged" not in row:
        return _route(None, reason="这一行没有组内判定")
    line = detector.get("group_z")
    flagged = bool(row["flagged"]) or (line is not None and row.get("robust_log_z", -99) >= line)
    return _route(flagged, {k: row[k] for k in ("value", "robust_log_z", "screening_score") if k in row})


def decide(name: str, detector: dict, routes: dict[str, dict]) -> Flag:
    """三条路合成一个结论。absolute 命中即定；两条相对路都命中才算；单路定罪名单例外。"""
    measured = {k: v for k, v in routes.items() if v["flagged"] is not None}
    if not measured:
        return Flag(name, None, {"routes": routes},
                    reason="; ".join(f'{k}: {v["reason"]}' for k, v in routes.items()))
    absolute = routes["absolute"]["flagged"]
    relatives = [routes[k]["flagged"] for k in ("parent", "group")
                 if routes[k]["flagged"] is not None]
    group_flagged = routes["group"]["flagged"] or False
    flagged = (bool(absolute) or (len(relatives) == 2 and all(relatives))
               or bool(detector.get("alone") and group_flagged))
    return Flag(name, flagged, {"routes": routes}, suspect=not flagged and any(relatives))


def evaluate(name: str, detector: dict, config: str, summary: dict, parents: dict,
             parent_z: float) -> Flag:
    routes = {"absolute": _absolute(name, detector, config, summary),
              "parent": _parent(name, detector, config, summary, parents, parent_z),
              "group": _group(name, detector, config, summary)}
    return decide(name, detector, routes)
