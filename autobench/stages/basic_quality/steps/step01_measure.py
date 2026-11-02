"""测量：拿到每条视频每个通道的原始数字。

CPU 和远端 GPU 专家产生原始记录，再聚合为 summary.json。
summary 里每个通道是一块 {'rows': [{'model': 配置, 'value': 数}], ...}，
组内刻度（robust_log_z / screening_score / flagged）如果已经算好就直接用。
"""

from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path

from ....app.data.types import ContractError

METRIC_BLOCKS = ("metrics",)


def measure(quality_dir: Path, unit: str) -> dict:
    path = quality_dir / unit / "summary.json"
    if not path.exists():
        raise ContractError(f"没有 {path}：请先通过 batch 测量，或指定完整测量归档")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary.get("metrics"), dict):
        raise ContractError(f"{path} 里没有 metrics 块")
    for name, block in summary["metrics"].items():
        rows = block.get("rows", [])
        names = [r["model"] for r in rows]
        if len(set(names)) != len(names):
            raise ContractError(f"{path}: {name} 有重复配置")
        if any(
            isinstance(r.get("value"), bool)
            or not isinstance(r.get("value"), (int, float))
            or not math.isfinite(r["value"])
            for r in rows
        ):
            raise ContractError(f"{path}: {name} 有无效测量值")
    return summary


def metric_rows(summary: dict, metric: str) -> list[dict] | None:
    block = (summary.get("metrics") or {}).get(metric)
    if not isinstance(block, dict) or "rows" not in block:
        return None
    return block["rows"]


def metric_block(summary: dict, metric: str) -> dict | None:
    block = (summary.get("metrics") or {}).get(metric)
    return block if isinstance(block, dict) and "rows" in block else None


def screen_rows(rows, direction, scale, transform="log1p"):
    """同题同 seed 的稳健组内刻度，保留原始观测及不可用原因。"""
    finite = [
        r
        for r in rows
        if isinstance(r.get("value"), (float, int))
        and not isinstance(r["value"], bool)
        and math.isfinite(r["value"])
    ]
    out = {"rows": finite, "transform": transform}
    if len(finite) < scale["min_configs"]:
        return dict(out, status="unavailable", reason="组内有效配置不足")
    values = np.array([r["value"] for r in finite], dtype=float)
    if (
        transform == "log"
        and np.any(values <= 0)
        or transform == "log1p"
        and np.any(values < 0)
    ):
        return dict(out, status="unavailable", reason="测量值不满足变换范围")
    xs = (
        np.log(values)
        if transform == "log"
        else np.log1p(values)
        if transform == "log1p"
        else values
    )
    center = float(np.median(xs))
    spread = float(np.median(np.abs(xs - center))) / scale["mad_to_sigma"]
    if spread < 1e-8:
        return dict(out, status="unavailable", reason="组内没有有效离散度")
    sign = 1 if direction == "higher" else -1
    for r, x in zip(finite, xs):
        z = sign * (x - center) / spread
        score = 100 / (1 + math.exp(-max(-700, min(700, z - scale["score_center_z"]))))
        r.update(
            robust_log_z=z,
            screening_score=score,
            flagged=score >= scale["screen_score"],
        )
    return dict(out, log_scale=spread)


def build_summaries(tasks, output, checks):
    """专家原始记录/数组 → 每个 unit 的 summary；不重新判定语义。"""
    from ....app.data.store import Store
    from ..experts.motion import summarize
    from ..experts.jitter import measure as jitter

    grouped = {}
    detectors = checks["thresholds"]["quality"]["detectors"]
    scale = checks["thresholds"]["quality"]["scale"]
    directions = {d.get("metric", n): d["worse_when"] for n, d in detectors.items()}
    for task in tasks:
        unit, name, folder = task["unit"], task["config"], Path(task["folder"])
        group = grouped.setdefault(
            unit, {"metrics": {}, "errors": [], "inputs": {}, "unit": unit}
        )
        group["inputs"][name] = task.get("video_sha256")
        metrics = {}
        for expert in ("signal", "musiq", "hadm", "dover"):
            record = folder / (expert + ".json")
            if record.exists():
                row = json.loads(record.read_text(encoding="utf-8"))
                if row.get("status") == "observed":
                    metrics.update(row.get("observations", {}).get("metrics", {}))
                else:
                    group["errors"].append(
                        {"model": name, "expert": expert, "error": row.get("error")}
                    )
        motion = folder / "quality_motion.tracks.npz"
        if motion.exists():
            with np.load(motion) as z:
                tr, vis = z["tracks"], z["visibility"].astype(bool)
                labels = [str(v) for v in z["labels"]]
                sub = np.array([v.startswith("subject") for v in labels])
                metrics["camera_jitter"] = jitter(tr, vis, labels, float(z["fps"]))
                if sub.any():
                    try:
                        metrics["subject_motion"] = summarize(tr[:, sub], vis[:, sub])[
                            "shape_change_percent"
                        ]
                    except ValueError as exc:
                        group["errors"].append(
                            {"model": name, "expert": "motion", "error": str(exc)}
                        )
        for metric, value in metrics.items():
            if (
                value is not None
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                group["metrics"].setdefault(metric, {"rows": []})["rows"].append(
                    {"model": name, "value": value}
                )
        hadm = folder / "hadm.frames.jsonl"
        if hadm.exists():
            frames = [
                json.loads(line)
                for line in hadm.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rule = detectors["human_structure"]["absolute"]
            run = best = 0
            for frame in frames:
                run = run + 1 if frame["maximum"] >= rule["confidence"] else 0
                best = max(best, run)
            step = (
                frames[1]["seconds"] - frames[0]["seconds"] if len(frames) > 1 else 0.0
            )
            block = group["metrics"].setdefault("human_structure", {"rows": []})
            row = next((r for r in block["rows"] if r["model"] == name), None)
            if row is None:
                row = {
                    "model": name,
                    "value": sum(f["maximum"] >= rule["confidence"] for f in frames),
                }
                block["rows"].append(row)
            row["longest_run_s"] = best * step
    for unit, summary in grouped.items():
        for name, block in list(summary["metrics"].items()):
            transform = (
                "raw"
                if name == "gen_artifact_warp"
                else "log"
                if name in ("camera_jitter", "subject_motion")
                else "log1p"
            )
            summary["metrics"][name] = screen_rows(
                block["rows"], directions.get(name, "higher"), scale, transform
            )
        Store.write_json(Path(output) / unit / "summary.json", summary)
    return output
