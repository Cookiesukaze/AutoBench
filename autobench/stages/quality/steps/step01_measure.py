"""测量：拿到每条视频每个通道的原始数字。

这一版只读归档的 summary.json——真实跑通道（MUSIQ/DOVER/抖动…）要 GPU 模型，
还没搬过来。summary 里每个通道是一块 {'rows': [{'model': 配置, 'value': 数}], ...}，
组内刻度（robust_log_z / screening_score / flagged）如果已经算好就直接用。
"""

from __future__ import annotations

import json
from pathlib import Path

from ....app.types import ContractError

METRIC_BLOCKS = ("metrics",)


def measure(quality_dir: Path, unit: str) -> dict:
    path = quality_dir / unit / "summary.json"
    if not path.exists():
        raise ContractError(f"没有 {path}：这一版画质只读归档 summary，真实测量尚未实现")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary.get("metrics"), dict):
        raise ContractError(f"{path} 里没有 metrics 块")
    return summary


def metric_rows(summary: dict, metric: str) -> list[dict] | None:
    block = (summary.get("metrics") or {}).get(metric)
    if not isinstance(block, dict) or "rows" not in block:
        return None
    return block["rows"]


def metric_block(summary: dict, metric: str) -> dict | None:
    block = (summary.get("metrics") or {}).get(metric)
    return block if isinstance(block, dict) and "rows" in block else None
