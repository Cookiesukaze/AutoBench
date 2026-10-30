"""CoTracker3 轨迹：主体上的点怎么走。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.types import ContractError


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "cotracker3.tracks.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path)
