"""DINOv3 特征：主体与参考图的相似度。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.types import ContractError


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "dinov3.features.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path, allow_pickle=True)
