"""WAFT 光流：背景的运动场，用来分离镜头与主体。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.types import ContractError


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "waft.flow.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path)


def fit(points: np.ndarray, moved: np.ndarray, ransac_iters: int = 0) -> tuple[float, np.ndarray]:
    """相似变换 moved = s·R·points + t，在中心化坐标上解。

    ransac_iters>0 时先跑一小轮 RANSAC，把少数不一致的点丢掉：背景掩码里混进来的前景运动
    会把镜头估计带偏。
    """
    pc, mc = points - points.mean(0), moved - moved.mean(0)

    def solve(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, np.ndarray]:
        a = np.sum(src * dst) / np.sum(src ** 2)
        b = np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0]) / np.sum(src ** 2)
        return a, b, dst.mean(0) - src.mean(0)

    if ransac_iters and len(points) >= 12:
        rng = np.random.default_rng(0)
        best_inliers = None
        for _ in range(ransac_iters):
            idx = rng.choice(len(pc), size=max(8, len(pc) // 2), replace=False)
            a, b, t = solve(pc[idx], mc[idx])
            rot = np.array([[a, -b], [b, a]])
            resid = np.linalg.norm(pc @ rot.T + t - mc, axis=1)
            inliers = resid < 3.0
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
        if best_inliers is not None and best_inliers.sum() >= 8:
            pc, mc = pc[best_inliers], mc[best_inliers]
    a, b, t = solve(pc, mc)
    return float(np.hypot(a, b)), t
