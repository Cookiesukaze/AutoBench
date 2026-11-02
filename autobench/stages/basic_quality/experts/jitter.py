"""背景点的短时运动残差；没有可见背景点时返回未测量。"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def measure(tracks, visibility, labels, fps):
    selected = np.array([str(v).startswith("background") for v in labels])
    if not selected.any() or len(tracks) < 14:
        return None
    shifts = np.diff(tracks[:, selected], axis=0)
    residual = shifts - gaussian_filter1d(shifts, 0.12 * fps, axis=0, mode="nearest")
    per_frame = np.mean(np.sum(residual**2, axis=-1), axis=-1)
    visible = visibility[:, selected].astype(bool).all(axis=1)
    valid = visible[:-1] & visible[1:]
    valid[:6] = False
    valid[-6:] = False
    return float(np.sqrt(np.mean(per_frame[valid]))) if valid.any() else None
