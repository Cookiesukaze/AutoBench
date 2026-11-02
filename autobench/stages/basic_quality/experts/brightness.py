"""亮度台阶原始指标；时间来自实际采样帧。"""

import numpy as np


def brightness_step(frames, times):
    if len(frames) < 4:
        return {
            "brightness_step_ratio": None,
            "brightness_step_relative": None,
            "brightness_step_at": None,
        }
    levels = np.array([float(f.mean()) for f in frames])
    changes = np.abs(np.diff(levels))
    at = int(np.argmax(changes))
    scale = float(np.median(np.delete(changes, at))) + 1e-6
    after = levels[at + 1 :]
    holds = abs(float(np.median(after)) - levels[at + 1]) <= 0.35 * max(
        1.0, abs(levels[at + 1])
    )
    return {
        "brightness_step_ratio": float(changes[at] / scale) if holds else 0.0,
        "brightness_step_relative": float(changes[at] / max(1.0, levels[at]))
        if holds
        else 0.0,
        "brightness_step_at": times[at + 1] if holds else None,
    }
