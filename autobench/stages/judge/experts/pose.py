"""RTMW 关键点：全身 133 点与左右手。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.types import ContractError

# COCO-WholeBody：手腕 9（人自己的左手）/ 10（右手），手部点 91-111 左、112-132 右
HAND_POINTS = {"left": [9] + list(range(91, 112)), "right": [10] + list(range(112, 133))}


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "pose.keypoints.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path)


def hand_summary(z, masks, window, fps: float, hand_score: float, hand_reach: float,
                 touch_px: float, step: int = 2) -> dict:
    """每只手到物体掩码的距离，按窗口统计。只报数，不判断哪只手在干活。"""
    from .sam import mask_near

    kp, sc, valid = z["keypoints"], z["scores"], z["valid_person"]
    first, last = max(0, round(window[0] * fps)), min(len(kp) - 1, round(window[1] * fps))
    rows: dict[str, list] = {"left": [], "right": []}
    for frame in range(first, last + 1, step):
        mask = mask_near(masks, frame)
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        for side, ids in HAND_POINTS.items():
            best = None
            for person in np.flatnonzero(valid[frame]):
                wrist = ids[0]
                height = float(z["boxes"][frame, person, 3] - z["boxes"][frame, person, 1])
                if sc[frame, person, wrist] < hand_score or not height > 0:
                    continue
                near_wrist = np.hypot(*(kp[frame, person, ids] - kp[frame, person, wrist]).T) <= hand_reach * height
                ok = [i for i, close in zip(ids, near_wrist) if close and sc[frame, person, i] >= hand_score]
                if not ok:
                    continue
                pts = kp[frame, person, ok]
                inside = bool(mask[np.clip(pts[:, 1].round().astype(int), 0, mask.shape[0] - 1),
                                   np.clip(pts[:, 0].round().astype(int), 0, mask.shape[1] - 1)].any())
                near = min(float(np.min(np.hypot(xs - x, ys - y))) for x, y in pts[::4])
                if best is None or near < best[0]:
                    best = (near, inside)
            rows[side].append(best)

    out = {}
    for side, got in rows.items():
        seen = [g for g in got if g is not None]
        out[side] = {
            "frames_checked": len(got),
            "frames_with_hand_found": len(seen),
            "min_distance_px": min((g[0] for g in seen), default=None),
            "mean_distance_px": float(np.mean([g[0] for g in seen])) if seen else None,
            "frames_hand_point_inside_mask": sum(g[1] for g in seen),
            "frames_within_touch_px": sum(1 for g in seen if g[0] <= touch_px),
        }
    out["note"] = ("left and right are the own hands of the person; "
                   "distance is pixels from hand keypoints to the object mask")
    return out
