"""运镜：背景光流拟合出镜头的平移与缩放，和题目要求的对一对。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .....app.types import Item, RuleResult
from ...experts import sam as sam_expert
from ...experts import waft as waft_expert

ZOOM_TOL = 0.35        # |log zoom| 超过这个才算真变焦（约 1.4 倍）
EXCESS_TOL = 3.0       # 主体比背景放大得快这么多倍，是伪影


def _axis(value: float, want: str, positive: str, negative: str,
          minimum: float, rest: float) -> bool:
    if want == "any":
        return True
    if want == "none":
        return abs(value) <= rest
    # 方向类要求只判"反了"：慢一点、幅度小一点，只要方向对就算过关。
    if want == positive:
        return value > -minimum
    return value < minimum


def decide(item: Item, folder: Path, frames: int, checks: dict) -> RuleResult | None:
    flow_file = folder / "waft.flow.npz"
    if not flow_file.exists():
        return None
    z = waft_expert.load(folder)
    mask_file = folder / "sam31.masks.npz"
    masks = np.load(mask_file) if mask_file.exists() else None
    _, _, h, w = z["flow"].shape
    ys, xs = np.mgrid[8:h:16, 8:w:16]
    grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(float) - [w / 2, h / 2]
    bg_s, bg_t, fg_s = [], [], []
    for flow, index in zip(z["flow"], z["indices"]):
        d = flow[:, ys.ravel(), xs.ravel()].T
        near = sam_expert.mask_near(masks, int(index)) if masks is not None else None
        subject = near[ys.ravel(), xs.ravel()] if near is not None else np.zeros(len(grid), bool)
        s, t = waft_expert.fit(grid[~subject], grid[~subject] + d[~subject], ransac_iters=24)
        bg_s.append(s)
        bg_t.append(t)
        if subject.sum() >= 20:
            fg_s.append(waft_expert.fit(grid[subject], grid[subject] + d[subject])[0])

    steps = frames - 1
    pan_frac = -np.mean([t[0] for t in bg_t]) * steps / w
    pan_signs = [-t[0] for t in bg_t]
    want_positive = True if item.params["pan"] == "right" else (False if item.params["pan"] == "left" else None)
    opposite_share = None
    if want_positive is not None and pan_signs:
        opposite = [x for x in pan_signs if (x < 0 if want_positive else x > 0)]
        opposite_share = len(opposite) / len(pan_signs)
    tilt_frac = np.mean([t[1] for t in bg_t]) * steps / h
    zoom = float(np.exp(np.mean(np.log(bg_s)) * steps))
    excess = float(np.exp((np.mean(np.log(fg_s)) - np.mean(np.log(bg_s))) * steps)) if fg_s else None
    measured = {"pan_fraction_right": float(pan_frac), "opposite_direction_share": opposite_share,
                "tilt_fraction_up": float(tilt_frac), "zoom_ratio": zoom,
                "subject_over_background_scale": excess, "flow_pairs": len(bg_s)}

    limits = item.rule
    still, move = limits["still_max_fraction"], limits["pan_tilt_min_fraction"]
    failed = []
    if not _axis(pan_frac, item.params["pan"], "right", "left", move, still):
        failed.append("pan")
    if not _axis(tilt_frac, item.params["tilt"], "up", "down", move, still):
        failed.append("tilt")
    if item.params["zoom"] in ("in", "out"):
        if not _axis(np.log(zoom), item.params["zoom"], "in", "out",
                     np.log(limits["zoom_min_ratio"]), np.log(1 + still)):
            failed.append("zoom")
    elif abs(np.log(zoom)) > ZOOM_TOL:
        failed.append("zoom")
    if item.params["position"] == "fixed" and excess is not None and excess > EXCESS_TOL:
        failed.append("position")

    if failed:
        return RuleResult(status="fail", measured=measured, failed=tuple(failed), review=True)
    if opposite_share is not None and opposite_share >= 0.3 and abs(pan_frac) > still:
        return RuleResult(status="partial", measured=measured,
                          reason="方向到了，但整段里有 %.0f%% 在往反方向移" % (100 * opposite_share))
    return RuleResult(status="pass", measured=measured)
