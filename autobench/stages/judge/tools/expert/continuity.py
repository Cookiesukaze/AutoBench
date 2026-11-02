"""连续性：主体的点在相邻帧之间有没有突然跳开。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .....app.data.types import Item, RuleResult
from ...experts import cotracker as cotracker_expert
from ...experts import sam as sam_expert


def decide(item: Item, folder: Path, meta: dict, checks: dict) -> RuleResult | None:
    tracks_file, mask_file = folder / "cotracker3.tracks.npz", folder / "sam31.masks.npz"
    if not tracks_file.exists() or not mask_file.exists():
        return None
    limit = item.rule["jump_px"]
    z, masks = cotracker_expert.load(folder), sam_expert.load_main(folder)
    tr, vis = z["tracks"], z["visibility"]
    keys = sam_expert.frame_keys(masks)
    on, seen = np.zeros(tr.shape[1]), np.zeros(tr.shape[1])
    for k in keys:
        m = sam_expert.mask_near(masks, k)
        if m is None:
            continue
        x = np.clip(tr[k, :, 0].round().astype(int), 0, m.shape[1] - 1)
        y = np.clip(tr[k, :, 1].round().astype(int), 0, m.shape[0] - 1)
        seen += vis[k]
        on += vis[k] & m[y, x]
    subject = on > 0.5 * np.maximum(seen, 1)
    if subject.sum() < 1:
        return None
    tr, vis = tr[:, subject], vis[:, subject]
    labels = z["labels"][subject]
    step = np.diff(tr, axis=0)
    both = vis[1:] & vis[:-1]
    median = np.median(np.diff(z["tracks"], axis=0), axis=1, keepdims=True)
    jump = np.linalg.norm(step - median, axis=2) * both
    worst = np.unravel_index(int(np.argmax(jump)), jump.shape)
    value = float(jump[worst])
    return RuleResult(
        status="fail" if value > limit else "pass",
        measured={"max_relative_jump_px": value, "frame": int(worst[0]) + 1,
                  "point_label": str(labels[worst[1]]), "subject_points": int(subject.sum())},
    )
