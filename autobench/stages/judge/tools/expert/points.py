"""Points for CoTracker, taken from SAM masks, so no model has to guess coordinates.

Every point is a query (frame, x, y). A thing that is already in the first frame gets its points
there; a thing that enters later (a person from the edge, an object brought in) gets them on the first
sampled frame where its mask appears. Subject points sit inside the masks, eroded so they stay off the
outline; background points are strong corners outside every mask. The same procedure on the same masks
gives the same points, so configurations of one question can share them.
"""

import json
from pathlib import Path

import cv2
import numpy as np

MIN_MASK_PIXELS = 150  # a mask smaller than this is noise, not an appearance
ERODE_PX = 5
BACKGROUND_GAP_PX = 40


def mask_frames(npz):
    return sorted(int(k[6:]) for k in npz.files if k.startswith("masks_"))


def first_appearance(npz, minimum=MIN_MASK_PIXELS):
    """(frame, union mask) of the first stored frame where the concept has a mask big enough, else None."""
    for frame in mask_frames(npz):
        masks = npz["masks_%d" % frame]
        if masks.ndim == 3 and len(masks):
            union = masks.any(0)
            if union.sum() >= minimum:
                return frame, union
    return None


def spread(points, n):
    """Farthest-point selection: n points far from one another, deterministic."""
    points = np.asarray(points, dtype=float)
    if len(points) <= n:
        return points
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < n:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(
            distance, np.linalg.norm(points - points[chosen[-1]], axis=1)
        )
    return points[chosen]


def sample_in_mask(mask, n):
    """n (x, y) pixels inside the mask, away from its outline, spread over it."""
    core = cv2.erode(
        mask.astype(np.uint8), np.ones((2 * ERODE_PX + 1,) * 2, np.uint8)
    ).astype(bool)
    region = core if core.sum() >= n else mask
    ys, xs = np.nonzero(region)
    if not len(xs):
        return np.zeros((0, 2))
    keep = np.arange(
        0, len(xs), max(1, len(xs) // 4000)
    )  # keep the selection cheap on big masks
    return spread(np.stack([xs[keep], ys[keep]], 1), n)


def sample_background(gray, excluded, n, gap=BACKGROUND_GAP_PX):
    """n corner points outside `excluded`, at least `gap` pixels apart."""
    allowed = (
        ~cv2.dilate(excluded.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=n * 6, qualityLevel=0.01, minDistance=gap, mask=allowed
    )
    if corners is None:
        return np.zeros((0, 2))
    return corners.reshape(-1, 2)[:n]


def build_targets(frame0_gray, masks, subjects, n_subject=6, n_background=8):
    """CoTracker queries from SAM masks.

    masks: {concept: opened npz}. subjects: the concepts to follow. Returns (points, notes) where each
    point is {frame, x, y (normalised to the frame size), label, role}; notes lists what could not be placed.
    """
    h, w = frame0_gray.shape
    points, notes, union0 = [], [], np.zeros((h, w), bool)
    for concept in subjects:
        found = first_appearance(masks[concept]) if concept in masks else None
        if found is None:
            notes.append(f"{concept}: no mask in any stored frame, no points placed")
            continue
        frame, mask = found
        if frame == 0:
            union0 |= mask
        for x, y in sample_in_mask(mask, n_subject):
            points.append(
                {
                    "frame": frame,
                    "x": float(x) / (w - 1),
                    "y": float(y) / (h - 1),
                    "label": f"subject: {concept}",
                    "role": "subject",
                }
            )
    for x, y in sample_background(frame0_gray, union0, n_background):
        points.append(
            {
                "frame": 0,
                "x": float(x) / (w - 1),
                "y": float(y) / (h - 1),
                "label": "background: corner",
                "role": "background",
            }
        )
    if not any(p["role"] == "background" for p in points):
        notes.append("no background corners found outside the masks")
    return points, notes


def write_targets(points, path):
    Path(path).write_text(
        json.dumps(
            {
                "points": [
                    {k: p[k] for k in ("frame", "x", "y", "label")} for p in points
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def basic_quality_config(points, question):
    """The first-frame points in the layout run_shared_motion.py reads. Later-entering subjects are left out:
    that script tracks from frame 0 only."""
    return {
        "schema_version": 1,
        "question": question,
        "coordinate_system": "normalized frame",
        "planning_origin": "sampled from SAM masks and image corners by points.py; shared by all configurations",
        "points": [
            {"x": p["x"], "y": p["y"], "role": p["role"], "label": p["label"]}
            for p in points
            if p["frame"] == 0
        ],
        "unavailable": {},
    }
