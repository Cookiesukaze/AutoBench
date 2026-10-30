"""接触：手握住了没有，由手部关键点和物体掩码的像素关系决定。

"握持有没有断"是关于像素的问题，不是判官能说服自己的问题。
关键点落进物体掩码里就是接触；整段都没靠近到人身高的三分之一就不是。
中间状态留给 VLM。
"""

from __future__ import annotations

from pathlib import Path

from .....app.types import Item, RuleResult
from ...experts import pose as pose_expert


def decide(item: Item, folder: Path, frames: int, checks: dict) -> RuleResult | None:
    concepts = item.concepts
    target = concepts[-1] if len(concepts) > 1 else (concepts[0] if concepts else None)
    if not target:
        return None
    pose_file = folder / "pose.keypoints.npz"
    mask_file = folder / f"sam31.{target}.masks.npz"
    if not pose_file.exists() or not mask_file.exists():
        return None

    judge_thresholds = checks["thresholds"]["judge"]
    z = pose_expert.load(folder)
    masks = pose_expert.np.load(mask_file, allow_pickle=True)
    summary = pose_expert.hand_summary(
        z, masks, [0.0, float(frames)], 24.0,
        hand_score=judge_thresholds["hand_score"], hand_reach=judge_thresholds["hand_reach"],
        touch_px=item.rule["touch_px"])
    inside = sum(summary[side]["frames_hand_point_inside_mask"] for side in ("left", "right"))
    touch = sum(summary[side]["frames_within_touch_px"] for side in ("left", "right"))
    found = sum(summary[side]["frames_with_hand_found"] for side in ("left", "right"))
    if found == 0:
        return None
    distances = [summary[s]["min_distance_px"] for s in ("left", "right")
                 if summary[s]["min_distance_px"] is not None]
    if not distances:
        return None
    measured = {"frames_hand_point_inside_object": inside,
                "frames_hand_within_touch_px": touch,
                "frames_with_hand_found": found,
                "min_distance_px": min(distances)}
    if inside >= item.rule["min_frames"] or touch >= max(3, found // 5):
        return RuleResult(status="pass", measured=measured,
                          reason="hand keypoints touch the object mask in %d frames (%d of them inside it): "
                                 "the hold is real" % (max(touch, inside), inside))
    if measured["min_distance_px"] > 250:
        return RuleResult(status="fail", measured=measured,
                          reason="no hand comes near the object at any frame (closest %d px)"
                                 % measured["min_distance_px"])
    return None
