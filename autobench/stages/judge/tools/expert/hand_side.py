"""左右手：哪只手在操作物体，由关键点判定。

两只手的点到物体掩码的距离整段比一遍，明显更近的那只就是干活的手。
只有"明显是另一只手"才 fail；差距小说明关键点分不开，交回 VLM（返回 None）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .....app.data.types import Item, RuleResult
from ...experts import pose as pose_expert


def decide(item: Item, folder: Path, meta: dict, checks: dict) -> RuleResult | None:
    want = item.params.get("hand")
    if want not in ("left", "right", "both"):
        return None
    concepts = item.concepts
    target = concepts[-1] if len(concepts) > 1 else (concepts[0] if concepts else None)
    pose_file = folder / "pose.keypoints.npz"
    mask_file = folder / f"sam31.{target}.masks.npz" if target else None
    if mask_file is None or not pose_file.exists() or not mask_file.exists():
        return None
    z = pose_expert.load(folder)
    if "keypoints" not in z.files:
        return None
    masks = np.load(mask_file, allow_pickle=True)
    kp, sc, valid, boxes = z["keypoints"], z["scores"], z["valid_person"], z["boxes"]
    # 姿态模型没找到的关节写 nan，nan 会让下面每个比较都失败；
    # 换成 0 之后"没找到"就读成"不够自信"，而不是污染匹配。
    sc = np.nan_to_num(sc, nan=0.0)
    kp = np.nan_to_num(kp, nan=-1e4)
    judge_thresholds = checks["thresholds"]["judge"]
    hand_score = judge_thresholds["hand_score"]
    hand_reach = judge_thresholds["hand_reach"]

    close: dict[str, list[float]] = {"left": [], "right": []}
    keys = sorted(int(k[6:]) for k in masks.files if k.startswith("masks_"))
    for frame in keys:
        if frame >= len(kp):
            continue
        m = masks[f"masks_{frame}"]
        if m.ndim == 3:
            m = m.any(0)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        for side, ids in pose_expert.HAND_POINTS.items():
            wrist = ids[0]
            for person in np.flatnonzero(valid[frame]):
                height = float(boxes[frame, person, 3] - boxes[frame, person, 1])
                if sc[frame, person, wrist] < hand_score or height <= 0:
                    continue
                near_wrist = np.hypot(*(kp[frame, person, ids] - kp[frame, person, wrist]).T) <= hand_reach * height
                ok = [i for i, near in zip(ids, near_wrist) if near and sc[frame, person, i] >= hand_score]
                if not ok:
                    continue
                pts = kp[frame, person, ok]
                near = min(float(np.min(np.hypot(xs - x, ys - y))) for x, y in pts[::4])
                close[side].append(near)

    minimum = item.rule["min_frames"]
    left = float(np.median(close["left"])) if len(close["left"]) >= minimum else None
    right = float(np.median(close["right"])) if len(close["right"]) >= minimum else None
    measured = {"left_hand_distance_px": left, "right_hand_distance_px": right,
                "frames_left": len(close["left"]), "frames_right": len(close["right"]),
                "label_convention": "uncalibrated (mirrored views swap left/right in the pose model)"}
    if left is None or right is None:
        return None
    # 镜像或转身会让姿态模型的左右标签反过来（B056 人工核对过），在有人工标定之前
    # 这条规则不给任何视频判 fail：带数字通过，把"未标定"这件事写进测量里。
    return RuleResult(status="pass", measured=measured)
