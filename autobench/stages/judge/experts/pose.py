"""RTMW 关键点：全身 133 点与左右手。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.data.types import ContractError

# COCO-WholeBody：手腕 9（人自己的左手）/ 10（右手），手部点 91-111 左、112-132 右
HAND_POINTS = {
    "left": [9] + list(range(91, 112)),
    "right": [10] + list(range(112, 133)),
}


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "pose.keypoints.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path)


def hand_summary(
    z,
    masks,
    window,
    fps: float,
    hand_score: float,
    hand_reach: float,
    touch_px: float,
    step: int = 2,
) -> dict:
    """每只手到物体掩码的距离，按窗口统计。只报数，不判断哪只手在干活。"""
    from .sam import mask_near

    kp, sc, valid = z["keypoints"], z["scores"], z["valid_person"]
    first, last = (
        max(0, round(window[0] * fps)),
        min(len(kp) - 1, round(window[1] * fps)),
    )
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
                height = float(
                    z["boxes"][frame, person, 3] - z["boxes"][frame, person, 1]
                )
                if sc[frame, person, wrist] < hand_score or not height > 0:
                    continue
                near_wrist = (
                    np.hypot(*(kp[frame, person, ids] - kp[frame, person, wrist]).T)
                    <= hand_reach * height
                )
                ok = [
                    i
                    for i, close in zip(ids, near_wrist)
                    if close and sc[frame, person, i] >= hand_score
                ]
                if not ok:
                    continue
                pts = kp[frame, person, ok]
                inside = bool(
                    mask[
                        np.clip(pts[:, 1].round().astype(int), 0, mask.shape[0] - 1),
                        np.clip(pts[:, 0].round().astype(int), 0, mask.shape[1] - 1),
                    ].any()
                )
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
    out["note"] = (
        "left and right are the own hands of the person; "
        "distance is pixels from hand keypoints to the object mask"
    )
    return out


from ....app.remote.gpu.worker import cached, decode, sha


def pose(a, modelpath, source, torch):
    """RTMW wholebody inference; COCO-WholeBody indices preserve left/right semantics."""
    import numpy as np

    np.bool = bool  # chumpy, a mmpose dependency, still reads the removed alias
    from mmpose.apis import MMPoseInferencer

    def build():
        config = (
            source
            / "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py"
        )
        checkpoint = (
            modelpath
            / "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
        )
        detector = source / "demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py"
        detector_weights = (
            modelpath / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
        )
        inferencer = MMPoseInferencer(
            pose2d=str(config),
            pose2d_weights=str(checkpoint),
            det_model=str(detector),
            det_weights=str(detector_weights),
            det_cat_ids=[0],
            device="cuda:0",
        )
        return inferencer  # mmengine inferencers build their model in eval mode

    inferencer = cached("rtmw_x", build)
    frames, fps = decode(a.video)
    # Directory-backed frames avoid a second video decoder and ensure every decoded source frame is used.
    import tempfile, cv2

    with tempfile.TemporaryDirectory(prefix="rtmw_frames_") as tmp:
        paths = []
        for i, frame in enumerate(frames):
            path = Path(tmp) / ("%06d.jpg" % i)
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths.append(str(path))
        keypoints = []
        scores = []
        boxes = []
        frame_ids = []
        for i, path in enumerate(paths):
            result = next(inferencer(path, return_vis=False))["predictions"][0]
            persons = []
            for pred in result:
                kp = np.asarray(pred["keypoints"], dtype=np.float32).reshape(-1, 2)
                sc = np.asarray(pred["keypoint_scores"], dtype=np.float32).reshape(-1)
                bbox = np.asarray(
                    pred.get("bbox", [0, 0, 0, 0]), dtype=np.float32
                ).reshape(-1)[:4]
                if len(kp) != 133 or len(sc) != 133:
                    raise ValueError("RTMW did not return 133 keypoints")
                persons.append((kp, sc, bbox))
            keypoints.append([x[0] for x in persons])
            scores.append([x[1] for x in persons])
            boxes.append([x[2] for x in persons])
            frame_ids.append(i)
    # Variable people/frame counts use padded arrays plus explicit valid-person mask.
    n = max((len(x) for x in keypoints), default=0)
    if n == 0:
        raise ValueError("RTMW found no people in any frame")
    kp = np.full((len(frames), n, 133, 2), np.nan, np.float32)
    sc = np.full((len(frames), n, 133), np.nan, np.float32)
    bb = np.full((len(frames), n, 4), np.nan, np.float32)
    valid = np.zeros((len(frames), n), bool)
    for i, row in enumerate(keypoints):
        for j, (p, s, b) in enumerate(zip(row, scores[i], boxes[i])):
            kp[i, j] = p
            sc[i, j] = s
            bb[i, j] = b
            valid[i, j] = True
    artifact = a.output.with_suffix(".pose.npz")
    np.savez_compressed(
        artifact,
        keypoints=kp,
        scores=sc,
        boxes=bb,
        valid_person=valid,
        fps=fps,
        frame_indices=np.asarray(frame_ids),
    )
    return {
        "artifact": str(artifact),
        "artifact_sha256": sha(artifact),
        "frames": len(frames),
        "fps": fps,
        "keypoint_count": 133,
        "person_detections": int(valid.sum()),
        "person_track_ids": None,
        "limitation": "per-frame detector ordering is not persistent identity; track IDs are intentionally absent until validated matching is implemented",
    }
