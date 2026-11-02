"""把专家数组整理成给 VLM 看的数字：每个检查项，只在它自己的时间窗里统计。

数字只说明"量到了什么"，不说明"这算不算错"。语义判断留给 steps。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.types import Item, Plan
from ..tools.expert import camera as camera_rule
from ..tools.expert import contact as contact_rule
from ..tools.expert import continuity as continuity_rule
from ..tools.expert import hand_side as hand_side_rule
from ..tools.expert import persistence as persistence_rule
from . import pose as pose_expert
from . import sam as sam_expert

RULES = {"camera": camera_rule.decide, "continuity": continuity_rule.decide,
         "persistence": persistence_rule.decide, "contact": contact_rule.decide,
         "hand_side": hand_side_rule.decide}


def item_window(item_id: str, located: dict, duration: float, margin: float) -> list[float] | None:
    owner = item_id.split(".")[0] if item_id[:1] == "E" and item_id[1:2].isdigit() else None
    if owner:
        event = located.get(owner) or {}
        if event.get("start") is None:
            return None
        return [max(0.0, event["start"] - margin), min(duration, event["end"] + margin)]
    if item_id == "G.end_state":
        return [max(0.0, duration - 1.0), duration]
    return [0.0, duration]


def _frames(window: list[float], fps: float, count: int):
    return range(max(0, round(window[0] * fps)), min(count - 1, round(window[1] * fps)) + 1)


def sam_summary(masks, window, fps: float, count: int) -> dict:
    """窗口内每帧最大实例面积。掩码只存在部分帧上，逐帧取最近的一张。"""
    keys = sam_expert.frame_keys(masks)
    areas, instances = [], []
    for frame in _frames(window, fps, count):
        if not keys:
            areas.append(0)
            instances.append(0)
            continue
        m = masks["masks_%d" % min(keys, key=lambda k: abs(k - frame))]
        if m.ndim == 3 and len(m):
            per = m.reshape(len(m), -1).sum(1)
            areas.append(int(per.max()))
            instances.append(len(m))
        else:
            areas.append(int(m.sum()))
            instances.append(1)
    if not areas:
        return {"frames": 0}
    return {"frames": len(areas), "frames_with_mask": sum(a > 0 for a in areas),
            "area_first": areas[0], "area_last": areas[-1],
            "area_min": min(areas), "area_max": max(areas),
            "instances_max": max(instances)}


def track_summary(z, window, fps: float) -> dict:
    tracks, vis = z["tracks"], z["visibility"].astype(bool)
    labels = [str(x) for x in z["labels"]]
    first, last = max(0, round(window[0] * fps)), min(len(tracks) - 1, round(window[1] * fps))
    groups: dict[str, list] = {"subject": [], "background": []}
    for i, label in enumerate(labels):
        seen = np.flatnonzero(vis[first:last + 1, i])
        if len(seen) < 2:
            continue
        role = label.split(":")[0]
        if role not in groups:
            continue
        a, b = tracks[first + seen[0], i], tracks[first + seen[-1], i]
        groups[role].append({"dx": float(b[0] - a[0]), "dy": float(b[1] - a[1]),
                             "visible_fraction": float(vis[first:last + 1, i].mean())})
    out = {}
    for role, rows in groups.items():
        if not rows:
            out[role] = {"points_followed": 0}
            continue
        move = [float(np.hypot(r["dx"], r["dy"])) for r in rows]
        out[role] = {"points_followed": len(rows), "mean_displacement_px": float(np.mean(move)),
                     "max_displacement_px": float(np.max(move)),
                     "mean_dx": float(np.mean([r["dx"] for r in rows])),
                     "mean_dy": float(np.mean([r["dy"] for r in rows])),
                     "mean_visible_fraction": float(np.mean([r["visible_fraction"] for r in rows]))}
    return out


def flow_summary(z, window, fps: float) -> dict:
    pairs = []
    for flow, index in zip(z["flow"], z["indices"]):
        if window[0] * fps - 1 <= index <= window[1] * fps:
            pairs.append({"frame": int(index), "mean_dx": float(flow[0].mean()),
                          "mean_dy": float(flow[1].mean()),
                          "mean_magnitude": float(np.sqrt((flow ** 2).sum(0)).mean())})
    return {"pairs": pairs}


def build(plan: Plan, located: dict, folder: Path, meta: dict, config) -> dict:
    """每个检查项的数值摘要。窗口由第一次调用定位到的事件时间决定。"""
    fps, count = meta["fps"], meta["frames"]
    duration = meta["last_frame_seconds"]
    checks = config.checks
    margin = config.vlm["frames"]["event_margin_s"]
    judge_thresholds = checks["thresholds"]["judge"]
    tracks = np.load(folder / "cotracker3.tracks.npz") if (folder / "cotracker3.tracks.npz").exists() else None
    flow = np.load(folder / "waft.flow.npz") if (folder / "waft.flow.npz").exists() else None
    pose = np.load(folder / "pose.keypoints.npz") if (folder / "pose.keypoints.npz").exists() else None
    sam_files = {p.name[len("sam31."):-len(".masks.npz")]: p
                 for p in folder.glob("sam31.*.masks.npz")}

    def masks_for(concept: str):
        path = sam_files.get(concept)
        return np.load(path, allow_pickle=True) if path else None

    evidence: dict[str, dict] = {}
    for event in plan.events:
        evidence[event.id] = _summarize(event.id, event.concepts, [e for e in ("cotracker3",)],
                                        located, fps, count, duration, margin, tracks, flow, pose,
                                        sam_files, masks_for, judge_thresholds, checks)
    for item in plan.all_items():
        evidence[item.id] = _summarize(item.id, item.concepts, [item.expert] if item.expert else [],
                                       located, fps, count, duration, margin, tracks, flow, pose,
                                       sam_files, masks_for, judge_thresholds, checks)
        if item.judged_by != "rule" and item.type in RULES:
            result = RULES[item.type](item, folder, meta, checks)
            if result is not None:
                evidence[item.id]["rule_numbers"] = {
                    "status_if_decided_by_rule": result.status, "measured": result.measured,
                    "failed_components": list(result.failed),
                    "note": "uncalibrated rule: numbers for reference, not a verdict"}
        if (item.type == "contact" and pose is not None and item.concepts
                and evidence[item.id]["window_seconds"] is not None):
            target = item.concepts[-1]
            masks = masks_for(target)
            if masks is not None:
                evidence[item.id]["contact"] = pose_expert.hand_summary(
                    pose, masks, evidence[item.id]["window_seconds"], fps,
                    hand_score=judge_thresholds["hand_score"],
                    hand_reach=judge_thresholds["hand_reach"],
                    touch_px=item.rule.get("touch_px", 30))
    return evidence


def _summarize(item_id, concepts, experts, located, fps, count, duration, margin,
               tracks, flow, pose, sam_files, masks_for, judge_thresholds, checks) -> dict:
    window = item_window(item_id, located, duration, margin)
    row: dict = {"window_seconds": window}
    if window is None:
        row["note"] = "the step was not located in the video, so no window was cut"
        return row
    for name in experts:
        if name == "sam31":
            got = {c: sam_summary(masks_for(c), window, fps, count) for c in concepts if c in sam_files}
            row["sam31"] = got
            unsighted = [c for c, s in got.items() if s.get("frames_with_mask", 0) == 0]
            if unsighted:
                row["sam31_note"] = ("the text prompt never segmented %s in any frame of this window; "
                                     "judge that object from the frames themselves and do not read the "
                                     "empty mask as \"present and unchanged\"" % ", ".join(unsighted))
        elif name == "cotracker3" and tracks is not None:
            row["cotracker3"] = track_summary(tracks, window, fps)
        elif name == "waft" and flow is not None:
            row["waft"] = flow_summary(flow, window, fps)
        elif name in ("hand_side", "pose") and pose is not None:
            if name == "pose":
                row["pose"] = {"frames_with_person": int(pose["valid_person"].any(1).sum()),
                               "people_max": int(pose["valid_person"].sum(1).max())}
    if concepts and "sam31" not in row and any(c in sam_files for c in concepts):
        row["sam31"] = {c: sam_summary(masks_for(c), window, fps, count)
                        for c in concepts if c in sam_files}
    row["not_available"] = [n for n in experts if n not in row]
    return row
