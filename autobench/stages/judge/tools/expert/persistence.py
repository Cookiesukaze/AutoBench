"""消失：被操作的物体在整个步骤里不能凭空不见、裂开或并成一个。

掩码面积曲线说话。概念曾经稳定可见（取运行最大值，晚出场的也算）之后，出现至少
zero_run_samples 个样本的零面积，且不落在片尾豁免区，就是一个消失候选。候选一律
按 fail 交回去，由 VLM 判"这是不是题面要求的离开"——飞走、放下，在这里长得一模一样。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .....app.data.types import Item, RuleResult
from ...experts import sam as sam_expert


def decide(item: Item, folder: Path, meta: dict, checks: dict) -> RuleResult | None:
    params = item.rule
    stable_min, zero_run = params["stable_min_px"], params["zero_run_samples"]
    tail_exempt, group_ratio = params["tail_exempt_samples"], params["size_group_ratio"]
    watched = [c for c in item.concepts if (folder / f"sam31.{c}.masks.npz").exists()]
    if not watched:
        return None

    findings, size_flags, series, not_segmented = [], [], {}, []
    for concept in watched:
        z = sam_expert.load(folder, concept)
        frames_idx = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("masks_"))
        areas = [(f, int(z[f"masks_{f}"].sum())) for f in frames_idx]
        runmax, stable_from = 0, None
        peak_all = max((a for _, a in areas), default=0)
        series[concept] = {"samples": len(areas), "nonzero_samples": sum(a > 0 for _, a in areas),
                           "peak_area_px": peak_all}
        if peak_all == 0:
            # 文本提示词整段没分到这个概念：这是"规则看不见它"，不是"它一直在"。
            not_segmented.append(concept)
            continue
        stable_here = min(stable_min, max(params["stable_floor_px"],
                                          int(params["stable_peak_share"] * peak_all)))
        for i, (_, a) in enumerate(areas):
            runmax = max(runmax, a)
            if stable_from is None and runmax >= stable_here:
                stable_from = i
        if stable_from is None:
            continue
        cur_zero, start_i = 0, None
        for i in range(stable_from, len(areas)):
            if areas[i][1] == 0:
                if cur_zero == 0:
                    start_i = i
                cur_zero += 1
            else:
                cur_zero = 0
        if cur_zero >= zero_run and start_i is not None and start_i < len(areas) - tail_exempt:
            limit_i = start_i - 1
            while limit_i > 0 and areas[limit_i][1] >= params["explosion_factor"] * stable_min:
                limit_i -= 1
            last_seen = next((areas[i] for i in range(limit_i, -1, -1) if areas[i][1] > 0), None)
            border = None
            if last_seen is not None:
                m = z[f"masks_{last_seen[0]}"]
                if m.ndim == 3:
                    m = m.any(0)
                ys, xs = np.where(m)
                h, w = m.shape
                margin = params["border_touch_px"]
                border = bool(xs.min() < margin or xs.max() > w - margin
                              or ys.min() < margin or ys.max() > h - margin)
            biggest = max(a for _, a in areas)
            smaller = min((a for _, a in areas if a > 0), default=0) or biggest
            explosion = (biggest >= params["explosion_factor"] * stable_min
                         and biggest > params["explosion_reappear_ratio"] * smaller)
            findings.append({
                "concept": concept, "zero_from_sample": start_i, "zero_samples": cur_zero,
                "peak_area_px": runmax,
                "last_seen_frame": last_seen[0] if last_seen else None,
                # 最后可见之后的第一个采样点：中间若冒出一块误分割，归零的起点不能被它盖住
                "first_zero_frame": next((f for f, _ in areas if last_seen is not None and f > last_seen[0]),
                                         areas[start_i][0]),
                "last_seen_touches_border": border,
                "area_explosion_after_reappear": bool(explosion),
                "verdict_hint": ("mask stayed mid-frame, then went to zero: it vanished in place, "
                                 "it did not walk out of the frame" if border is False else
                                 "last seen at the frame border: leaving the frame is plausible"),
            })

        # 合并 / 分裂：实例数掉了但面积没归零。只数演员掩码沾到的实例，
        # 免得镜面反射、背景道具的分割噪声被读成被操作物体合并。
        actor_name = item.params.get("actor_concept") or item.params.get("actor")
        actor_file = folder / f"sam31.{actor_name}.masks.npz" if actor_name else None
        actor = np.load(actor_file, allow_pickle=True) if actor_file and actor_file.exists() else None
        counts = []
        for f, _ in areas:
            m = z[f"masks_{f}"]
            insts = list(m) if m.ndim == 3 else [m]
            if actor is not None:
                near = sam_expert.mask_near(actor, f)
                if near is not None and near.any():
                    insts = [i for i in insts if (i & near).sum() > params["actor_overlap_share"] * max(1, int(i.sum()))]
            counts.append(len(insts))
        baseline_n = int(np.median(counts)) if counts else 0
        if baseline_n >= 2:
            for i in range(1, len(counts)):
                if counts[i] < counts[i - 1] and counts[i] >= 1:
                    run = 1
                    while i + run < len(counts) and counts[i + run] == counts[i]:
                        run += 1
                    if run < params["merge_min_samples"]:
                        continue
                    if sam_expert.instances_left_frame(z, areas[i - 1][0], areas[i][0],
                                                   margin=params["border_touch_px"]):
                        continue
                    before, after = z[f"masks_{areas[i - 1][0]}"], z[f"masks_{areas[i][0]}"]
                    if before.ndim == 3 and after.ndim == 3:
                        gone = before.any(0) & ~after.any(0)
                        if gone.any() and sam_expert.covered_by_other(
                                folder, areas[i][0], gone,
                                folder / f"sam31.{concept}.masks.npz") >= params["occlusion_cover"]:
                            continue
                    findings.append({
                        "concept": concept, "merge_from_instances": counts[i - 1],
                        "merge_to_instances": counts[i], "merge_at_sample": i, "merge_samples": run,
                        "verdict_hint": "instances were tracked separately and then as fewer while "
                                        "none left the frame: they merged",
                    })
                    break

    stats_file = folder / "area_stats.json"
    if stats_file.exists() and (folder / "config.json").exists():
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        config_name = json.loads((folder / "config.json").read_text(encoding="utf-8"))["config"]
        for concept in watched:
            per_config = {c: s.get("concepts", {}).get(concept, {}).get("median_area", 0)
                          for c, s in stats.items() if not c.startswith("_") and isinstance(s, dict)}
            mine = per_config.get(config_name, 0)
            others = [a for c, a in per_config.items() if c != config_name and a > 0]
            # 基准取"其它配置里最小的那个"：好几个配置同时有这个毛病时，均值会把它稀释掉。
            if mine > 0 and others:
                baseline = min(others)
                if mine > group_ratio * baseline:
                    size_flags.append({"concept": concept, "area_px": mine,
                                       "other_configs_min_px": baseline,
                                       "ratio": round(mine / baseline, 1)})

    if not series:
        return None
    if all(concept in not_segmented for concept in series):
        return None
    measured = {"concepts": series, "not_segmented": not_segmented}
    if not findings and not size_flags:
        return RuleResult(status="pass", measured=measured)
    return RuleResult(
        status="fail", review=True,
        measured={**measured, "vanish": findings, "size_group": size_flags,
                  "review_hint": "a departure or size difference the prompt asks for must pass; "
                                 "only the number-backed vanish or outlier is a fail"},
    )
