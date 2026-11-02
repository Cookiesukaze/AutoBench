"""使用正式汇总总分排序，按题目重采样三维得分；不另算一套总分。"""

import csv
import json
from pathlib import Path
import numpy as np
from ....app.data.types import split_unit


def _rank_of(total):
    t = np.round(total, 12)
    return 1 + (t[..., None, :] > t[..., :, None]).sum(axis=-1)


def rank(per_video, per_config, options, min_coverage):
    eligible = [
        c
        for c, row in per_config.items()
        if row["total"] is not None and row["coverage"] >= min_coverage
    ]
    unit_sets = {c: set(per_video[c]) for c in eligible}
    # 不同题集没有直接可比的总分，不静默取交集并换掉正式分数。
    comparable = len({tuple(sorted(v)) for v in unit_sets.values()}) <= 1
    ranked = (
        sorted(eligible, key=lambda c: (-round(per_config[c]["total"], 12), c))
        if comparable
        else []
    )
    rows = {
        c: dict(
            status="ranked" if c in ranked else "unranked",
            total=r["total"],
            categories={k: r.get(k) for k in ("instruction", "physics", "quality")},
            reason=None if c in ranked else "incomplete_or_different_units",
            dimensions=r["by_dimension"],
        )
        for c, r in per_config.items()
    }
    questions = sorted({split_unit(u)[0] for c in ranked for u in per_video[c]})
    if ranked:
        point = np.array([per_config[c]["total"] for c in ranked])
        for i, c in enumerate(ranked):
            rows[c]["rank"] = int(_rank_of(point)[i])
    valid_count = 0
    if len(ranked) >= 2 and questions:
        rng = np.random.default_rng(options["seed"])
        draws = rng.integers(0, len(questions), size=(options["n"], len(questions)))
        sums, counts = [], []
        for c in ranked:
            grouped = [
                [v["total"] for u, v in per_video[c].items() if split_unit(u)[0] == q]
                for q in questions
            ]
            sums.append([sum(v) for v in grouped])
            counts.append([len(v) for v in grouped])
        sums, counts = np.array(sums), np.array(counts)
        boot = np.stack(
            [sums[:, draw].sum(axis=1) / counts[:, draw].sum(axis=1) for draw in draws]
        )
        valid_count = len(boot)
        ranks = _rank_of(boot)
        lo, hi = 100 * options["alpha"] / 2, 100 * (1 - options["alpha"] / 2)
        for i, c in enumerate(ranked):
            next_c = ranked[i + 1] if i + 1 < len(ranked) else None
            delta = np.round(boot[:, i] - boot[:, i + 1], 12) if next_c else None
            wins = (
                float((delta > 0).mean() + 0.5 * (delta == 0).mean())
                if next_c
                else None
            )
            rows[c].update(
                total_ci=np.percentile(boot[:, i], [lo, hi]).tolist(),
                rank_ci=np.percentile(ranks[:, i], [lo, hi]).tolist(),
                p_rank_1=float((ranks[:, i] == 1).mean()),
                next=next_c,
                p_beats_next=wins,
                separable_from_next=None
                if wins is None
                else wins >= options["separable_at"],
            )
    return dict(
        ranked_order=ranked,
        rows=rows,
        questions=questions,
        config=dict(
            category_weights={"content": 1, "physics": 1, "quality": 1}, **options
        ),
        bootstrap={"replicates": options["n"], "valid": valid_count},
        note="不同配置的评测单元不一致，未排名"
        if not comparable
        else ("仅一个配置，无两两比较" if len(ranked) == 1 else None),
    )


def write(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (out_dir / "leaderboard.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["rank", "config", "status", "total", "instruction", "physics", "quality"]
        )
        order = report["ranked_order"] + [
            c for c in report["rows"] if c not in report["ranked_order"]
        ]
        for c in order:
            r = report["rows"][c]
            writer.writerow(
                [
                    r.get("rank"),
                    c,
                    r["status"],
                    r["total"],
                    *[
                        r["categories"][k]
                        for k in ("instruction", "physics", "quality")
                    ],
                ]
            )
    return out_dir
