"""正式三维等权计分；第一关告警不参与分值换算。"""

from dataclasses import replace
import math
from ...app.data.types import ContractError
from ...app.data.types import split_unit
from .steps import step01_video, step02_aggregate, step03_leaderboard
from .audit import dimensions


def run(results, plans, quality_scores, config, expected=None):
    per_video, scored, all_rows = {}, {}, {}
    for name, by_unit in sorted(results.items()):
        per_video[name], scored[name], all_rows[name] = {}, {}, {}
        for unit, result in sorted(by_unit.items()):
            question, _ = split_unit(unit)
            score, rows = step01_video.video(
                plans[question], result, config.checks["weights"]["aggregation"]
            )
            q = (quality_scores.get(name) or {}).get(unit)
            if q is not None and (
                isinstance(q, bool)
                or not isinstance(q, (int, float))
                or not math.isfinite(q)
                or not 0 <= q <= 1
            ):
                raise ContractError(f"{name}/{unit}: 正式画质分必须在 [0,1]")
            score = replace(score, quality=q)
            scored[name][unit] = score
            all_rows[name][unit] = rows
            total = (
                None
                if None in (score.instruction, score.physics, q)
                else (score.instruction + score.physics + q) / 3
            )
            per_video[name][unit] = dict(
                instruction=score.instruction,
                physics=score.physics,
                quality=q,
                total=total,
                coverage=score.coverage,
                undetermined=step01_video.undetermined(result),
            )
    expected = (
        expected
        if expected is not None
        else {c: sorted(by) for c, by in results.items()}
    )
    per_config = {}
    for name in sorted(set(expected) | set(scored)):
        by = scored.get(name, {})
        row = step02_aggregate.aggregate(by)
        row["quality"] = (
            sum(s.quality for s in by.values()) / len(by)
            if by and all(s.quality is not None for s in by.values())
            else None
        )
        missing = sorted(set(expected.get(name, [])) - set(by))
        open_units = sorted(
            u for u, s in by.items() if None in (s.instruction, s.physics, s.quality)
        )
        row["missing_units"], row["undetermined_units"] = missing, open_units
        parts = [row.get(k) for k in ("instruction", "physics", "quality")]
        row["total"] = (
            None if missing or open_units or None in parts else sum(parts) / 3
        )
        row["status"] = "undetermined" if row["total"] is None else "measured"
        per_config[name] = row
    board = step03_leaderboard.rank(
        per_video,
        per_config,
        config.checks["weights"]["bootstrap"],
        config.checks["weights"]["min_coverage"],
    )
    return dict(
        per_video=per_video,
        per_config=per_config,
        board=board,
        dimensions=dimensions.summarize(all_rows),
    )
