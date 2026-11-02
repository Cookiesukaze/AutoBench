"""bootstrap 排名：谁的分数高，是真的高还是抖动。

输入是每个配置每个 unit 的逐行账（step01 的 rows），不是三个汇总分——
排名要按维度重新算，不能用已经平均过的数再平均一次。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ....app.types import split_unit

MEASURED = {"pass": 1.0, "fail": 0.0}
NOT_APPLICABLE = ("not_triggered", "not_applicable", "event_not_done")
UNMEASURED = ("unjudgeable", "needs_review")


@dataclass(frozen=True)
class Record:
    config: str
    unit: str
    check_id: str
    dimension: str
    value: float | None


@dataclass
class Options:
    category_weights: dict = field(default_factory=lambda: {"content": 1.0, "physics": 1.0, "quality": 1.0})
    min_coverage: float = 0.5
    n_boot: int = 2000
    seed: int = 20261027
    alpha: float = 0.05
    separable_at: float = 0.95


def records(by_config: dict[str, dict[str, list[dict]]], categories: dict[str, str]) -> list[Record]:
    out: list[Record] = []
    for config, by_unit in by_config.items():
        for unit, rows in by_unit.items():
            for row in rows:
                value = row.get("value")
                if value is not None:
                    out.append(Record(config, unit, row["id"], row["dimension"], float(value)))
                elif row.get("why") in NOT_APPLICABLE:
                    continue
                else:
                    out.append(Record(config, unit, row["id"], row["dimension"], None))
    if not out:
        raise ValueError("没有任何可排名的行")
    for record in out:
        prefix = record.dimension.split("-")[0]
        if prefix not in categories:
            raise ValueError(f"维度 {record.dimension!r} 的前缀不在 categories 里")
    return out


def _category(dimension: str, categories: dict[str, str]) -> str:
    return categories[dimension.split("-")[0]]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def cells(records_: list[Record], categories: dict[str, str]) -> dict:
    """{config: {question: {dimension: 值或 None}}}。没触发过的维度不出现，触发但没测到的是 None。"""
    seen = set()
    per_video: dict[tuple, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    for r in records_:
        key = (r.config, r.unit, r.check_id)
        if key in seen:
            raise ValueError(f"重复的检查项 {key}")
        seen.add(key)
        per_video[(r.config, r.unit)][r.dimension].append(r.value)
    seeds: dict[str, dict[str, dict[str, list[float | None]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for (config, unit), dims in per_video.items():
        question, _ = split_unit(unit)
        for dim, values in dims.items():
            seeds[config][question][dim].append(_mean([v for v in values if v is not None]))
    return {c: {q: {d: _mean([v for v in vs if v is not None]) for d, vs in dims.items()}
                for q, dims in qs.items()} for c, qs in seeds.items()}


def _to_tensor(cell_map: dict) -> dict:
    configs = sorted(cell_map)
    questions = sorted({q for qs in cell_map.values() for q in qs})
    dims = sorted({d for qs in cell_map.values() for ds in qs.values() for d in ds})
    shape = (len(configs), len(dims), len(questions))
    values = np.zeros(shape)
    measured = np.zeros(shape, bool)
    planned = np.zeros(shape, bool)
    for i, c in enumerate(configs):
        for k, q in enumerate(questions):
            for j, d in enumerate(dims):
                if d in cell_map[c].get(q, {}):
                    planned[i, j, k] = True
                    v = cell_map[c][q][d]
                    if v is not None:
                        values[i, j, k], measured[i, j, k] = v, True
    return {"configs": configs, "questions": questions, "dims": dims,
            "values": values, "measured": measured, "planned": planned}


def _dim_scores(t: dict, weights: np.ndarray) -> np.ndarray:
    w = np.atleast_2d(weights).astype(float)
    num = np.einsum("cdq,bq->bcd", t["values"], w)
    den = np.einsum("cdq,bq->bcd", t["measured"].astype(float), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _board(t: dict, ranked: list[str], options: Options, categories: dict[str, str]) -> tuple[list[str], dict]:
    rows = [t["configs"].index(c) for c in ranked]
    board, dropped = [], {}
    for j, d in enumerate(t["dims"]):
        support = t["planned"][rows, j, :].any(axis=0)
        if not support.any():
            dropped[d] = "没有任何参评配置有这个维度"
            continue
        cover = t["measured"][rows, j, :][:, support].mean(axis=1)
        worst = float(cover.min())
        if worst < options.min_coverage:
            dropped[d] = f"最低覆盖 {worst:.2f} 低于 {options.min_coverage}"
        else:
            board.append(d)
    present = {_category(d, categories) for d in board}
    for c in options.category_weights:
        if options.category_weights[c] > 0 and c not in present:
            dropped[f"category:{c}"] = "这个类别没有任何维度进榜"
    return board, dropped


def _rank_of(total: np.ndarray) -> np.ndarray:
    t = np.round(total, 12)
    return 1 + (t[..., None, :] > t[..., :, None]).sum(axis=-1)


def _interval(col: np.ndarray, lo: float, hi: float) -> list:
    x = col[~np.isnan(col)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [None, None]


def _quality(quality: dict[str, float | None], config: str) -> float:
    """没有画质分就是 NaN：宁可不排名，也不拿内容分冒充总分。"""
    value = quality.get(config)
    return float("nan") if value is None else float(value)


def _category_means(dim_scores: np.ndarray, board: list[str], dims: list[str],
                    categories: dict[str, str], cats: list[str]) -> np.ndarray:
    """[..., C, 类别] 的类别均分。维度先归到类别，再在类别内平均。"""
    index = {d: i for i, d in enumerate(dims)}
    out = []
    for cat in cats:
        ix = [index[d] for d in board if _category(d, categories) == cat]
        if not ix:
            out.append(np.full(dim_scores.shape[:2], np.nan))
        else:
            out.append(np.nanmean(dim_scores[..., ix], axis=-1))
    return np.stack(out, axis=-1)


def _total(category_means: np.ndarray, weights: dict[str, float],
           quality: np.ndarray) -> np.ndarray:
    """三个维度 1:1:1 的总分。任何一维缺失就是 NaN，不重新分配权重。

    quality 是配置级常数，按配置广播到每个重采样上。
    """
    cats = ["content", "physics", "quality"]
    w = np.array([weights[c] for c in cats], dtype=float)
    means = np.array(category_means, dtype=float, copy=True)
    if means.ndim == 2:
        means = means[None]
    q = np.broadcast_to(np.asarray(quality, dtype=float), means.shape[:2])
    means[..., 2] = q
    present = ~np.isnan(means)
    numerator = np.where(present, np.nan_to_num(means) * w, 0.0).sum(axis=-1)
    denominator = (present * w).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denominator > 0, numerator / denominator, np.nan)
    # 缺任何一维 → 总分未定：不用剩下的维度凑一个数。
    return np.where(present.all(axis=-1), out, np.nan)


def rank(by_config: dict[str, dict[str, list[dict]]], options: Options,
         categories: dict[str, str], quality: dict[str, float | None],
         excluded: dict | None = None) -> dict:
    excluded = excluded or {}
    t = _to_tensor(cells(records(by_config, categories), categories))
    unknown = sorted(set(excluded) - set(t["configs"]))
    if unknown:
        raise KeyError(f"被排除的配置没有任何判定结果：{unknown}")
    ranked = [c for c in t["configs"] if c not in excluded]
    if len(ranked) < 2:
        raise ValueError("参评配置少于两个")
    board, dropped = _board(t, ranked, options, categories)
    if not board:
        raise ValueError(f"没有一个维度的覆盖够：{dropped}")

    rng = np.random.default_rng(options.seed)
    draws = rng.integers(0, len(t["questions"]), size=(options.n_boot, len(t["questions"])))
    boot_w = np.stack([np.bincount(row, minlength=len(t["questions"])) for row in draws])

    idx = [t["configs"].index(c) for c in ranked]
    cats = ["content", "physics", "quality"]
    q_vec = np.array([_quality(quality, c) for c in t["configs"]], dtype=float).reshape(1, -1)
    point_dim = _dim_scores(t, np.ones(len(t["questions"])))[0]
    point_total = _total(_category_means(point_dim[None], board, t["dims"], categories, cats)[0],
                         options.category_weights, q_vec)[0]
    boot_dim = _dim_scores(t, boot_w)
    boot_total = _total(_category_means(boot_dim, board, t["dims"], categories, cats),
                        options.category_weights, q_vec)
    valid = ~np.isnan(boot_total[:, idx]).any(axis=1)
    if not valid.any():
        raise ValueError("没有任何一次重采样拿到完整的三维总分：画质费率或判定结果缺失")
    bt = boot_total[valid][:, idx]
    lo, hi = 100 * options.alpha / 2, 100 * (1 - options.alpha / 2)
    ranks = _rank_of(bt)
    order = list(np.argsort(-np.round(point_total[idx], 12), kind="stable"))
    wins = (bt[:, :, None] > bt[:, None, :]).mean(axis=0) + 0.5 * (
        np.round(bt[:, :, None], 12) == np.round(bt[:, None, :], 12)).mean(axis=0)

    rows_out: dict[str, Any] = {}
    for c in t["configs"]:
        ci = t["configs"].index(c)
        dims = {}
        for j, d in enumerate(t["dims"]):
            support = t["planned"][idx, j, :].any(axis=0)
            dims[d] = {
                "score": None if np.isnan(point_dim[ci, j]) else float(point_dim[ci, j]),
                "ci": _interval(boot_dim[:, ci, j], lo, hi),
                "questions_measured": int(t["measured"][ci, j].sum()),
                "questions_planned": int(t["planned"][ci, j].sum()),
                "coverage": float(t["measured"][ci, j, support].mean()) if support.any() else None,
                "on_board": d in board,
            }
        rows_out[c] = {"status": "excluded" if c in excluded else "ranked", "dimensions": dims}
        if c in excluded:
            rows_out[c]["excluded"] = excluded[c]
            rows_out[c]["total"] = None if np.isnan(point_total[ci]) else float(point_total[ci])
    for k, c in enumerate(ranked):
        ci = t["configs"].index(c)
        nxt = order.index(k)
        better = order[nxt + 1] if nxt + 1 < len(order) else None
        cats_out = {}
        for cat in sorted({_category(d, categories) for d in board}):
            ix = [t["dims"].index(d) for d in board if _category(d, categories) == cat]
            cats_out[cat] = float(point_dim[ci, ix].mean())
        cats_out["quality"] = quality.get(c)
        rows_out[c].update({
            "total": float(point_total[ci]), "total_ci": _interval(bt[:, k], lo, hi),
            "categories": cats_out, "rank": int(_rank_of(point_total[idx])[k]),
            "rank_ci": [float(np.percentile(ranks[:, k], lo)), float(np.percentile(ranks[:, k], hi))],
            "p_rank_1": float((ranks[:, k] == 1).mean()), "mean_rank": float(ranks[:, k].mean()),
            "p_beats_next": float(wins[k, better]) if better is not None else None,
            "next": ranked[better] if better is not None else None,
            "separable_from_next": bool(wins[k, better] >= options.separable_at) if better is not None else None,
        })
    return {
        "config": {"category_weights": options.category_weights, "min_coverage": options.min_coverage,
                   "n_boot": options.n_boot, "seed": options.seed, "alpha": options.alpha,
                   "separable_at": options.separable_at},
        "questions": t["questions"], "board_dimensions": board, "dropped": dropped,
        "bootstrap": {"replicates": options.n_boot, "valid": int(valid.sum())},
        "ranked_order": [ranked[k] for k in order], "rows": rows_out,
    }


def write(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    board = report["board_dimensions"]
    head = ["rank", "config", "status", "total", "total_ci_low", "total_ci_high",
            "p_rank_1", "p_beats_next", "separable_from_next"] + board
    with open(out_dir / "leaderboard.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(head)
        for c in report["ranked_order"]:
            row = report["rows"][c]
            writer.writerow([row["rank"], c, row["status"], _fmt(row["total"]),
                             *(_fmt(x) for x in row["total_ci"]),
                             _fmt(row["p_rank_1"], 2), _fmt(row["p_beats_next"], 2),
                             row["separable_from_next"],
                             *(_fmt(row["dimensions"][d]["score"]) for d in board)])
    return out_dir


def _fmt(x, digits: int = 3) -> str:
    return "" if x is None else f"{x:.{digits}f}"
