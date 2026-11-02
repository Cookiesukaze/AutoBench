"""命令行入口：plan / quality / judge / score。"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import cost, run
from .config import load
from .store import Store

COMMANDS = ("plan", "quality", "judge", "score")


def _plan(args: argparse.Namespace) -> None:
    store = Store(load(), args.run_id)
    plan = run.plan_one(store, load(), cost.Recorder(), args.specs, args.question, args.image_dir)
    print(f"plan: {plan.question_id} 事件 {len(plan.events)} 检查项 {sum(1 for _ in plan.all_items())}")


def _judge(args: argparse.Namespace) -> None:
    store = Store(load(), args.run_id)
    result = run.judge_one(store, load(), cost.Recorder(), args.config, args.unit,
                           args.video, args.image, args.experts)
    undecided = [k for k, v in result["items"].items() if v.get("status") == "unjudgeable"]
    print(f"judge: {args.config} {args.unit} 未定 {len(undecided)} 项")


def _score(args: argparse.Namespace) -> None:
    store = Store(load(), args.run_id)
    report = run.score_all(store, load(), cost.Recorder())
    rows = report["board"]["rows"]
    print("score:", [(c, None if rows[c]["total"] is None else round(rows[c]["total"], 4))
                     for c in report["board"]["ranked_order"]])


def _quality(args: argparse.Namespace) -> None:
    report = run.quality(Store(load(), args.run_id), load(), cost.Recorder(),
                         args.quality_dir, args.units)
    print("quality:", {c: (None if r is None else round(r, 3)) for c, r in report["rates"].items()})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="autobench")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-id", required=True, help="这一次评测的目录名")

    p = sub.add_parser("plan", parents=[common])
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.set_defaults(func=_plan)

    q = sub.add_parser("quality", parents=[common])
    q.add_argument("--quality-dir", type=Path, required=True,
                   help="归档测量目录，形如 <run>/stage/quality")
    q.add_argument("--units", nargs="+", required=True)
    q.set_defaults(func=_quality)

    j = sub.add_parser("judge", parents=[common])
    j.add_argument("--config", required=True, help="配置名，如 T1")
    j.add_argument("--unit", required=True, help="形如 R041_E1_E1__seed48")
    j.add_argument("--video", type=Path, required=True)
    j.add_argument("--image", type=Path, required=True, help="参考图")
    j.add_argument("--experts", type=Path, required=True, help="专家数组目录")
    j.set_defaults(func=_judge)

    s = sub.add_parser("score", parents=[common])
    s.set_defaults(func=_score)

    args = parser.parse_args(argv)
    args.func(args)
