"""按顺序调度四个阶段，并且是唯一给阶段计时的地方。

阶段自己不知道前后有谁；顺序、参数、记账都在这里。
"""

from __future__ import annotations

from pathlib import Path

from ..stages.judge import run as judge_stage
from ..stages.judge.tools.llm.client import Client
from ..stages.plan import run as plan_stage
from ..stages.quality import run as quality_stage
from ..stages.score import run as score_stage
from ..stages.score.steps import step03_leaderboard
from . import cost
from .config import Config
from .store import Store
from .types import ContractError, Plan, plan_from_dict, plan_to_dict, split_unit


def plan_one(store: Store, config: Config, recorder: cost.Recorder, specs: Path,
             question: str, image_dir: Path) -> Plan:
    with recorder.stage("plan", question):
        plan = plan_stage.run(specs, question, image_dir, config)
        store.write_json(store.plan_file(question), plan_to_dict(plan))
    recorder.write(store)
    return plan


def judge_one(store: Store, config: Config, recorder: cost.Recorder, config_name: str,
              unit: str, video: Path, reference: Path, experts_dir: Path) -> dict:
    question, _ = split_unit(unit)
    plan = plan_from_dict(store.read_json(store.plan_file(question)))
    workdir = store.judged_dir(config_name, unit)
    client = Client(config.vlm, workdir, recorder)
    with recorder.stage("judge", unit, config=config_name):
        result = judge_stage.run(client, plan, video, reference, experts_dir, workdir, config)
        store.write_json(store.result_file(config_name, unit), result)
    recorder.write(store)
    return result


def _quality_rates(store: Store) -> dict[str, float | None]:
    """画质阶段还没实现；有 quality.json 就用它的 rates，没有就是全未测量。"""
    path = store.root / "quality.json"
    if not path.exists():
        return {}
    return dict(store.read_json(path).get("rates") or {})


def score_all(store: Store, config: Config, recorder: cost.Recorder) -> dict:
    units = store.judged_units()
    results: dict[str, dict[str, dict]] = {}
    plans: dict[str, Plan] = {}
    for config_name, unit_list in units.items():
        for unit in unit_list:
            question, _ = split_unit(unit)
            if question not in plans:
                plans[question] = plan_from_dict(store.read_json(store.plan_file(question)))
            results.setdefault(config_name, {})[unit] = store.read_json(
                store.result_file(config_name, unit))
    rates = _quality_rates(store)
    with recorder.stage("score", "all"):
        report = score_stage.run(results, plans, rates, config)
        store.write_json(store.score_file, {k: report[k] for k in
                                            ("per_video", "per_config", "dimensions")})
        if report["board"]["ranked_order"]:
            step03_leaderboard.write(report["board"], store.board_dir)
    recorder.write(store)
    return report


def quality(store: Store, config: Config, recorder: cost.Recorder, quality_dir: Path,
            units: list[str], parents: dict | None = None) -> dict:
    with recorder.stage("quality", "all"):
        report = quality_stage.run(units, quality_dir, config, parents)
        store.write_json(store.root / "quality.json", report)
    recorder.write(store)
    return report
