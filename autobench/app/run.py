"""单阶段调用、输入核对和成本记录。

阶段自己不知道前后有谁；顺序、参数、记账都在这里。
"""

from __future__ import annotations

from pathlib import Path
import threading

from ..stages.judge import run as judge_stage
from ..stages.judge.tools.llm.client import Client
from ..stages.judge.steps import step00_reference
from ..stages.plan import run as plan_stage
from ..stages.quality import run as quality_stage
from ..stages.basic_quality import run as basic_quality_stage
from ..stages.score import run as score_stage
from ..stages.score.steps import step03_leaderboard
from .config import Config
from .data.store import Recorder, Store
from .data.types import ContractError, Plan, plan_from_dict, plan_to_dict, split_unit


_REFERENCE_LOCK = threading.Lock()


def _reference_audit(store, config, recorder, plan, reference):
    if not any(i.type == "identity" for i in plan.all_items()):
        return {}
    # 同题所有配置共用一次参考图核查，避免并发多票使适用项集合各不相同。
    folder = store.root / "reference" / plan.question_id
    path = folder / "applicability.json"
    from .batch.inputs import fingerprint
    identity = fingerprint(dict(plan=plan_to_dict(plan), vlm=config.vlm))
    with _REFERENCE_LOCK:
        if path.exists():
            saved = store.read_json(path)
            if saved.get("identity") != identity:
                raise ContractError("参考图核查缓存与当前计划或模型配置不符，请使用新 run-id")
            return saved["items"]
        client = Client(config.vlm, folder, recorder)
        _, audit = step00_reference.run(client, plan, reference)
        store.write_json(path, dict(identity=identity, items=audit))
        return audit


def plan_one(
    store: Store,
    config: Config,
    recorder: Recorder,
    specs: Path,
    question: str,
    image_dir: Path,
) -> Plan:
    with recorder.stage("plan", question, store=store):
        plan = plan_stage.run(specs, question, image_dir, config)
        store.write_json(store.plan_file(question), plan_to_dict(plan))
    recorder.write(store)
    return plan


def judge_one(
    store: Store,
    config: Config,
    recorder: Recorder,
    config_name: str,
    unit: str,
    video: Path,
    reference: Path,
    experts_dir: Path,
) -> dict:
    question, _ = split_unit(unit)
    plan = plan_from_dict(store.read_json(store.plan_file(question)))
    from .batch.inputs import sha256

    if sha256(reference) != plan.reference_sha256:
        raise ContractError("judge 的参考图与计划 SHA256 不符")
    workdir = store.judged_dir(config_name, unit)
    store.write_json(
        workdir / "identity.json",
        dict(config=config_name, unit=unit, video_sha256=sha256(video)),
    )
    client = Client(config.vlm, workdir, recorder)
    with recorder.stage("judge", unit, store=store, config=config_name):
        reference_audit = _reference_audit(store, config, recorder, plan, reference)
        result = judge_stage.run(
            client, plan, video, reference, experts_dir, workdir, config, reference_audit
        )
        store.write_json(store.result_file(config_name, unit), result)
    recorder.write(store)
    return result


def _quality_scores(store: Store) -> dict:
    """正式画质逐视频得分；不接纳旧 rate 作为替代分数。"""
    path = store.root / "quality.json"
    if not path.exists():
        return {}
    report = store.read_json(path)
    if "scores" not in report:
        raise ContractError(
            "quality.json 是旧问题率格式；请先执行 basic-quality 和正式 quality"
        )
    return report["scores"]


def score_all(
    store: Store, config: Config, recorder: Recorder, expected: dict | None = None
) -> dict:
    if expected is None and (store.root / "manifest.json").exists():
        expected = store.read_json(store.root / "manifest.json")["expected"]
    units = (
        {
            c: [u for u in us if store.result_file(c, u).exists()]
            for c, us in expected.items()
        }
        if expected is not None
        else store.judged_units()
    )
    results: dict[str, dict[str, dict]] = {}
    plans: dict[str, Plan] = {}
    for config_name, unit_list in units.items():
        for unit in unit_list:
            question, _ = split_unit(unit)
            if question not in plans:
                plans[question] = plan_from_dict(
                    store.read_json(store.plan_file(question))
                )
            results.setdefault(config_name, {})[unit] = store.read_json(
                store.result_file(config_name, unit)
            )
    scores = _quality_scores(store)
    with recorder.stage("score", "all", store=store):
        report = score_stage.run(results, plans, scores, config, expected)
        store.write_json(
            store.score_file,
            {k: report[k] for k in ("per_video", "per_config", "dimensions")},
        )
        step03_leaderboard.write(report["board"], store.board_dir)
    recorder.write(store)
    return report


def quality(
    store: Store,
    config: Config,
    recorder: Recorder,
    quality_dir: Path,
    units: list[str],
    expected: dict | None = None,
) -> dict:
    with recorder.stage("quality", "all", store=store):
        report = quality_stage.run(units, quality_dir, config, expected)
        store.write_json(store.root / "quality.json", report)
    recorder.write(store)
    return report


def basic_quality(
    store: Store,
    config: Config,
    recorder: Recorder,
    quality_dir: Path,
    units: list[str],
    parents: dict | None = None,
    expected: dict | None = None,
) -> dict:
    with recorder.stage("basic_quality", "all", store=store):
        report = basic_quality_stage.run(units, quality_dir, config, parents, expected)
        store.write_json(store.root / "basic_quality.json", report)
    recorder.write(store)
    return report
