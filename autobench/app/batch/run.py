"""批次入口：精确输入、阶段续跑、逐视频失败隔离与正式计分。"""

from __future__ import annotations
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .. import run as stages
from ..data.store import Recorder
from ..data.types import ContractError, plan_from_dict, split_unit
from .inputs import fingerprint, sha256, source_fingerprint, videos
from .state import State, exclusive


def tree_signature(folder):
    if folder is None:
        return None
    return {
        p.relative_to(folder).as_posix(): sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def import_quality(source, target, tasks):
    """只导入能够核对 unit、配置和视频摘要的归档。"""
    for unit in sorted({t["unit"] for t in tasks}):
        src = source / unit / "summary.json"
        row = json.loads(src.read_text(encoding="utf-8"))
        if row.get("unit") != unit:
            raise ContractError(f"{src}: 缺少或不匹配的 unit；旧归档需先补齐来源证据")
        for task in (t for t in tasks if t["unit"] == unit):
            if row.get("inputs", {}).get(task["config"]) != task["video_sha256"]:
                raise ContractError(f'{src}: {task["config"]} 的视频 SHA256 不匹配')
        target.joinpath(unit).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target / unit / "summary.json")


def run(
    store,
    config,
    *,
    specs,
    image_dir,
    manifest=None,
    videos_dir=None,
    parents=None,
    quality_dir=None,
    experts_dir=None,
    remote=False,
    judge=False,
):
    from ...stages.plan.steps.step01_load import image_for
    from ...stages.basic_quality.experts.signal import measure
    from ...stages.basic_quality.steps.step01_measure import build_summaries

    if remote and (quality_dir is not None or experts_dir is not None):
        raise ContractError("--remote 与归档导入不能混用")
    for folder in (quality_dir, experts_dir):
        if folder is not None and (
            folder.resolve().is_relative_to(store.root.resolve())
            or store.root.resolve().is_relative_to(folder.resolve())
        ):
            raise ContractError("归档输入目录不能与本批次产物目录重叠")
    by_config = videos(manifest, videos_dir)
    expected = {c: sorted(by) for c, by in by_config.items()}
    units = sorted({u for by in expected.values() for u in by})
    questions = sorted({split_unit(u)[0] for u in units})
    parents = parents or {}
    source = source_fingerprint()
    tasks = [
        dict(config=c, unit=u, video=p, video_sha256=sha256(Path(p)))
        for c, by in by_config.items()
        for u, p in by.items()
    ]
    # Flat artifact folder names must remain unambiguous even when configs contain underscores.
    names = [t["config"] + "_" + t["unit"] for t in tasks]
    if len(set(names)) != len(names):
        raise ContractError("配置与 unit 组合产生同名目录，请更换配置名")
    references = {q: sha256(image_for(image_dir, q)) for q in questions}
    identity = fingerprint(
        dict(
            tasks=tasks,
            specs=sha256(specs),
            references=references,
            checks=config.checks,
            vlm=config.vlm,
            source=source,
            parents=parents,
            remote=config.paths["remote"] if remote else None,
            gpus=config.gpus if remote else None,
            quality=tree_signature(quality_dir),
            experts=tree_signature(experts_dir),
        )
    )
    with exclusive(store.root):
        state = State(store)
        manifest_file = store.root / "manifest.json"
        if (
            manifest_file.exists()
            and store.read_json(manifest_file)["identity"] != identity
        ):
            raise ContractError(
                "同一 run-id 的输入、代码或配置已变更；请使用新 run-id，避免混入旧结果"
            )
        store.write_json(
            manifest_file,
            dict(
                identity=identity,
                expected=expected,
                tasks=tasks,
                references=references,
                source=source,
                parents=parents,
            ),
        )
        failures = []

        def task_run(key, artifacts, function, task_identity=None):
            task_identity = task_identity or identity
            if state.reusable(key, task_identity, artifacts):
                return True
            state.record(key, task_identity, "running")
            for artifact in artifacts:
                if artifact.exists():
                    artifact.unlink()
            recorder = Recorder()
            try:
                function(recorder)
                state.record(key, task_identity, "done", artifacts)
                return True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                state.record(key, task_identity, "failed", error=error)
                failures.append(dict(task=key, error=error))
                return False
            finally:
                recorder.write(store)

        valid_plans = {}
        for q in questions:
            if task_run(
                "plan/" + q,
                [store.plan_file(q)],
                lambda rec, q=q: stages.plan_one(
                    store, config, rec, specs, q, image_dir
                ),
            ):
                valid_plans[q] = plan_from_dict(store.read_json(store.plan_file(q)))
        ready = []
        for task in tasks:
            q, _ = split_unit(task["unit"])
            if q not in valid_plans:
                continue
            plan = valid_plans[q]
            concepts = sorted(
                set(plan.subject_descriptions)
                | {c for e in plan.events for c in e.concepts}
                | {c for item in plan.all_items() for c in item.concepts}
            )
            ready.append(
                dict(
                    task,
                    concepts=concepts,
                    descriptions=plan.subject_descriptions,
                    pose=any(
                        i.expert == "pose" or i.type in ("contact", "hand_side")
                        for i in plan.all_items()
                    ),
                )
            )
        quality_output = store.root / "quality"
        evidence_root = store.root / "evidence"
        if remote and ready:
            from ..remote import stage

            remote_recorder = Recorder()
            with remote_recorder.stage("basic_quality", "remote", store=store):
                downloaded, status = stage.run(store, config, ready, identity)
            import_quality(downloaded / "quality", quality_output, ready)
            from ..remote.collect import promote

            promote(downloaded / "evidence", evidence_root)
            failures.extend(status.get("errors", []))
        elif quality_dir is not None:
            import_quality(quality_dir, quality_output, tasks)
        else:
            for task in tasks:
                folder = store.evidence_dir(task["config"], task["unit"])
                task["folder"] = str(folder)

                def cpu(rec, task=task, folder=folder):
                    with rec.stage(
                        "basic_quality", task["unit"], config=task["config"]
                    ):
                        value = measure(Path(task["video"]))
                        store.write_json(
                            folder / "signal.json",
                            dict(
                                status="observed",
                                expert="signal",
                                video_sha256=task["video_sha256"],
                                observations=dict(metrics=value),
                            ),
                        )

                ok = task_run(
                    "measure/" + task["config"] + "/" + task["unit"],
                    [folder / "signal.json"],
                    cpu,
                )
                if not ok:
                    store.write_json(
                        folder / "signal.json",
                        dict(status="unavailable", error="CPU measurement failed"),
                    )
            build_summaries(tasks, quality_output, config.checks)
        if experts_dir is not None:
            for task in ready:
                folder = experts_dir / (task["config"] + "_" + task["unit"])
                record = store.read_json(folder / "identity.json")
                if any(
                    record.get(k) != task[k] for k in ("config", "unit", "video_sha256")
                ):
                    raise ContractError(f"{folder}: 专家证据身份不匹配")
                shutil.copytree(
                    folder,
                    store.evidence_dir(task["config"], task["unit"]),
                    dirs_exist_ok=True,
                )
        # Missing/failed upstream videos remain in expected, hence cannot disappear from totals.
        for unit in units:
            path = quality_output / unit / "summary.json"
            if not path.exists():
                store.write_json(
                    path,
                    dict(unit=unit, metrics={}, inputs={}, errors=["no measurement"]),
                )
        gate = stages.basic_quality(
            store, config, Recorder(), quality_output, units, parents, expected
        )
        stages.quality(store, config, Recorder(), quality_output, units, expected)
        admitted = [
            t
            for t in ready
            if gate["per_video_admission"][t["config"]][t["unit"]]["allowed"]
        ]
        if judge:

            def evaluate(task):
                c, u = task["config"], task["unit"]
                q, _ = split_unit(u)
                folder = store.judged_dir(c, u)

                def call(rec):
                    store.write_json(
                        folder / "identity.json",
                        dict(config=c, unit=u, video_sha256=task["video_sha256"]),
                    )
                    stages.judge_one(
                        store,
                        config,
                        rec,
                        c,
                        u,
                        Path(task["video"]),
                        image_for(image_dir, q),
                        store.evidence_dir(c, u),
                    )

                dependency = fingerprint(
                    dict(
                        batch=identity,
                        plan=sha256(store.plan_file(q)),
                        evidence=tree_signature(store.evidence_dir(c, u)),
                    )
                )
                return task_run(
                    "judge/" + c + "/" + u, [store.result_file(c, u)], call, dependency
                )

            with ThreadPoolExecutor(max_workers=config.vlm["concurrency"]) as pool:
                for future in as_completed(
                    [pool.submit(evaluate, t) for t in admitted]
                ):
                    future.result()
            score_report = stages.score_all(store, config, Recorder(), expected)
            failures.extend(
                dict(
                    task="score/" + c,
                    error="三维总分未定",
                    missing_units=row["missing_units"],
                    undetermined_units=row["undetermined_units"],
                )
                for c, row in score_report["per_config"].items()
                if row["total"] is None
            )
        blocked = [
            dict(config=t["config"], unit=t["unit"], reason="unmeasured_basic_quality")
            for t in ready
            if t not in admitted
        ]
        status = (
            ("partial" if failures or blocked else "complete")
            if judge
            else "ready_for_judge"
        )
        if not judge and (failures or blocked):
            status = "partial_ready_for_judge"
        report = dict(
            status=status,
            identity=identity,
            videos=len(tasks),
            admitted=len(admitted),
            failures=failures,
            blocked=blocked,
            judge_requested=judge,
            measurement_mode="remote"
            if remote
            else "archive"
            if quality_dir
            else "cpu",
            note="CPU 模式缺少 GPU 专家通道，缺失项保留未测量；第一关告警不直接淘汰配置",
        )
        store.write_json(store.root / "batch_report.json", report)
        return report
