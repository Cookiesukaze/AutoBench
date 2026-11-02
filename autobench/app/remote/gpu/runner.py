"""远端专家调度：分割后构造跟踪点，模型按专家常驻，逐卡预约显存。"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from ...batch.inputs import sha256, safe_name
from ...data.store import Store
from ...data.types import ContractError
from ....stages.judge.tools.expert import points


class Ledger:
    def __init__(self, free_gib, headroom):
        self.room = {int(g): v - headroom for g, v in free_gib.items()}
        self.capacity = dict(self.room)
        self.condition = threading.Condition()

    def reserve(self, need):
        with self.condition:
            if not self.room or need > max(self.capacity.values()):
                raise ContractError(f"无单卡可容纳 {need} GiB")
            while True:
                fits = [g for g, room in self.room.items() if room >= need]
                if fits:
                    gpu = min(fits, key=lambda g: self.room[g])
                    self.room[gpu] -= need
                    return gpu
                self.condition.wait()

    def release(self, gpu, need):
        with self.condition:
            self.room[gpu] += need
            self.condition.notify_all()


def gpu_environment(gpu):
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not uuid.startswith("GPU-") or chr(10) in uuid:
        raise ContractError("无法解析物理 GPU UUID")
    return dict(
        os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1"
    )


def _run_job(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    root = Path(job_path).parent
    output = root / "out"
    output.mkdir(exist_ok=True)
    status = output / "status.json"
    inventory_path = root / "inventory.json"
    if inventory_path.exists():
        inventory_path.unlink()
    Store.write_json(status, {"stage": "running"})
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {}
    for line in rows.splitlines():
        g, total, used = [int(x.strip()) for x in line.split(",")]
        if g in job["gpus"] and used < job["gpu_config"]["free_mib"]:
            free[g] = (total - used) / 1024
    ledger = Ledger(free, job["gpu_config"]["headroom_gib"])
    paths = job["remote"]
    processes = []
    process_lock = threading.Lock()
    errors = []

    def pool(expert, specs):
        if not specs:
            return
        need = job["gpu_config"]["budget_gib"][expert] if expert != "signal" else 0.0
        gpu = None
        queue = root / "queues" / expert
        queue.mkdir(parents=True, exist_ok=True)
        for i, spec in enumerate(specs):
            Store.write_json(queue / f"{i:06d}.job.json", spec)
        (queue / "STOP").touch()
        interpreter = paths["expert_python"].get(expert, "tracker")
        python = str(Path(paths["base"]) / paths["python"][interpreter])
        argv = [
            python,
            "-c",
            "from autobench.app.remote.gpu.worker import main;main()",
            "--expert",
            expert,
            "--evidence",
            str(root / "manifests"),
            "--sources",
            str(Path(paths["base"]) / paths["third_party"]),
            "--prompts-dir",
            str(root / "code/autobench/prompt/quality"),
            "--queue",
            str(queue),
            "--hadm-repo",
            str(Path(paths["base"]) / paths["hadm"]["repo"]),
            "--hadm-weights",
            str(Path(paths["base"]) / paths["hadm"]["weights"]),
        ]
        if paths.get("dover_repo"):
            argv.extend(
                ["--dover-repo", str(Path(paths["base"]) / paths["dover_repo"])]
            )
        log = output / (expert + ".log")
        argv.extend(
            [
                "--musiq-weights",
                str(Path(paths["base"]) / paths["musiq"]["weights"]),
                "--musiq-sha256",
                paths["musiq"]["sha256"],
            ]
        )
        proc = None
        try:
            gpu = ledger.reserve(need) if need else None
            env = (
                gpu_environment(gpu)
                if gpu is not None
                else dict(os.environ, PYTHONUNBUFFERED="1")
            )
            with log.open("w") as stream:
                proc = subprocess.Popen(
                    argv,
                    cwd=root / "code",
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                with process_lock:
                    processes.append(proc)
                if proc.wait(timeout=paths["timeouts"]["poll_limit_s"]):
                    raise ContractError(f"{expert} 进程失败，见 {log}")
            for spec in specs:
                path = Path(spec["output"])
                if not path.exists():
                    raise ContractError(f"{expert} 缺少观测记录 {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] != "observed":
                    errors.append(
                        {
                            "expert": expert,
                            "output": str(path),
                            "error": record.get("error"),
                        }
                    )
        except Exception as exc:
            errors.append({"expert": expert, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if gpu is not None:
                ledger.release(gpu, need)

    experts = [
        "sam31",
        "cotracker3",
        "waft",
        "dinov3",
        "pose",
        "signal",
        "musiq",
        "hadm",
    ]
    if paths.get("dover_repo"):
        experts.append("dover")
    jobs = {e: [] for e in experts}
    try:
        for task in job["tasks"]:
            folder = output / "evidence" / f"{task['config']}_{task['unit']}"
            folder.mkdir(parents=True, exist_ok=True)
            task["folder"] = str(folder)
            if sha256(Path(task["video"])) != task["video_sha256"]:
                raise ContractError("上传视频 SHA256 不符")
            Store.write_json(
                folder / "identity.json",
                {k: task[k] for k in ("config", "unit", "video_sha256")},
            )
            Store.write_json(folder / "config.json", {"config": task["config"]})
            for concept in task["concepts"]:
                safe_name(concept)
                jobs["sam31"].append(
                    dict(
                        video=task["video"],
                        output=str(folder / f"sam31.{concept}.json"),
                        concept=concept,
                        description=task["descriptions"].get(concept, ""),
                    )
                )
        pool("sam31", jobs["sam31"])
        area_stats = {}
        for task in job["tasks"]:
            stats = area_stats.setdefault(task["unit"], {})
            concepts = {}
            for concept in task["concepts"]:
                path = Path(task["folder"]) / f"sam31.{concept}.masks.npz"
                if path.exists():
                    with np.load(path) as masks:
                        areas = [
                            int(masks[k].sum())
                            for k in masks.files
                            if k.startswith("masks_")
                        ]
                    concepts[concept] = dict(
                        median_area=float(np.median(areas)) if areas else 0
                    )
            stats[task["config"]] = dict(concepts=concepts)
        for task in job["tasks"]:
            Store.write_json(
                Path(task["folder"]) / "area_stats.json", area_stats[task["unit"]]
            )
        quality_targets = {}
        for task in job["tasks"]:
            folder = Path(task["folder"])
            masks = {}
            for concept in task["concepts"]:
                path = folder / f"sam31.{concept}.masks.npz"
                if path.exists():
                    masks[concept] = np.load(path)
            cap = cv2.VideoCapture(task["video"])
            ok, frame = cap.read()
            cap.release()
            if not ok:
                for z in masks.values():
                    z.close()
                errors.append(
                    dict(config=task["config"], unit=task["unit"], error="无法解码视频")
                )
                continue
            targets, notes = points.build_targets(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), masks, task["concepts"]
            )
            for z in masks.values():
                z.close()
            target_path = folder / "targets.json"
            points.write_targets(targets, target_path)
            Store.write_json(folder / "points_notes.json", notes)
            quality_targets.setdefault(task["unit"], str(target_path))
            for expert in (e for e in experts if e != "sam31"):
                if expert == "pose" and not task["pose"]:
                    continue
                spec = dict(
                    video=task["video"], output=str(folder / (expert + ".json"))
                )
                if expert == "cotracker3":
                    spec["targets"] = str(target_path)
                jobs[expert].append(spec)
            jobs["cotracker3"].append(
                dict(
                    video=task["video"],
                    output=str(folder / "quality_motion.json"),
                    targets=quality_targets[task["unit"]],
                )
            )
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [
                executor.submit(pool, e, specs)
                for e, specs in jobs.items()
                if e != "sam31"
            ]
            for future in futures:
                future.result()
        canonical = {
            "cotracker3.tracks.npz": "cotracker3.tracks.npz",
            "waft.flow.npz": "waft.flow.npz",
            "pose.pose.npz": "pose.keypoints.npz",
            "dinov3.features.npz": "dinov3.features.npz",
        }
        for task in job["tasks"]:
            folder = Path(task["folder"])
            for src, dst in canonical.items():
                if src != dst and (folder / src).exists():
                    shutil.copy2(folder / src, folder / dst)
        from ....stages.basic_quality.steps.step01_measure import build_summaries

        build_summaries(job["tasks"], output / "quality", job["checks"])
        Store.write_json(
            status, {"stage": "PARTIAL" if errors else "DONE", "errors": errors}
        )
    except BaseException:
        import traceback

        Store.write_json(status, {"stage": "FAILED", "error": traceback.format_exc()})
        raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    inventory = {
        str(p.relative_to(output)).replace("\\", "/"): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    Store.write_json(root / "inventory.json", inventory)


def run_job(job_path):
    try:
        return _run_job(job_path)
    except BaseException:
        import traceback

        Store.write_json(
            Path(job_path).parent / "out/status.json",
            dict(stage="FAILED", error=traceback.format_exc()),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    run_job(parser.parse_args().job)


if __name__ == "__main__":
    main()
