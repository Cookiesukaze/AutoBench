"""远端常驻专家进程：校验权重、CUDA 推理、保留原始观测和失败。"""

from __future__ import annotations
import argparse
import json
import time
import traceback
import importlib.metadata
from pathlib import Path
import cv2
import numpy as np
from ...batch.inputs import sha256 as sha
from ...batch.inputs import fingerprint
from ...data.store import Store

_MODELS = {}


def cached(key, build):
    if key not in _MODELS:
        _MODELS[key] = build()
    return _MODELS[key]


def weight_sha(path):
    path = Path(path)
    stat = path.stat()
    return cached(
        ("weight-sha", str(path), stat.st_size, stat.st_mtime_ns), lambda: sha(path)
    )


def verify_manifest(path, expert):
    evidence = json.loads(path.read_text(encoding="utf-8"))
    files = evidence.get("files", [])
    if expert == "cotracker3":
        files = [f for f in files if f["name"] == "scaled_offline.pth"]
        if len(files) != 1:
            raise ValueError("CoTracker 离线权重验证缺失")
    elif evidence.get("status") != "downloaded_verified" or not files:
        raise ValueError("专家权重未经完整验证")
    if expert == "waft":
        assets = evidence.get("constructor_assets", {})
        auxiliary = assets.get("depth", {}).get("files", []) + assets.get(
            "timm_assets", []
        )
        if not auxiliary:
            raise ValueError("WAFT 构造依赖验证缺失")
        files = files + auxiliary
    for entry in files:
        if weight_sha(entry["path"]) != entry["sha256"]:
            raise ValueError("专家权重 SHA256 不符")
    return evidence


def decode(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames or fps <= 0:
        raise ValueError(f"无效视频：{path}")
    return np.stack(frames), fps


def observe(args, spec):
    from ....stages.judge.experts import cotracker, dinov, pose, sam, waft
    from ....stages.basic_quality.experts import signal
    from ....stages.quality.experts import musiq

    output = Path(spec["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {"expert": args.expert, "status": "running"}
    try:
        record["video_sha256"] = sha(Path(spec["video"]))
        identity = dict(spec=spec, video=record["video_sha256"], expert=args.expert)
        if spec.get("targets"):
            identity["targets"] = sha(Path(spec["targets"]))
        manifest_path = Path(args.evidence) / (args.expert + ".json")
        if manifest_path.exists():
            identity["manifest"] = sha(manifest_path)
            if args.expert in ("cotracker3", "waft", "sam31", "dinov3", "pose"):
                verify_manifest(manifest_path, args.expert)
        if args.expert == "hadm":
            identity["weights"] = cached(
                ("weight-sha", args.hadm_weights), lambda: sha(Path(args.hadm_weights))
            )
        record["identity"] = fingerprint(identity)
        if args.expert == "musiq":
            identity["weights"] = cached(
                ("weight-sha", args.musiq_weights),
                lambda: sha(Path(args.musiq_weights)),
            )
            if identity["weights"] != args.musiq_sha256:
                raise ValueError("MUSIQ 权重 SHA256 不符")
            record["identity"] = fingerprint(identity)
        if output.exists():
            previous = json.loads(output.read_text(encoding="utf-8"))
            observations = previous.get("observations", {})
            artifact = observations.get("artifact")
            valid = not artifact or (
                Path(artifact).is_file()
                and sha(Path(artifact)) == observations.get("artifact_sha256")
            )
            if (
                previous.get("status") == "observed"
                and previous.get("identity") == record["identity"]
                and valid
            ):
                return previous
        options = argparse.Namespace(
            **spec, prompts_dir=Path(args.prompts_dir), sources=Path(args.sources)
        )
        options.video, options.output = Path(options.video), output
        options.targets = Path(spec["targets"]) if spec.get("targets") else None
        options.concept = spec.get("concept")
        options.description = spec.get("description", "")
        if args.expert == "signal":
            observations = {"metrics": signal.measure(options.video)}
        elif args.expert == "musiq":
            observations = {
                "metrics": {
                    "technical_quality": musiq.measure(
                        options.video, Path(args.musiq_weights)
                    )
                }
            }
        elif args.expert == "hadm":
            from ....stages.basic_quality.experts.hadm import measure

            observations = measure(
                options.video, output, Path(args.hadm_repo), Path(args.hadm_weights)
            )
        elif args.expert == "dover":
            from ....stages.basic_quality.experts.dover import measure

            observations = {"metrics": measure(options.video, args.dover_repo)}
        else:
            manifest = Path(args.evidence) / (args.expert + ".json")
            evidence = verify_manifest(manifest, args.expert)
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("专家要求 CUDA；禁止自动切换 CPU")
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            source = Path(evidence.get("source_path") or options.sources / args.expert)
            modelpath = Path(evidence["model_path"])
            functions = {
                "cotracker3": cotracker.track,
                "dinov3": dinov.dinov3,
                "pose": pose.pose,
                "sam31": sam.sam31,
                "waft": waft.waft,
            }
            if args.expert == "dinov3":
                observations = functions[args.expert](options, modelpath, torch)
            else:
                observations = functions[args.expert](options, modelpath, source, torch)
            torch.cuda.synchronize()
            record.update(
                manifest_sha256=sha(manifest),
                revision=evidence["revision"],
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
        record.update(status="observed", observations=observations)
        record["versions"] = {}
        for package in ("numpy", "opencv-python", "torch", "pyiqa", "transformers"):
            try:
                record["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        json.dumps(record, allow_nan=False)
    except Exception as exc:
        record.update(
            status="unavailable",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    record["elapsed_s"] = time.monotonic() - started
    if record["status"] == "unavailable":
        record.pop("observations", None)
        for suffix in (
            ".tracks.npz",
            ".masks.npz",
            ".features.npz",
            ".pose.npz",
            ".frames.jsonl",
            ".flow.npz",
        ):
            artifact = output.with_suffix(suffix)
            if artifact.exists():
                artifact.unlink()
        if args.expert == "pose" and (output.parent / "pose.keypoints.npz").exists():
            (output.parent / "pose.keypoints.npz").unlink()
    Store.write_json(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert",
        required=True,
        choices=(
            "cotracker3",
            "dinov3",
            "pose",
            "sam31",
            "waft",
            "signal",
            "musiq",
            "hadm",
            "dover",
        ),
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--hadm-repo")
    parser.add_argument("--hadm-weights")
    parser.add_argument("--dover-repo")
    parser.add_argument("--musiq-weights")
    parser.add_argument("--musiq-sha256")
    args = parser.parse_args(argv)
    args.queue.mkdir(parents=True, exist_ok=True)
    while True:
        jobs = sorted(args.queue.glob("*.job.json"))
        if not jobs:
            if (args.queue / "STOP").exists():
                return
            time.sleep(0.25)
            continue
        job = jobs[0]
        taken = job.with_suffix(".taken")
        try:
            job.rename(taken)
        except FileNotFoundError:
            continue
        row = observe(args, json.loads(taken.read_text(encoding="utf-8")))
        taken.with_suffix(".done").write_text(
            json.dumps({"status": row["status"]}), encoding="utf-8"
        )
        taken.unlink()


if __name__ == "__main__":
    main()
