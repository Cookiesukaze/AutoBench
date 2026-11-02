import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from autobench.app.config import load
from autobench.app.data.store import Store
from autobench.app.batch.inputs import sha256
from autobench.app.remote.gpu.runner import run_job
from autobench.app.remote.gpu.worker import main as worker
from autobench.app.remote.collect import verify, promote


class RemoteTests(unittest.TestCase):
    def test_promote_removes_stale_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            destination = Path(tmp) / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "current").write_bytes(b"new")
            (destination / "obsolete.npz").write_bytes(b"old")
            promote(source, destination)
            self.assertFalse((destination / "obsolete.npz").exists())
            self.assertEqual((destination / "current").read_bytes(), b"new")

    def test_remote_snapshot_launch_and_verified_collection(self):
        import shlex
        from dataclasses import replace
        from autobench.app.remote.stage import run

        class FakeRemote:
            def __init__(self, paths):
                self.token = ""
                self.root = ""

            def ssh(self, command):
                if " -c " in command:
                    args = shlex.split(command)
                    compile(args[2], "<spawn>", "exec")
                    compile(args[4], "<launcher>", "exec")
                    self.token = args[5]
                    return self.token
                if "/attempt.txt" in command:
                    return self.token
                if "/status.json" in command:
                    return json.dumps(dict(stage="DONE", errors=[]))
                if "/inventory.json" in command:
                    return json.dumps({"value.json": sha256(value)})
                return ""

            def upload(self, files, destination):
                self.root = destination
                self.assert_code = "code/autobench/app/remote/gpu/runner.py" in files
                if not self.assert_code:
                    raise AssertionError("missing snapshot")
                job = json.loads(files["job.json"].read_text(encoding="utf-8"))
                if not job["tasks"][0]["video"].startswith(destination + "/videos/"):
                    raise AssertionError("video not remapped")

            def download(self, source, destination):
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "value.json").write_bytes(value.read_bytes())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = root / "value"
            value.write_bytes(b"{}")
            video = root / "video"
            video.write_bytes(b"video")
            c = load()
            c = replace(
                c, paths=dict(c.paths, local=dict(data_root=tmp, runs_subdir="runs"))
            )
            store = Store(c, "snapshot")
            with patch("autobench.app.remote.stage.Remote", FakeRemote):
                collected, status = run(
                    store,
                    c,
                    [
                        dict(
                            config="A",
                            unit="Q__seed48",
                            video=str(video),
                            video_sha256=sha256(video),
                        )
                    ],
                    "identity",
                )
            self.assertEqual(status["stage"], "DONE")
            self.assertEqual((collected / "value.json").read_bytes(), b"{}")

    def test_persistent_expert_waves_dependencies_and_inventory(self):
        launches = []

        class Process:
            def __init__(self, argv, **kwargs):
                self.argv = argv
                self.returncode = None
                launches.append(argv[argv.index("--expert") + 1])

            def wait(self, timeout=None):
                expert = self.argv[self.argv.index("--expert") + 1]
                queue = Path(self.argv[self.argv.index("--queue") + 1])
                for path in queue.glob("*.job.json"):
                    spec = json.loads(path.read_text(encoding="utf-8"))
                    out = Path(spec["output"])
                    obs = {}
                    if expert == "sam31":
                        mask = np.zeros((1, 64, 64), bool)
                        mask[:, 16:48, 16:48] = True
                        np.savez_compressed(
                            out.with_suffix(".masks.npz"), masks_0=mask, masks_16=mask
                        )
                    elif expert == "cotracker3":
                        points = json.loads(Path(spec["targets"]).read_text())["points"]
                        n = len(points)
                        tracks = np.zeros((18, n, 2), np.float32)
                        for i, p in enumerate(points):
                            tracks[:, i] = [p["x"] * 63, p["y"] * 63]
                        np.savez_compressed(
                            out.with_suffix(".tracks.npz"),
                            tracks=tracks,
                            visibility=np.ones((18, n), bool),
                            labels=np.array([p["label"] for p in points]),
                            fps=12.0,
                        )
                    elif expert == "signal":
                        obs = {"metrics": {"sharpness": 100.0, "flicker_v2": 1.0}}
                    elif expert == "musiq":
                        obs = {"metrics": {"technical_quality": 70.0}}
                    elif expert == "hadm":
                        obs = {"metrics": {"human_structure": 12}}
                        rows = [
                            json.dumps(
                                dict(
                                    frame=i,
                                    seconds=i / 12,
                                    maximum=0.6 if i < 12 else 0,
                                )
                            )
                            for i in range(18)
                        ]
                        out.with_suffix(".frames.jsonl").write_text(chr(10).join(rows))
                    Store.write_json(out, dict(status="observed", observations=obs))
                self.returncode = 0
                return 0

            def poll(self):
                return self.returncode

        def query(argv, **kwargs):
            return (
                "0, 81920, 0"
                if "--query-gpu=index,memory.total,memory.used" in argv
                else "GPU-physical"
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64)
            )
            rng = np.random.default_rng(1)
            frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            for _ in range(18):
                writer.write(frame)
            writer.release()
            tasks = [
                dict(
                    config=c,
                    unit="Q__seed48",
                    video=str(video),
                    video_sha256=sha256(video),
                    concepts=["cat"],
                    descriptions={"cat": "black cat"},
                    pose=False,
                )
                for c in ("A", "B")
            ]
            c = load()
            job = root / "job.json"
            Store.write_json(
                job,
                dict(
                    tasks=tasks,
                    remote=c.paths["remote"],
                    gpu_config=c.gpus,
                    checks=c.checks,
                    gpus=[0],
                ),
            )
            with patch("subprocess.check_output", side_effect=query), patch(
                "subprocess.Popen", Process
            ):
                run_job(job)
            self.assertEqual(len(launches), 7)
            self.assertEqual(launches[0], "sam31")
            self.assertEqual(launches.count("cotracker3"), 1)
            status = json.loads((root / "out/status.json").read_text())
            self.assertEqual(status["stage"], "DONE", status)
            summary = json.loads(
                (root / "out/quality/Q__seed48/summary.json").read_text()
            )
            self.assertEqual(
                summary["metrics"]["human_structure"]["rows"][0]["longest_run_s"], 1.0
            )
            self.assertEqual(summary["inputs"]["B"], sha256(video))
            verify(root / "out", json.loads((root / "inventory.json").read_text()))

    def test_worker_reuses_verified_observation_and_retries_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue"
            queue.mkdir()
            video = root / "video"
            video.write_bytes(b"video")
            out = root / "signal.json"
            spec = {"video": str(video), "output": str(out)}
            args = [
                "--expert",
                "signal",
                "--evidence",
                str(root / "manifests"),
                "--sources",
                str(root),
                "--prompts-dir",
                str(root),
                "--queue",
                str(queue),
            ]

            def submit():
                Store.write_json(queue / "000.job.json", spec)
                (queue / "STOP").touch()
                worker(args)

            with patch(
                "autobench.stages.basic_quality.experts.signal.measure",
                return_value={"sharpness": 2},
            ) as model:
                submit()
                submit()
                self.assertEqual(model.call_count, 1)
                video.write_bytes(b"changed")
                submit()
                self.assertEqual(model.call_count, 2)
            video.write_bytes(b"broken")
            with patch(
                "autobench.stages.basic_quality.experts.signal.measure",
                side_effect=ValueError("broken"),
            ):
                submit()
            self.assertEqual(json.loads(out.read_text())["status"], "unavailable")
            with patch(
                "autobench.stages.basic_quality.experts.signal.measure",
                return_value={"sharpness": 3},
            ):
                submit()
            self.assertEqual(json.loads(out.read_text())["status"], "observed")


if __name__ == "__main__":
    unittest.main()
