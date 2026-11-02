import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from autobench.app.config import load
from autobench.app.data.store import Store
from autobench.app.data.types import ContractError, Event, Item, Plan, plan_to_dict
from autobench.app.batch.inputs import sha256
from autobench.app.batch.state import State, exclusive
from autobench.app.batch.run import run as batch
from autobench.app.remote.gpu.runner import Ledger, gpu_environment
from autobench.app.remote.collect import verify
from autobench.stages.basic_quality.steps import step02_three_routes as routes
from autobench.stages.basic_quality.experts.signal import brightness_step, measure
from autobench.stages.quality.steps.step02_percent import convert
from autobench.stages.score.run import run as score

UNIT = "Q__seed48"


def plan():
    return Plan(
        "Q",
        "a task",
        "image",
        (
            Event(
                "E1",
                "moves",
                dict(full="full", partial="partial", none="none"),
                "CA-05",
            ),
        ),
        (Item("G.body", "body", "MP-02", "vlm", None, "body is coherent"),),
    )


def result(open=False):
    return dict(
        events={"E1": dict(status="resolved", score=1.0, occurred=True)},
        items={"G.body": dict(status="unjudgeable" if open else "pass")},
    )


class RoutesTests(unittest.TestCase):
    def test_absolute_parent_and_group(self):
        rule = dict(worse_when="higher", absolute=10, p=0.3)
        summary = {
            "metrics": {
                "x": dict(
                    transform="raw",
                    log_scale=1,
                    rows=[
                        dict(model="base", value=1, flagged=False),
                        dict(model="new", value=6, flagged=True),
                    ],
                )
            }
        }
        flag = routes.evaluate("x", rule, "new", summary, {"new": "base"}, 3)
        self.assertTrue(flag.flagged)
        self.assertFalse(flag.routes["absolute"]["flagged"])
        self.assertTrue(flag.routes["parent"]["flagged"])
        self.assertTrue(flag.routes["group"]["flagged"])
        summary["metrics"]["x"]["rows"][1]["value"] = 11
        self.assertTrue(routes.evaluate("x", rule, "new", summary, {}, 3).flagged)

    def test_one_relative_is_suspect_and_alone_exception(self):
        summary = {
            "metrics": {"x": dict(rows=[dict(model="new", value=6, flagged=True)])}
        }
        rule = dict(worse_when="higher", p=0.3)
        flag = routes.evaluate("x", rule, "new", summary, {}, 3)
        self.assertFalse(flag.flagged)
        self.assertTrue(flag.suspect)
        self.assertTrue(
            routes.evaluate("x", dict(rule, alone=True), "new", summary, {}, 3).flagged
        )

    def test_all_missing_remains_unmeasured(self):
        self.assertIsNone(
            routes.evaluate(
                "x", dict(worse_when="higher", p=0.3), "new", {"metrics": {}}, {}, 3
            ).flagged
        )

    def test_hadm_requires_contiguous_time(self):
        rule = load().checks["thresholds"]["quality"]["detectors"]["human_structure"]
        for seconds, hit in ((0.9, False), (1.0, True)):
            s = {
                "metrics": {
                    "human_structure": {
                        "rows": [dict(model="T", value=40, longest_run_s=seconds)]
                    }
                }
            }
            self.assertEqual(
                routes.evaluate("human_structure", rule, "T", s, {}, 3).flagged, hit
            )

    def test_brightness_uses_sample_times_and_dict_thresholds(self):
        frames = [np.full((8, 8), v, np.uint8) for v in (20, 20, 80, 80, 80, 80)]
        data = brightness_step(frames, [0, 1, 2, 3, 4, 5])
        self.assertEqual(data["brightness_step_at"], 2)
        s = {
            "metrics": {
                k: {"rows": [dict(model="T", value=v)]}
                for k, v in data.items()
                if v is not None
            }
        }
        rule = load().checks["thresholds"]["quality"]["detectors"]["brightness_step"]
        self.assertTrue(routes.evaluate("brightness_step", rule, "T", s, {}, 3).flagged)


class ScoreTests(unittest.TestCase):
    def test_percentile_ties_missing_direction(self):
        r = dict(method="percentile", direction="higher")
        self.assertEqual(
            convert(dict(a=1, b=2, c=2, d=None), r), dict(a=0, b=0.75, c=0.75, d=None)
        )
        self.assertEqual(
            convert(dict(a=1, b=2), dict(r, direction="lower")), dict(a=1, b=0)
        )
        self.assertEqual(convert(dict(a=1), r), dict(a=None))

    def test_percentage_range_is_explicit(self):
        r = dict(method="percentage", direction="higher", range=[0, 100])
        self.assertEqual(convert(dict(a=73), r), dict(a=0.73))
        with self.assertRaises(ContractError):
            convert(dict(a=101), r)

    def test_three_equal_dimensions_and_board_identical(self):
        report = score(
            {"A": {UNIT: result()}, "B": {UNIT: result()}},
            {"Q": plan()},
            {"A": {UNIT: 0.2}, "B": {UNIT: 0.8}},
            load(),
        )
        self.assertEqual(report["board"]["ranked_order"], ["B", "A"])
        self.assertAlmostEqual(report["per_config"]["A"]["total"], 2.2 / 3)
        for c in ("A", "B"):
            self.assertEqual(
                report["board"]["rows"][c]["total"], report["per_config"][c]["total"]
            )
        self.assertGreater(report["board"]["rows"]["B"]["p_beats_next"], 0.99)

    def test_failed_and_open_videos_cannot_rank(self):
        report = score(
            {"A": {UNIT: result(True)}},
            {"Q": plan()},
            {"A": {UNIT: 1}},
            load(),
            {"A": [UNIT], "B": [UNIT]},
        )
        self.assertEqual(report["board"]["ranked_order"], [])
        self.assertIsNone(report["per_config"]["A"]["total"])
        self.assertEqual(report["per_config"]["B"]["missing_units"], [UNIT])

    def test_invalid_quality_rejected(self):
        with self.assertRaises(ContractError):
            score(
                {"A": {UNIT: result()}},
                {"Q": plan()},
                {"A": {UNIT: float("nan")}},
                load(),
            )


class StateTests(unittest.TestCase):
    def test_exact_resume_and_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load()
            c = replace(
                c, paths=dict(c.paths, local=dict(data_root=tmp, runs_subdir="runs"))
            )
            s = Store(c, "test")
            state = State(s)
            p = s.root / "value.json"
            s.write_json(p, {"x": 1})
            state.record("one", "id", "done", [p])
            self.assertTrue(state.reusable("one", "id", [p]))
            self.assertFalse(state.reusable("one", "changed", [p]))
            s.write_json(p, {"x": 2})
            self.assertFalse(state.reusable("one", "id", [p]))
            with exclusive(s.root):
                with self.assertRaises(ContractError):
                    with exclusive(s.root):
                        pass

    def test_gpu_capacity_release_and_uuid(self):
        ledger = Ledger({0: 40, 1: 80}, 3)
        self.assertEqual(ledger.reserve(32), 0)
        self.assertEqual(ledger.reserve(56), 1)
        ledger.release(0, 32)
        self.assertEqual(ledger.reserve(32), 0)
        with self.assertRaises(ContractError):
            ledger.reserve(100)
        with patch("subprocess.check_output", return_value="GPU-real-uuid"):
            self.assertEqual(
                gpu_environment(3)["CUDA_VISIBLE_DEVICES"], "GPU-real-uuid"
            )

    def test_collection_checks_hash_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data"
            p.write_bytes(b"data")
            verify(Path(tmp), {"data": sha256(p)})
            with self.assertRaises(ContractError):
                verify(Path(tmp), {"data": "bad"})
            with self.assertRaises(ContractError):
                verify(Path(tmp), {"../escape": "bad"})


class BatchTests(unittest.TestCase):
    def test_real_plan_and_judge_flow_with_scripted_client(self):
        class ScriptedClient:
            def __init__(self, vlm, cache_dir, recorder):
                self.recorder = recorder

            def ask_validated(self, name, prompt, frames, validate, stage):
                raw = prompt.split("本次输入", 1)[1].lstrip()
                payload = json.JSONDecoder().raw_decode(raw)[0]
                evidence = [f["id"] for f in frames if f["id"] != "reference"]
                events = [
                    dict(
                        id=e["id"],
                        score=1,
                        occurred=True,
                        start=0,
                        end=0.5,
                        reason="visible motion",
                        evidence=evidence,
                    )
                    for e in payload.get("events", [])
                ]
                items = [
                    dict(
                        id=i["id"],
                        status=1.0 if i["id"] == "G.end_state" else "pass",
                        reason="visible evidence",
                        evidence=evidence,
                    )
                    for i in payload.get("items", [])
                ]
                row = dict(events=events, items=items, overall="visible")
                validate(row)
                self.recorder.model_call(0, 0, cached=False)
                return row

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            image = images / "Q.jpg"
            cv2.imwrite(str(image), np.zeros((64, 64, 3), np.uint8))
            video = root / "video.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64)
            )
            for i in range(18):
                writer.write(np.full((64, 64, 3), 20 + i, np.uint8))
            writer.release()
            spec = dict(
                prompt_en="A cat walks.",
                subjects=[
                    dict(
                        id="cat",
                        description="black cat",
                        kind="animal",
                        person_type=None,
                        segment_word="cat",
                        source="in_picture",
                    )
                ],
                segments=[
                    dict(
                        actor="cat",
                        action="walks",
                        object=None,
                        destination=None,
                        destination_relation=None,
                        hand="none",
                        instrument=None,
                        start=0,
                        end=1,
                        result="The cat moves.",
                        sustain=False,
                        tests="",
                    )
                ],
                camera=dict(
                    motion="static", target="cat", amplitude="medium", speed="medium"
                ),
                end_frame=None,
            )
            specs = root / "specs.json"
            Store.write_json(
                specs,
                dict(
                    tasks=[dict(question_id="Q", image_sha256=sha256(image), spec=spec)]
                ),
            )
            manifest = root / "videos.json"
            Store.write_json(
                manifest, {"A": {UNIT: str(video)}, "B": {UNIT: str(video)}}
            )
            c = load()
            c = replace(
                c, paths=dict(c.paths, local=dict(data_root=tmp, runs_subdir="runs"))
            )
            store = Store(c, "full")
            with patch("autobench.app.run.Client", ScriptedClient):
                report = batch(
                    store,
                    c,
                    specs=specs,
                    image_dir=images,
                    manifest=manifest,
                    judge=True,
                )
            self.assertEqual(report["status"], "complete", report)
            self.assertEqual(
                store.read_json(store.score_file)["per_config"]["A"]["total"], 2.5 / 3
            )
            self.assertEqual(
                len(
                    store.read_json(store.board_dir / "leaderboard.json")[
                        "ranked_order"
                    ]
                ),
                2,
            )
            calls = sum(r["calls"] for r in store.read_cost())
            self.assertGreater(calls, 0)
            with patch(
                "autobench.app.run.Client", side_effect=AssertionError("must resume")
            ):
                report = batch(
                    store,
                    c,
                    specs=specs,
                    image_dir=images,
                    manifest=manifest,
                    judge=True,
                )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(sum(r["calls"] for r in store.read_cost()), calls)

    def test_cpu_batch_resume_and_missing_video_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image = image_dir / "Q.jpg"
            cv2.imwrite(str(image), np.zeros((64, 64, 3), np.uint8))
            video = root / "good.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64)
            )
            for i in range(18):
                frame = np.full((64, 64, 3), 40, np.uint8)
                cv2.circle(frame, (15 + i, 32), 8, (180, 180, 180), -1)
                writer.write(frame)
            writer.release()
            bad = root / "bad.mp4"
            bad.write_bytes(b"broken")
            manifest = root / "videos.json"
            manifest.write_text(
                json.dumps({"A": {UNIT: str(video)}, "B": {UNIT: str(bad)}})
            )
            specs = root / "specs.json"
            specs.write_text("{}")
            c = load()
            c = replace(
                c, paths=dict(c.paths, local=dict(data_root=tmp, runs_subdir="runs"))
            )
            store = Store(c, "batch")

            def make_plan(store, config, rec, specs, q, image_dir):
                p = replace(plan(), reference_sha256=sha256(image))
                store.write_json(store.plan_file(q), plan_to_dict(p))
                return p

            with patch("autobench.app.run.plan_one", side_effect=make_plan) as planner:
                report = batch(
                    store, c, specs=specs, image_dir=image_dir, manifest=manifest
                )
                self.assertEqual(planner.call_count, 1)
                self.assertIn("partial", report["status"])
                self.assertEqual(report["videos"], 2)
                batch(store, c, specs=specs, image_dir=image_dir, manifest=manifest)
                self.assertEqual(planner.call_count, 1)
            quality = store.read_json(store.root / "quality.json")
            self.assertIsNone(quality["scores"]["B"][UNIT])
            gate = store.read_json(store.root / "basic_quality.json")
            self.assertFalse(gate["per_video_admission"]["B"][UNIT]["allowed"])
            video.write_bytes(b"changed")
            with self.assertRaises(ContractError):
                batch(store, c, specs=specs, image_dir=image_dir, manifest=manifest)

    def test_short_black_video_does_not_emit_nan_or_fake_distortion(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "black.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (32, 32)
            )
            writer.write(np.zeros((32, 32, 3), np.uint8))
            writer.release()
            row = measure(video)
            self.assertIsNone(row["flicker"])
            self.assertIsNone(row["gen_artifact"])
            json.dumps(row, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
