import sys
import json


class HADM:
    def __init__(self, repo, weights, variant):
        import os
        import torch
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.engine.defaults import DefaultInferencer
        from detectron2.modeling import ema
        from detectron2.data.datasets.human_artifact_dataset import (
            LOCAL_HUMAN_ARTIFACT_CATEGORIES,
            GLOBAL_HUMAN_ARTIFACT_CATEGORIES,
        )

        assert torch.cuda.is_available(), "CUDA required"
        os.chdir(repo)
        name = "local" if variant == "L" else "global"
        self.cfg = LazyConfig.load(
            f"projects/ViTDet/configs/eva2_o365_to_coco/demo_{name}.py"
        )
        self.cfg.train.init_checkpoint = str(weights)
        self.cfg.train.model_ema.enabled = True
        self.cfg.train.model_ema.use_ema_weights_for_eval_only = True
        model = instantiate(self.cfg.model).to("cuda")
        ema.may_build_model_ema(self.cfg, model)
        checkpoint = DetectionCheckpointer(
            model, **ema.may_get_ema_checkpointer(self.cfg, model)
        )
        checkpoint.load(str(weights))
        assert model.ema_state.has_inited(), "EMA missing"
        ema.apply_model_ema(model)
        self.predictor = DefaultInferencer(self.cfg, model)
        categories = (
            LOCAL_HUMAN_ARTIFACT_CATEGORIES
            if variant == "L"
            else GLOBAL_HUMAN_ARTIFACT_CATEGORIES
        )
        self.classes = [v["class_name"] for v in categories]

    def detect(self, bgr):
        import torch

        with torch.inference_mode():
            instances = self.predictor(bgr)["instances"].to("cpu")
        return [
            dict(box=box, score=score, class_id=cls, label=self.classes[cls])
            for box, score, cls in zip(
                instances.pred_boxes.tensor.tolist(),
                instances.scores.tolist(),
                instances.pred_classes.tolist(),
            )
        ]


def measure(video, output, repo, weights):
    import cv2
    from ....app.remote.gpu.worker import cached, sha

    sys.path.insert(0, str(repo))
    expert = cached(("hadm", str(weights)), lambda: HADM(repo, weights, "L"))
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    rows = []
    try:
        if fps <= 0:
            raise ValueError("HADM 输入 fps 无效")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = expert.detect(frame)
            rows.append(
                {
                    "frame": len(rows),
                    "seconds": len(rows) / fps,
                    "maximum": max((d["score"] for d in detections), default=0.0),
                    "detections": detections,
                }
            )
    finally:
        capture.release()
    if not rows:
        raise ValueError("HADM 无可解码帧")
    artifact = output.with_suffix(".frames.jsonl")
    artifact.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {
        "artifact": str(artifact),
        "artifact_sha256": sha(artifact),
        "weight_sha256": sha(weights),
        "fps": fps,
        "frames": len(rows),
        "metrics": {"human_structure": sum(r["maximum"] >= 0.5 for r in rows)},
    }
