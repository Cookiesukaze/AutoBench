"""MUSIQ 原始画质分；模型在专家进程内复用。"""

import numpy as np
import cv2
from ....app.remote.gpu.worker import cached, decode


def measure(video, weights):
    import torch
    import pyiqa

    if not torch.cuda.is_available():
        raise RuntimeError("MUSIQ 要求 CUDA")
    model = cached(
        ("musiq", str(weights)),
        lambda: pyiqa.create_metric(
            "musiq", device="cuda", pretrained_model_path=str(weights)
        ),
    )
    frames, _ = decode(video)
    values = []
    for index in (
        np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    ):
        frame = cv2.resize(frames[index], (448, 448), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()[None].cuda() / 255
        with torch.inference_mode():
            values.append(float(model(tensor).flatten()[0]))
    return float(np.median(values))
