"""本地可复现的视频测量；仅输出原始指标，不负责三路判定。"""

from pathlib import Path
import cv2
import numpy as np

from .distortion import _gen_artifact
from .flicker import flicker, flicker_v2_channel


def sampled_gray(video: Path, count: int = 6):
    capture = cv2.VideoCapture(str(video))
    frames, times = [], []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if total < 1 or fps <= 0:
            raise ValueError(f"无效视频：{video}")
        for index in sorted(
            set(np.linspace(0, total - 1, min(count, total)).round().astype(int))
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"视频帧无法解码：{video} frame={index}")
            gray = cv2.cvtColor(cv2.resize(frame, (448, 448)), cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            times.append(float(index / fps))
    finally:
        capture.release()
    return frames, times


from .brightness import brightness_step


def measure(video: Path):
    frames, times = sampled_gray(video)
    result = {
        "sharpness": float(
            np.median([cv2.Laplacian(g, cv2.CV_64F).var() for g in frames])
        )
    }
    result.update(brightness_step(frames, times))
    luma = flicker(video)
    result["flicker"] = luma["luma_rms"]
    result["flicker_v2"] = flicker_v2_channel(video)
    result.update(_gen_artifact(video))
    return result
