"""抽帧：同一道题的所有视频，每个窗口抽一样多帧。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jpeg(path: Path, frame: np.ndarray, quality: int) -> None:
    # OpenCV's Windows filename API cannot reliably open Unicode paths.
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(str(path))
    path.write_bytes(encoded.tobytes())


def frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def locate_indices(count: int, per_video: int) -> list[int]:
    """等间隔取帧，最后一帧一定在内。"""
    if count <= per_video:
        return list(range(count))
    return sorted({round((count - 1) * k / (per_video - 1)) for k in range(per_video)})


def window_indices(start: float | None, end: float | None, fps: float, count: int,
                   per_window: int) -> list[int]:
    if start is None or end is None or end < start:
        return []
    first, last = max(0, round(start * fps)), min(count - 1, round(end * fps))
    if per_window == 1:
        return [(first + last) // 2]
    if first >= last:
        return [first] * per_window
    return [round(first + (last - first) * k / (per_window - 1)) for k in range(per_window)]


def tail_indices(count: int, fps: float, per_window: int) -> list[int]:
    step = fps / max(1, per_window - 1)
    return sorted({max(0, count - 1 - round(k * step)) for k in range(per_window)})


def padded(start: float, end: float, duration: float, margin: float) -> tuple[float, float]:
    return max(0.0, start - margin), min(duration, end + margin)


def fusion_indices(windows: dict[str, list[int]], count: int, fps: float,
                   per_window: int) -> list[int]:
    indices: list[int] = []
    for event_id in sorted(windows):
        indices += windows[event_id]
    return sorted(set(indices + tail_indices(count, fps, per_window)))


def review_indices(window: list[float], fps: float, count: int, per_window: int,
                   margin: float, sample: int = 0, samples: int = 1) -> list[int]:
    """复核窗口内的取样点。

    sample/samples > 1 时不是"同一批帧问几遍"，而是把窗口切成 per_window×samples 个点，
    第 k 个视角取第 k、k+samples、k+2·samples… 个点：三个视角合起来是一张稠密网格，各自仍只送 per_window 张。
    """
    start, end = window
    first, last = max(0, round((start - margin) * fps)), min(count - 1, round((end + margin) * fps))
    if 0 <= last - first + 1 <= per_window:
        return list(range(first, last + 1))
    grid = window_indices(max(0.0, start - margin), end + margin, fps, count,
                          per_window * samples)
    return sorted(set(grid[sample % samples::samples]))


def sheet(records: list[dict], root: Path, cols: int, rows: int, side: int, gap: int,
          quality: int) -> list[dict]:
    """把若干帧拼成一张 contact sheet：每格左上角红字帧号，空位留白。

    返回的每张图带 tiles（这份图里有哪些帧号），模型引用证据仍用帧号，不用格子位置。
    """
    if not records:
        raise ValueError("没有帧可以拼接")
    root.mkdir(parents=True, exist_ok=True)
    per_sheet = cols * rows
    sheets: list[dict] = []
    for start in range(0, len(records), per_sheet):
        group = records[start:start + per_sheet]
        tiles = [cv2.imdecode(np.frombuffer(Path(record["path"]).read_bytes(),
                                          dtype=np.uint8), cv2.IMREAD_COLOR)
                 for record in group]
        if any(tile is None for tile in tiles):
            raise ValueError("有帧读不出来，不能拼接")
        height, width = tiles[0].shape[:2]
        canvas = np.full((rows * height + (rows + 1) * gap, cols * width + (cols + 1) * gap, 3),
                         255, dtype=np.uint8)
        for i, (tile, record) in enumerate(zip(tiles, group)):
            row, col = divmod(i, cols)
            x0, y0 = gap + col * (width + gap), gap + row * (height + gap)
            canvas[y0:y0 + height, x0:x0 + width] = tile
        scale = min(1.0, side / max(canvas.shape[1], canvas.shape[0]))
        if scale < 1:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # 帧号在缩放之后才写：整张缩小一半时，格子里的字号不能跟着缩小，否则读不出来。
        font = max(0.5, min(1.6, canvas.shape[1] / 900))
        for i, record in enumerate(group):
            row, col = divmod(i, cols)
            x0 = int((gap + col * (width + gap)) * scale) + 6
            y0 = int((gap + row * (height + gap)) * scale) + int(26 * font) + 8
            cv2.putText(canvas, record["id"], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 0, 255), max(1, int(2 * font)), cv2.LINE_AA)
        path = root / ("sheet%02d_%s.jpg" % (len(sheets), _digest(json.dumps(
            [r["id"] for r in group]).encode())[:8]))
        _write_jpeg(path, canvas, quality)
        sheets.append({"id": "S%d" % len(sheets), "tiles": [r["id"] for r in group],
                       "seconds": [r["seconds"] for r in group],
                       "layout": "%d 列 × %d 行，从左到右、从上到下" % (cols, rows),
                       "path": str(path), "sha256": _digest(path.read_bytes())})
    return sheets


def extract(video: Path, root: Path, indices: list[int], quality: int,
            crop: tuple[float, float, float, float] | None = None,
            prefix: str = "F"):
    """把帧写成 JPEG，返回帧记录和这条视频的时钟。

    prefix 用来区分同一帧号的两种看法：原图（F）与放大后的裁切（Z）。
    """
    capture = cv2.VideoCapture(str(video))
    fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count < 1:
        capture.release()
        raise ValueError(f"无法解码视频 {video}")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < count:
                raise ValueError(f"帧号 {index} 超出视频范围")
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"无法解出第 {index} 帧")
            if crop:
                x0, y0, x1, y1 = crop
                h, w = frame.shape[:2]
                frame = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
                if frame.size == 0:
                    raise ValueError("裁剪区域为空")
            path = root / ("f%05d_%s.jpg" % (index, _digest(json.dumps([index, crop]).encode())[:10]))
            _write_jpeg(path, frame, quality)
            records.append({"id": "%s%d" % (prefix, index), "index": index,
                            "seconds": round(index / fps, 3),
                            "crop_xyxy_normalized": crop, "path": str(path),
                            "sha256": _digest(path.read_bytes())})
    finally:
        capture.release()
    return records, {"fps": fps, "frames": count, "last_frame_seconds": (count - 1) / fps}
