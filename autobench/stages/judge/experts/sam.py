"""SAM 3.1 掩码：主体的实例分割与面积。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.types import ContractError


def load(folder: Path, concept: str) -> np.lib.npyio.NpzFile:
    path = folder / f"sam31.{concept}.masks.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path, allow_pickle=True)


def load_main(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "sam31.masks.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path, allow_pickle=True)


def frame_keys(masks) -> list[int]:
    return sorted(int(k[6:]) for k in masks.files if k.startswith("masks_"))


def mask_near(masks, index: int) -> np.ndarray | None:
    """离给定帧最近的那张掩码，合成成一张布尔图。"""
    keys = frame_keys(masks)
    if not keys:
        return None
    near = masks["masks_%d" % min(keys, key=lambda k: abs(k - index))]
    return near.any(0) if near.ndim == 3 and len(near) else None


def instance_boxes(masks, index: int, min_area: int = 30) -> list[tuple[int, int, int, int]]:
    keys = frame_keys(masks)
    if not keys:
        return []
    near = masks["masks_%d" % min(keys, key=lambda k: abs(k - index))]
    if near.ndim != 3 or not len(near):
        return []
    boxes = []
    for inst in near:
        ys, xs = np.nonzero(inst)
        if len(ys) >= min_area:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def instances_left_frame(z, prev_frame: int, cur_frame: int, margin: int = 8) -> bool:
    """上一帧还在、这一帧消失的实例，是不是从画面边缘走出去的。"""
    try:
        prev_ids = [int(x) for x in z[f"ids_{prev_frame}"]]
        cur_ids = [int(x) for x in z[f"ids_{cur_frame}"]]
    except KeyError:
        return False
    gone = [pid for pid in prev_ids if pid not in cur_ids]
    if not gone:
        return False
    masks = z[f"masks_{prev_frame}"]
    if masks.ndim != 3:
        return False
    h, w = masks.shape[1], masks.shape[2]
    for pid in gone:
        index = prev_ids.index(pid)
        if index >= masks.shape[0]:
            continue
        ys, xs = np.where(masks[index])
        if not len(xs):
            continue
        if xs.min() < margin or xs.max() > w - margin or ys.min() < margin or ys.max() > h - margin:
            return True
    return False


def covered_by_other(folder: Path, frame_index: int, gone: np.ndarray, own_file: Path) -> float:
    """消失的像素里，有多大比例被别的概念的掩码盖住。遮挡不是合并。"""
    total = int(gone.sum())
    if not total:
        return 0.0
    best = 0.0
    for path in sorted(Path(folder).glob("sam31.*.masks.npz")):
        if path == own_file:
            continue
        try:
            near = mask_near(np.load(path, allow_pickle=True), frame_index)
        except (OSError, ValueError):
            continue
        if near is None or not near.any():
            continue
        best = max(best, float((gone & near).sum()) / total)
    return best
