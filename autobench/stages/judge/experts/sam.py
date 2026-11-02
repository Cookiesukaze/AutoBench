"""SAM 3.1 掩码：主体的实例分割与面积。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.data.types import ContractError


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


def instance_boxes(
    masks, index: int, min_area: int = 30
) -> list[tuple[int, int, int, int]]:
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
        if (
            xs.min() < margin
            or xs.max() > w - margin
            or ys.min() < margin
            or ys.max() > h - margin
        ):
            return True
    return False


def covered_by_other(
    folder: Path, frame_index: int, gone: np.ndarray, own_file: Path
) -> float:
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


import re, sys
from ....app.remote.gpu.worker import cached, sha


def _prompt_tokens(text):
    """Lowercase word tokens of a prompt text: every character that is not a-z, 0-9, apostrophe or
    hyphen becomes a word separator, so punctuation and case never split a phrase."""
    return [t for t in re.sub(r"[^a-z0-9'-]+", " ", text.lower()).split()]


def fallback_phrase(description, concept):
    """A 2-4 word phrase from the subject's `description` that contains `concept`, or None.

    When the bare concept word (the spec's segment_word) segments nothing, SAM's text grounding
    still finds the object under the fuller wording it was described with (measured on R049/T10:
    'celery' 0/124 frames, 'green celery stalk' 86/124). The rule is purely positional, no keyword
    list: normalise both texts with _prompt_tokens, find where the concept's token sequence occurs
    in the description's tokens (the spec validation guarantees the segment_word is a word of its
    description, so the phrase is the concept's own language by construction), then take that
    occurrence plus its immediate left and right neighbours. With a 1-2 word concept that is a
    2-4 word phrase, hard-capped at 4. None when the concept is absent from the description or the
    description adds no word around it."""
    tokens = _prompt_tokens(description)
    word = _prompt_tokens(concept)
    if not word or len(word) > len(tokens):
        return None
    at = next(
        (
            i
            for i in range(len(tokens) - len(word) + 1)
            if tokens[i : i + len(word)] == word
        ),
        None,
    )
    if at is None or len(tokens) == len(word):
        return None
    phrase = tokens[max(0, at - 1) : min(len(tokens), at + len(word) + 1)]
    return " ".join(phrase[-4:] if len(phrase) > 4 else phrase)


def frames_with_mask(rows):
    """How many of the propagated frames carry any mask at all."""
    return sum(1 for r in rows if max(r["mask_areas_pixels"], default=0) > 0)


def needs_prompt_fallback(rows):
    """True when the propagate loop saw frames but never produced one mask: the prompt found
    nothing anywhere, which is a prompt failure, not evidence that the object is absent."""
    return bool(rows) and frames_with_mask(rows) == 0


def sam31(a, modelpath, source, torch):
    if not a.concept:
        raise ValueError("SAM requires a predeclared task concept")
    import numpy as np

    def build():
        sys.path.insert(0, str(source))
        from sam3.model_builder import build_sam3_predictor

        # Upstream supports SDPA for pre-Hopper hardware; A800 cannot use FA3.
        predictor = build_sam3_predictor(
            checkpoint_path=str(modelpath / "sam3.1_multiplex.pt"),
            version="sam3.1",
            compile=False,
            warm_up=False,
            use_fa3=False,
            async_loading_frames=False,
        )
        assert all(p.device.type == "cuda" for p in predictor.model.parameters())
        checkpoint = torch.load(
            modelpath / "sam3.1_multiplex.pt", map_location="cpu", weights_only=True
        )
        checkpoint = checkpoint.get("model", checkpoint)
        missing_parameters = set(dict(predictor.model.named_parameters())) - set(
            checkpoint
        )
        if missing_parameters:
            raise ValueError(
                "Unloaded SAM parameters: " + str(sorted(missing_parameters)[:10])
            )
        del checkpoint
        # Upstream base dispatcher passes a SAM3-only keyword to SAM3.1.
        # Ignore only the unsupported False value; never permit state offload.
        native_init = predictor.model.init_state

        def init_cuda_state(*args, offload_state_to_cpu=False, **kwargs):
            if offload_state_to_cpu:
                raise ValueError("CPU state offload forbidden")
            return native_init(*args, **kwargs)

        predictor.model.init_state = init_cuda_state
        return predictor

    predictor = cached("sam31", build)

    def segment(text):
        """One session over the whole video with `text` as the prompt; the exact loop the one-shot
        run used, so a retried prompt is measured the same way as the first one."""
        session = None
        rows = []
        masks = []
        indices = []
        ids = []
        try:
            with torch.inference_mode():
                session = predictor.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": str(a.video),
                        "offload_video_to_cpu": False,
                        "offload_state_to_cpu": False,
                    }
                )["session_id"]
                predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session,
                        "frame_index": 0,
                        "text": text,
                    }
                )
                for response in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                    }
                ):
                    i = int(response["frame_index"])
                    out = response["outputs"]
                    mm = np.asarray(out["out_binary_masks"])
                    oid = np.asarray(out["out_obj_ids"])
                    rows.append(
                        {
                            "frame": i,
                            "instance_ids": oid.tolist(),
                            "mask_areas_pixels": [int(x.sum()) for x in mm],
                        }
                    )
                    if i % 8 == 0:
                        masks.append(mm)
                        indices.append(i)
                        ids.append(oid)
                if rows and indices[-1] != rows[-1]["frame"]:
                    masks.append(mm)
                    indices.append(i)
                    ids.append(oid)
        finally:
            if session is not None:
                predictor.handle_request(
                    {"type": "close_session", "session_id": session}
                )
        if not rows:
            raise ValueError("SAM returned no frames")
        return rows, masks, indices, ids

    rows, masks, indices, ids = segment(a.concept)
    fallback = {}
    phrase = (
        fallback_phrase(getattr(a, "description", "") or "", a.concept)
        if needs_prompt_fallback(rows)
        else None
    )
    if phrase:
        # The concept word found nothing in any frame: rerun the same video with the description's
        # phrase. If that also finds nothing, the all-zero masks and the frame rows stay exactly as
        # the first pass produced them, so the empty-mask note downstream still says the truth.
        f_rows, f_masks, f_indices, f_ids = segment(phrase)
        fallback = {
            a.concept: {
                "first_prompt": a.concept,
                "fallback_prompt": phrase,
                "frames_with_mask_after": frames_with_mask(f_rows),
            }
        }
        if fallback[a.concept]["frames_with_mask_after"] > 0:
            rows, masks, indices, ids = f_rows, f_masks, f_indices, f_ids
    artifact = a.output.with_suffix(".masks.npz")
    payload = {}
    for i, m, oid in zip(indices, masks, ids):
        payload[f"masks_{i}"] = m
        payload[f"ids_{i}"] = oid
    np.savez_compressed(artifact, **payload)
    result = {
        "artifact": str(artifact),
        "artifact_sha256": sha(artifact),
        "concept": a.concept,
        "frames_processed": len(rows),
        "frame_observations": rows,
        "recipe": {
            "version": "sam3.1",
            "use_fa3": False,
            "compile": False,
            "cpu_offload": False,
        },
        "limitation": "concept-conditioned segmentation/tracking; mask instances are not semantic object count or target correctness",
    }
    if fallback:
        result["prompt_fallback_used"] = fallback
    return result
