"""DINOv3 全帧特征与视频内相似度；不冒充主体或参考图身份判定。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.data.types import ContractError


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "dinov3.features.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path, allow_pickle=True)


from ....app.remote.gpu.worker import cached, decode, sha


def dinov3(a, modelpath, torch):
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    def build():
        processor = AutoImageProcessor.from_pretrained(
            str(modelpath), local_files_only=True
        )
        model = (
            AutoModel.from_pretrained(str(modelpath), local_files_only=True)
            .cuda()
            .eval()
        )
        assert all(p.device.type == "cuda" for p in model.parameters())
        return processor, model

    processor, model = cached("dinov3", build)
    frames, fps = decode(a.video)
    indices = np.linspace(0, len(frames) - 1, 8).round().astype(int)
    vectors = []
    shapes = []
    with torch.inference_mode():
        for i in indices:
            inputs = processor(
                images=Image.fromarray(frames[i]), return_tensors="pt"
            ).to("cuda")
            out = model(**inputs)
            if not torch.isfinite(out.last_hidden_state).all():
                raise ValueError("Nonfinite DINO features")
            vectors.append(out.pooler_output.float()[0])
            shapes.append(list(out.last_hidden_state.shape))
        v = torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
        similarity = (v @ v.T).cpu().numpy()
    artifact = a.output.with_suffix(".features.npz")
    np.savez_compressed(
        artifact,
        normalized_cls=v.cpu().numpy(),
        similarity=similarity,
        frame_indices=indices,
    )
    return {
        "artifact": str(artifact),
        "artifact_sha256": sha(artifact),
        "frames": indices.tolist(),
        "hidden_shapes": shapes,
        "cosine_to_first": similarity[0].tolist(),
        "recipe": "official AutoImageProcessor defaults, FP32 CUDA; eight independent frames",
        "limitation": "global frame features; no localized target identity, count or physics verdict",
    }
