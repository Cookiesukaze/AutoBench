"""WAFT 光流：背景的运动场，用来分离镜头与主体。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.data.types import ContractError


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "waft.flow.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path)


def fit(
    points: np.ndarray, moved: np.ndarray, ransac_iters: int = 0
) -> tuple[float, np.ndarray]:
    """相似变换 moved = s·R·points + t，在中心化坐标上解。

    ransac_iters>0 时先跑一小轮 RANSAC，把少数不一致的点丢掉：背景掩码里混进来的前景运动
    会把镜头估计带偏。
    """
    pc, mc = points - points.mean(0), moved - moved.mean(0)

    def solve(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, np.ndarray]:
        a = np.sum(src * dst) / np.sum(src**2)
        b = np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0]) / np.sum(src**2)
        return a, b, dst.mean(0) - src.mean(0)

    if ransac_iters and len(points) >= 12:
        rng = np.random.default_rng(0)
        best_inliers = None
        for _ in range(ransac_iters):
            idx = rng.choice(len(pc), size=max(8, len(pc) // 2), replace=False)
            a, b, t = solve(pc[idx], mc[idx])
            rot = np.array([[a, -b], [b, a]])
            resid = np.linalg.norm(pc @ rot.T + t - mc, axis=1)
            inliers = resid < 3.0
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
        if best_inliers is not None and best_inliers.sum() >= 8:
            pc, mc = pc[best_inliers], mc[best_inliers]
    a, b, t = solve(pc, mc)
    return float(np.hypot(a, b)), t


import json, os, sys
from ....app.remote.gpu.worker import cached, decode, sha


def waft(a, modelpath, source, torch):
    import numpy as np
    from types import SimpleNamespace

    def build():
        sys.path.insert(0, str(source))
        os.chdir(source)
        from model import fetch_model
        from inference_tools import InferenceWrapper

        config = json.loads((source / "config/a1/tar-c-t.json").read_text())
        model = fetch_model(SimpleNamespace(**config))
        checkpoint = modelpath / "waft_tar-c-t.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        loaded = model.load_state_dict(state, strict=False)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError("WAFT checkpoint mismatch: " + str(loaded))
        model = model.cuda().eval()
        return config, InferenceWrapper(
            model,
            scale=config["scale"],
            train_size=config["image_size"],
            pad_to_train_size=False,
            tiling=False,
        )

    config, wrapped = cached("waft", build)
    frames, fps = decode(a.video)
    # Adjacent full-resolution pairs at fixed intervals; no downscaled substitute.
    indices = sorted(
        set(
            np.linspace(0, len(frames) - 2, min(8, len(frames) - 1))
            .round()
            .astype(int)
            .tolist()
        )
    )
    flows = []
    values = []
    with torch.inference_mode():
        for i in indices:
            inputs = [
                torch.from_numpy(frames[t]).permute(2, 0, 1)[None].float().cuda()
                for t in (i, i + 1)
            ]
            flow = wrapped.calc_flow(*inputs)["flow"][-1][0]
            if not torch.isfinite(flow).all():
                raise ValueError("Nonfinite flow")
            magnitude = flow.square().sum(0).sqrt()
            values.append(
                {
                    "from_frame": i,
                    "to_frame": i + 1,
                    "seconds": i / fps,
                    "mean_displacement_pixels": magnitude.mean().item(),
                    "p95_displacement_pixels": torch.quantile(magnitude, 0.95).item(),
                }
            )
            flows.append(flow.cpu().numpy())
    artifact = a.output.with_suffix(".flow.npz")
    np.savez_compressed(artifact, flow=np.stack(flows), indices=indices, fps=fps)
    return {
        "pairs": values,
        "artifact": str(artifact),
        "artifact_sha256": sha(artifact),
        "config": config,
        "limitation": "sampled optical flow, not physics correctness or monotonic quality score",
    }
