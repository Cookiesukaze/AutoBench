"""CoTracker3 轨迹：主体上的点怎么走。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ....app.data.types import ContractError


def load(folder: Path) -> np.lib.npyio.NpzFile:
    path = folder / "cotracker3.tracks.npz"
    if not path.exists():
        raise ContractError(f"缺少 {path}")
    return np.load(path)


import json, sys
from ....app.remote.gpu.worker import cached, decode, sha


def track(a, modelpath, source, torch):
    import numpy as np

    def build():
        sys.path.insert(0, str(source))
        from cotracker.predictor import CoTrackerPredictor

        return (
            CoTrackerPredictor(
                checkpoint=str(modelpath / "scaled_offline.pth"),
                window_len=60,
                v2=False,
            )
            .cuda()
            .eval()
        )

    model = cached("cotracker3", build)
    frames, fps = decode(a.video)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()
    kwargs = {"grid_size": 16, "backward_tracking": True}
    labels = []
    if a.targets:
        target = json.loads(a.targets.read_text(encoding="utf-8"))
        points = target["points"]
        if not points:
            raise ValueError("没有可跟踪的声明点；不能悄悄改用网格点")
        h, w = frames.shape[1:3]
        q = [[p.get("frame", 0), p["x"] * (w - 1), p["y"] * (h - 1)] for p in points]
        kwargs = {
            "queries": torch.tensor([q], device="cuda", dtype=torch.float32),
            "backward_tracking": True,
        }
        labels = [p["label"] for p in points]
    with torch.inference_mode():
        tracks, visible = model(video, **kwargs)
    tr = tracks[0].cpu().numpy()
    vis = visible[0].cpu().numpy()
    artifact = a.output.with_suffix(".tracks.npz")
    np.savez_compressed(
        artifact, tracks=tr, visibility=vis, fps=fps, labels=np.array(labels)
    )
    h, w = frames.shape[1:3]
    selected = np.linspace(0, len(frames) - 1, min(16, len(frames))).round().astype(int)
    observations = []
    for i, label in enumerate(labels):
        observations.append(
            {
                "label": label,
                "coordinate_system": "normalized source image",
                "samples": [
                    {
                        "frame": int(t),
                        "seconds": float(t / fps),
                        "x": float(tr[t, i, 0] / (w - 1)),
                        "y": float(tr[t, i, 1] / (h - 1)),
                        "visible": bool(vis[t, i]),
                    }
                    for t in selected
                ],
                "visible_fraction": float(vis[:, i].mean()),
            }
        )
    return {
        "artifact": str(artifact),
        "artifact_sha256": sha(artifact),
        "point_observations": observations,
        "points": int(tr.shape[1]),
        "frames": len(frames),
        "fps": fps,
        "limitation": "point correspondence, not semantic identity or object count",
        "configuration": {
            "window_len": 60,
            **{k: v for k, v in kwargs.items() if k != "queries"},
        },
    }
