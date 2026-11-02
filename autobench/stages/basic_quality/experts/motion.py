"""分离二维整体变换与主体内部几何变化。"""

import numpy as np


def fit_similarity(reference, current):
    """拟合 reference 到 current 的平移、等比缩放与旋转。"""
    x = np.asarray(reference, dtype=float)
    y = np.asarray(current, dtype=float)
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    variance = np.sum(xc**2)
    if variance < 1e-8:
        raise ValueError("Degenerate reference landmarks")
    u, s, vt = np.linalg.svd(xc.T @ yc)
    signs = np.ones(2)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(s * signs) / variance)
    if scale <= 1e-8:
        raise ValueError("Degenerate fitted scale")
    prediction = xc @ rotation * scale + y.mean(0)
    residual = (y - prediction) / scale
    return dict(
        scale=scale,
        residual_rms_px=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        reference_radius_px=float(np.sqrt(np.mean(np.sum(xc**2, axis=1)))),
        centroid_displacement_px=float(np.linalg.norm(y.mean(0) - x.mean(0))),
    )


def summarize(tracks, visibility):
    tracks = np.asarray(tracks, dtype=float)
    visibility = np.asarray(visibility, dtype=bool)
    rows = []
    for frame, current in enumerate(tracks):
        good = visibility[0] & visibility[frame]
        if good.sum() < 4:
            continue
        result = fit_similarity(tracks[0, good], current[good])
        result.update(
            frame=frame,
            visible_points=int(good.sum()),
            deformation_percent=100
            * result["residual_rms_px"]
            / result["reference_radius_px"],
        )
        rows.append(result)
    if len(rows) < len(tracks) * 0.7:
        raise ValueError("Insufficient visible landmark coverage")
    changes = [r for r in rows if r["frame"] > 0]
    return dict(
        shape_change_rms_px=float(
            np.sqrt(np.mean([r["residual_rms_px"] ** 2 for r in changes]))
        ),
        shape_change_percent=float(
            np.sqrt(np.mean([r["deformation_percent"] ** 2 for r in changes]))
        ),
        valid_frame_fraction=len(rows) / len(tracks),
        endpoint_scale=rows[-1]["scale"],
        endpoint_frame=rows[-1]["frame"],
        observations=rows,
    )
