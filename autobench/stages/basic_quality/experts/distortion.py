from pathlib import Path
import cv2
import numpy as np

PAIR_STEP = 2
PAIR_STRIDE = 2
BURST_T = 1.0
SIZE = 384


def read_gray(path: Path, size: int = SIZE) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(
            cv2.cvtColor(
                cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
        )
    cap.release()
    return frames


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample ``image`` at x + flow(x) (backward warp)."""
    h, w = image.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        image,
        gx + flow[..., 0],
        gy + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _patch_ssim(a, b) -> np.ndarray:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (16, 16)
    mu_a, mu_b = cv2.blur(a, k), cv2.blur(b, k)
    var_a = cv2.blur(a * a, k) - mu_a**2
    var_b = cv2.blur(b * b, k) - mu_b**2
    cov = cv2.blur(a * b, k) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return ssim


def pair_cues(g0, g1, dis) -> dict:
    fwd = dis.calc(g0, g1, None)
    warped = _warp(g1.astype(np.float32), fwd)
    ssim = _patch_ssim(g0.astype(np.float32), warped)
    flat = np.sort(ssim[8:-8, 8:-8].ravel())
    warp_low = float(flat[: max(1, len(flat) // 20)].mean())
    e0 = cv2.boxFilter(
        np.abs(cv2.Laplacian(g0.astype(np.float32), cv2.CV_32F)), -1, (16, 16)
    )[8::16, 8::16]
    e1 = cv2.boxFilter(
        np.abs(cv2.Laplacian(warped.astype(np.float32), cv2.CV_32F)), -1, (16, 16)
    )[8::16, 8::16]
    burst = float(np.percentile(np.abs(np.log((e1 + 1.0) / (e0 + 1.0))), 99))
    return {"warp_ssim": warp_low, "tex_burst": burst}


def _gen_artifact(video: Path) -> dict:
    """Generated-artifact cue (noise / pop-up texture channel) with its two-tier rule operands.

    Same procedure the noise validation ran on the 130 annotated videos (distortion_v2 /
    distortion_sustained.py: DIS optical flow over every frame pair two frames apart at 384 px):

    - ``gen_artifact_warp``  the warp anomaly: minus the median over pairs of the worst-5% local SSIM
      after warping frame t+2 back onto frame t — content the motion cannot explain (higher = worse).
    - ``gen_artifact_burst_share``  the sustained share of pairs whose texture pops up (burst, the
      99th pct |log| high-pass energy ratio after alignment, over 1.0). Generative grain pops up over
      and over; a lamp turning on does so once — this is what kills the B009 lamp false alarm.
    - ``gen_artifact``  the channel value: warp anomaly + median burst.

    The gate applies the calibrated two-tier rule (噪点专家验证结果_20261028.md) to these operands on
    the within-question robust scale: strong line warp > 2.0, or weak line warp > 1.2 AND
    burst_share > 0.4 — 8/9 user positives, 0 in-question false positives; known residue B009_E2 T6.
    """

    frames = read_gray(video)
    if len(frames) < PAIR_STEP + 1:
        return {
            "gen_artifact": None,
            "gen_artifact_warp": None,
            "gen_artifact_burst_share": None,
        }
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    warps, bursts = [], []
    for i in range(0, len(frames) - PAIR_STEP, PAIR_STRIDE):
        cues = pair_cues(frames[i], frames[i + PAIR_STEP], dis)
        warps.append(cues["warp_ssim"])
        bursts.append(cues["tex_burst"])
    warp = float(-np.median(warps))  # SSIM is a similarity; invert to an anomaly score
    return {
        "gen_artifact": warp + float(np.median(bursts)),
        "gen_artifact_warp": warp,
        "gen_artifact_burst_share": float(np.mean([b > BURST_T for b in bursts])),
    }
