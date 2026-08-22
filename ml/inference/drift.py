"""Input-distribution drift monitoring (spec §75).

Captures reference statistics from the training dataset manifest + images
(volatility proxies via per-image candle-density/contrast, resolution) and
compares live windows against them, emitting warnings on material shift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ReferenceStats:
    image_mean: float
    image_std: float
    edge_density_mean: float
    resolution: tuple[int, int]
    n_images: int
    meta: dict = field(default_factory=dict)


def build_reference(dataset_root: str | Path, max_images: int = 400) -> ReferenceStats:
    """Compute reference statistics from a dataset's images."""
    import cv2

    root = Path(dataset_root)
    images = sorted((root / "images" / "train").glob("*.png"))[:max_images]
    if not images:
        raise ValueError(f"no training images under {root}")
    means, stds, edges = [], [], []
    res = None
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        res = img.shape[:2]
        means.append(float(img.mean()))
        stds.append(float(img.std()))
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        edges.append(float((mag > 60).mean()))
    return ReferenceStats(
        image_mean=float(np.mean(means)),
        image_std=float(np.mean(stds)),
        edge_density_mean=float(np.mean(edges)),
        resolution=(int(res[0]), int(res[1])) if res else (0, 0),
        n_images=len(images),
    )


def load_or_build_reference(dataset_root: str | Path, cache: str | Path = "artifacts/eval/drift_reference.json") -> ReferenceStats:
    cache = Path(cache)
    if cache.exists():
        d = json.loads(cache.read_text())
        return ReferenceStats(**d)
    ref = build_reference(dataset_root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ref.__dict__))
    return ref


def monitor_window(image: np.ndarray, ref: ReferenceStats) -> dict:
    """Compare one live-rendered window (HxWx3 uint8) against the reference.

    Returns shift metrics + warning list. Warnings use conservative
    thresholds; they flag investigation, not failure.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edge_density = float((np.sqrt(gx**2 + gy**2) > 60).mean())
    warnings: list[str] = []
    if abs(gray.mean() - ref.image_mean) > max(12.0, 0.35 * ref.image_std):
        warnings.append(f"brightness shifted (live {gray.mean():.0f} vs train {ref.image_mean:.0f}) — check rendering config/theme")
    if image.shape[:2] != ref.resolution:
        warnings.append(f"resolution differs (live {image.shape[:2]} vs train {ref.resolution})")
    if abs(edge_density - ref.edge_density_mean) > max(0.02, 0.5 * ref.edge_density_mean):
        warnings.append(
            f"chart density shifted (live {edge_density:.3f} vs train {ref.edge_density_mean:.3f}) — "
            "different volatility regime or candle width"
        )
    return {
        "live_mean": round(float(gray.mean()), 1),
        "reference_mean": round(ref.image_mean, 1),
        "live_edge_density": round(edge_density, 4),
        "reference_edge_density": round(ref.edge_density_mean, 4),
        "warnings": warnings,
        "drifted": bool(warnings),
    }


def volatility_regime(df: pd.DataFrame, window: int = 20) -> str:
    """Classify current realised-volatility regime for drift context."""
    ret = df["close"].pct_change().tail(window)
    vol = float(ret.std())
    if vol < 0.008:
        return "low"
    if vol < 0.02:
        return "normal"
    if vol < 0.04:
        return "elevated"
    return "high"
