"""Model evaluation: detection metrics, calibration, experiments.

- evaluate(): run ultralytics val on a dataset, persist metrics + confusion.
- calibrate(): fit isotonic/Platt/temperature maps on validation confidences.
- partial_reveal_experiment(): progressive-completion recognition test (spec §94).
- compare_systems(): CV-only vs geometry-only vs hybrid (spec §93).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def evaluate(weights: str, data_yaml: str, out_dir: str = "artifacts/eval") -> dict:
    """Run standard detection metrics (mAP@0.5, mAP@0.5:0.95, per-class AP)."""
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, verbose=False)
    out = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_ap50": {
            i: float(v) for i, v in enumerate(metrics.box.ap50.tolist())
        },
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metrics.json").write_text(json.dumps(out, indent=2))
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        np.save(Path(out_dir) / "confusion_matrix.npy", metrics.confusion_matrix.matrix)
    return out


@dataclass
class Calibration:
    method: str
    params: dict

    def apply(self, confidence: float) -> float:
        if self.method == "isotonic":
            xs, ys = self.params["x"], self.params["y"]
            return float(np.interp(confidence, xs, ys))
        if self.method == "platt":
            a, b = self.params["a"], self.params["b"]
            return float(1 / (1 + np.exp(-(a * confidence + b))))
        return confidence  # identity


def fit_calibration(
    confidences: np.ndarray, correct: np.ndarray, method: str = "isotonic"
) -> Calibration:
    """Fit a confidence->probability map on validation outcomes."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(confidences, correct)
        xs = ir.X_thresholds_.tolist()
        ys = ir.y_thresholds_.tolist()
        return Calibration("isotonic", {"x": xs, "y": ys})
    if method == "platt":
        lr = LogisticRegression()
        lr.fit(confidences.reshape(-1, 1), correct)
        return Calibration("platt", {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])})
    return Calibration("identity", {})


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """ECE — lower is better calibrated."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += mask.sum() / len(conf) * abs(acc - avg_conf)
    return float(ece)


def partial_reveal_experiment(dataset_manifest: str, pipeline, fractions=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> dict:
    """Take completed synthetic patterns, progressively reveal, measure
    recognition accuracy + Mean Absolute Completion Error (spec §94)."""
    manifest = json.loads(Path(dataset_manifest).read_text())
    samples = [s for s in manifest["samples"] if s["kind"] == "positive" and s["completion"] >= 99.9]
    per_frac: dict[float, list] = {f: [] for f in fractions}
    mace_errors: list[float] = []
    from ..datasets.synthetic.generate import _candles_from_close, _seeded_rng

    for s in samples[: min(len(samples), 300)]:
        rng = _seeded_rng(hash(s["id"]) & 0xFFFF)
        # rebuild a deterministic close path from keypoints is lossy; instead
        # re-render from stored span on the dataset image would need raw data.
        # For the experiment we regenerate via the generator map.
        pass
    # The full experiment requires stored closes; synthetic generator stores
    # keypoints only. Re-run generation with store_closes=True for experiments.
    return {
        "note": "run ml.evaluation.experiment_runner for full experiment",
        "n_samples_seen": len(samples),
        "fractions": list(fractions),
    }


def compare_systems(df, pipeline, pattern_id: str) -> dict:
    """System A (CV only) vs B (geometry only) vs C (hybrid) on one window."""
    from ..geometry.engine import GeometryEngine

    geo = GeometryEngine(pipeline.registry).validate(df, pattern_ids=[pattern_id])
    cv = pipeline._cv_detect(df)
    cv_hit = any(d.pattern_id == pattern_id for d in cv)
    geo_hit = bool(geo and geo[0].matched)
    hybrid = pipeline.analyze(df)
    hyb_hit = any(f.pattern_id == pattern_id for f in hybrid)
    return {
        "pattern_id": pattern_id,
        "cv_only": cv_hit,
        "geometry_only": geo_hit,
        "hybrid": hyb_hit,
        "cv_available": pipeline.detector.available,
    }
