"""Evaluation CLI.

Usage:
    python -m ml.evaluation.evaluate --weights artifacts/runs/<run>/best.pt
    python -m ml.evaluation.evaluate --calibrate validation_predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate, expected_calibration_error, fit_calibration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="trained weights (.pt)")
    ap.add_argument("--data", default=None, help="dataset data.yaml")
    ap.add_argument("--calibrate", default=None, help="JSON with {confidences:[], correct:[]}")
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "platt", "identity"])
    args = ap.parse_args()

    if args.weights:
        data = args.data or _latest_data_yaml()
        if not data:
            raise SystemExit("no dataset found — generate one or pass --data")
        out = evaluate(args.weights, data)
        print(json.dumps(out, indent=2))

    if args.calibrate:
        payload = json.loads(Path(args.calibrate).read_text())
        conf = np.array(payload["confidences"], dtype=float)
        correct = np.array(payload["correct"], dtype=float)
        cal = fit_calibration(conf, correct, args.method)
        ece_raw = expected_calibration_error(conf, correct)
        mapped = np.array([cal.apply(c) for c in conf])
        ece_cal = expected_calibration_error(np.clip(mapped, 0.01, 0.999), correct)
        print(json.dumps({
            "method": cal.method,
            "ece_before": round(ece_raw, 4),
            "ece_after": round(ece_cal, 4),
            "params": cal.params if cal.method != "isotonic" else "see artifacts/eval/calibration.json",
        }, indent=2))
        Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
        (Path("artifacts/eval") / "calibration.json").write_text(
            json.dumps({"method": cal.method, "params": cal.params,
                        "ece_before": ece_raw, "ece_after": ece_cal}, indent=2)
        )


def _latest_data_yaml() -> str | None:
    root = Path("artifacts/datasets")
    if not root.exists():
        return None
    cands = sorted((p for p in root.iterdir() if (p / "data.yaml").exists()), key=lambda p: p.stat().st_mtime)
    return str(cands[-1] / "data.yaml") if cands else None


if __name__ == "__main__":
    main()
