"""Partial-detection experiment (spec §94).

Takes completed synthetic patterns, progressively reveals 30%→100% of the
formation, and measures:
  1. whether the system recognises the correct pattern,
  2. Mean Absolute Completion Error (MACE) between true and estimated %,
  3. false early-detection rate (wrong pattern matched at ≥0.5 geometry).

Usage:
    python -m ml.evaluation.experiment_runner [--samples 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def run_experiment(samples: int = 100, seed: int = 11) -> dict:
    from ..completion.estimator import estimate_completion
    from ..datasets.synthetic.generate import (
        DatasetSpec,
        _candles_from_close,
        _seeded_rng,
        _truncate_for_completion,
        build_generator_map,
    )
    from ..geometry.engine import GeometryEngine
    from ..registry.registry import get_registry

    registry = get_registry()
    engine = GeometryEngine(registry)
    gen_map = build_generator_map(registry)
    pattern_ids = sorted(gen_map.keys())

    rng = _seeded_rng(seed)
    per_frac: dict[float, dict[str, list]] = {f: {"correct": [], "mace": [], "wrong": []} for f in FRACTIONS}

    for i in range(samples):
        pid = pattern_ids[i % len(pattern_ids)]
        gen, base_v = gen_map[pid]
        v = {**base_v, "volatility": 0.006, "base_price": 100.0}
        try:
            sp = gen(v, rng)
        except Exception:
            continue
        defn = registry.get(pid)
        if not defn.supports_partial_detection:
            continue
        for f in FRACTIONS:
            closes, true_pct = _truncate_for_completion(sp, f)
            df = _candles_from_close(closes, rng, 0.006)
            results = engine.validate(df, pattern_ids=[pid])
            est_pct = 0.0
            correct = False
            if results:
                r = results[0]
                est = estimate_completion(defn, r)
                est_pct = est.completion
                correct = r.matched
            wrong_hits = [
                r.pattern_id
                for r in engine.validate(df)[:5]
                if r.matched and r.pattern_id != pid
            ][:1]
            per_frac[f]["correct"].append(correct)
            per_frac[f]["mace"].append(abs(est_pct - true_pct))
            per_frac[f]["wrong"].append(bool(wrong_hits))

    out = {"n_samples": samples, "fractions": {}}
    maes = []
    for f in FRACTIONS:
        d = per_frac[f]
        if not d["correct"]:
            continue
        acc = float(np.mean(d["correct"]))
        mace = float(np.mean(d["mace"]))
        wrong = float(np.mean(d["wrong"]))
        maes.append(mace)
        out["fractions"][f"{f:.0%}"] = {
            "recognition_accuracy": round(acc, 3),
            "mace_pct": round(mace, 1),
            "false_early_detection_rate": round(wrong, 3),
            "n": len(d["correct"]),
        }
    out["overall_mace_pct"] = round(float(np.mean(maes)), 1) if maes else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()
    out = run_experiment(args.samples)
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "partial_reveal.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
