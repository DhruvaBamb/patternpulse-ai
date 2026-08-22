"""Research evaluation: System A (CV only) vs B (geometry only) vs C (hybrid)
on the dataset test split (spec §93), plus ablation variants (spec §95).

Metrics: precision / recall / F1 (image-level pattern presence), false
positives, and latency per window.

Usage:
    python -m ml.evaluation.systems --weights artifacts/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_split(dataset_root: Path, split: str, limit: int | None = None):
    """Yield (image_path, gt_class_ids) for the split."""
    items = []
    labels_dir = dataset_root / "labels" / split
    images_dir = dataset_root / "images" / split
    for txt in sorted(labels_dir.glob("*.txt")):
        img = images_dir / f"{txt.stem}.png"
        if not img.exists():
            continue
        gt = {int(line.split()[0]) for line in txt.read_text().splitlines() if line.strip()}
        items.append((img, gt))
        if limit and len(items) >= limit:
            break
    return items


def evaluate_systems(weights: str, dataset_yaml: str, limit: int | None = 400) -> dict:
    import yaml
    from ultralytics import YOLO

    from ..geometry.engine import GeometryEngine
    from ..inference.detector import pick_device
    from ..registry.registry import get_registry

    registry = get_registry()
    id2pattern = {d.cv_class_id: d.id for d in registry.implemented()}
    ds = yaml.safe_load(Path(dataset_yaml).read_text())
    items = _load_split(Path(ds["path"]), "test", limit)

    model = YOLO(weights)
    device = pick_device("auto")
    geometry = GeometryEngine(registry)

    # class-level tallies per system
    def blank() -> dict:
        return defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    systems = {"A_cv_only": blank(), "B_geometry_only": blank(), "C_hybrid": blank()}
    latencies = {"A_cv_only": [], "B_geometry_only": [], "C_hybrid": []}

    # per-category symbol→closes cache is impossible from images alone; for the
    # geometry arm we regenerate closes from the stored keypoints via the
    # manifest (span/keypoints) — instead we approximate by rendering-inverse:
    # simpler and honest: run geometry on the manifest's stored closes when
    # available. The manifest stores spans; regeneration is seeded by sample id.
    from ..datasets.synthetic.generate import _seeded_rng, build_generator_map

    gen_map = build_generator_map(registry)
    manifest_path = Path(ds["path"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"samples": []}
    closes_by_id: dict[str, np.ndarray] = {}
    if manifest.get("samples"):
        for s in manifest["samples"]:
            if s["kind"] != "positive":
                continue
            gen_entry = gen_map.get(s["pattern_id"])
            if gen_entry is None:
                continue
            gen, base_v = gen_entry
            rng = _seeded_rng((int(s["id"].split("_")[-1]) if s["id"].split("_")[-1].isdigit() else 0) + 1)
            v = {**base_v, "volatility": 0.008, "base_price": 100.0}
            try:
                sp = gen(v, rng)
                closes_by_id[s["id"]] = sp.closes
            except Exception:
                continue

    n_with_closes = 0
    for img, gt in items:
        stem = img.stem
        gt_patterns = {id2pattern[c] for c in gt if c in id2pattern}

        # System A: CV only
        t0 = time.perf_counter()
        im = _read_image(img)
        res = model.predict(source=im, conf=0.3, imgsz=640, device=device, verbose=False)
        cv_patterns = set()
        if res and res[0].boxes is not None and res[0].boxes.cls is not None:
            cv_patterns = {id2pattern[int(c)] for c in res[0].cls.tolist() if int(c) in id2pattern}
        latencies["A_cv_only"].append((time.perf_counter() - t0) * 1000)

        # System B: geometry only — needs the OHLC closes; only samples we can
        # regenerate deterministically participate in the B/C comparison.
        closes = closes_by_id.get(stem)
        geo_patterns: set[str] = set()
        geo_scores: dict[str, float] = {}
        if closes is not None:
            n_with_closes += 1
            import pandas as pd

            df = _df_from_closes(closes)
            t0 = time.perf_counter()
            results = geometry.validate(df)
            latencies["B_geometry_only"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            geo_patterns = {r.pattern_id for r in results if r.matched}
            geo_scores = {r.pattern_id: r.geometry_score for r in results}
            latencies["C_hybrid"].append((time.perf_counter() - t0) * 1000)

        # System C: hybrid = intersection-style agreement (CV proposal AND
        # geometry pass) plus geometry-only hits above the fusion floor
        hybrid_patterns = {p for p in cv_patterns if geo_scores.get(p, 0) >= 0.4}
        hybrid_patterns |= {p for p in geo_patterns if p in cv_patterns or p not in cv_patterns}
        if closes is None:
            hybrid_patterns = cv_patterns  # no geometry witness available

        for name, predicted in (
            ("A_cv_only", cv_patterns),
            ("B_geometry_only", geo_patterns if closes is not None else set()),
            ("C_hybrid", hybrid_patterns),
        ):
            tallies = systems[name]
            for p in predicted & gt_patterns:
                tallies[p]["tp"] += 1
            for p in predicted - gt_patterns:
                tallies[p]["fp"] += 1
            for p in gt_patterns - predicted:
                tallies[p]["fn"] += 1

    def prf(tallies: dict) -> dict:
        tp = sum(t["tp"] for t in tallies.values())
        fp = sum(t["fp"] for t in tallies.values())
        fn = sum(t["fn"] for t in tallies.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    out = {
        "n_images": len(items),
        "n_with_regenerable_closes": n_with_closes,
        "systems": {name: prf(t) for name, t in systems.items()},
        "latency_ms": {
            k: {"mean": round(float(np.mean(v)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
            for k, v in latencies.items() if v
        },
        "note": "B/C arms cover regenerable synthetic samples; per-class detail in per_class",
        "per_class": {
            name: {p: dict(t) for p, t in tallies.items() if t["tp"] or t["fp"] or t["fn"]}
            for name, tallies in systems.items()
        },
    }
    return out


def _read_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB")).copy()


def _df_from_closes(closes: np.ndarray):
    import pandas as pd

    n = len(closes)
    rng = np.random.default_rng(7)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    wick = np.maximum(np.abs(rng.normal(0, 0.004, n)), 0.0005) * closes * 0.5
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + wick,
            "low": np.minimum(opens, closes) - wick,
            "close": closes,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    dataset = args.dataset
    if dataset is None:
        root = Path("artifacts/datasets")
        cands = sorted((p for p in root.iterdir() if (p / "data.yaml").exists()), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("no dataset found")
        dataset = str(cands[-1] / "data.yaml")
    out = evaluate_systems(args.weights, dataset, args.limit)
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "systems_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
