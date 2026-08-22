"""Hard-example mining (spec §42).

Runs the trained detector over a dataset split, matches predictions against
ground-truth labels, and exports confusing samples (false positives, false
negatives, low-confidence correct detections, and known confusion pairs
like Flag↔Pennant, Hammer↔Hanging Man) into a review directory for
retraining emphasis.

Usage:
    python -m ml.evaluation.mining --weights artifacts/runs/<run>/best.pt \
        [--dataset artifacts/datasets/<v>/data.yaml] [--split valid] [--max 200]
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Confusing pairs from the spec (registry ids).
CONFUSION_PAIRS: list[tuple[str, str]] = [
    ("bull_flag", "bull_pennant"),
    ("bear_flag", "bear_pennant"),
    ("double_top", "triple_top"),
    ("double_bottom", "triple_bottom"),
    ("hammer", "hanging_man"),
    ("doji", "spinning_top"),
    ("head_and_shoulders", "triple_top"),
    ("cup_and_handle", "rounded_bottom"),
]

# both directions, as a set for O(1) lookup
_CONFUSION_SET = set(CONFUSION_PAIRS) | {(b, a) for a, b in CONFUSION_PAIRS}


def _load_labels(split_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """image stem -> [(class, cx, cy, w, h), ...] normalised YOLO boxes."""
    out: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for txt in sorted((split_dir / "labels" / split_dir.name).glob("*.txt")):
        boxes = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c, cx, cy, w, h = parts
                boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
        out[txt.stem] = boxes
    return out


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def mine(
    weights: str,
    dataset_yaml: str,
    split: str = "valid",
    conf: float = 0.25,
    iou_thr: float = 0.5,
    low_conf_thr: float = 0.55,
    max_export: int = 200,
    out_root: str = "artifacts/hard_examples",
) -> dict:
    from ultralytics import YOLO
    import yaml as _yaml

    from ..registry.registry import get_registry

    registry = get_registry()
    id2name = {d.cv_class_id: d.id for d in registry.implemented()}
    ds = _yaml.safe_load(Path(dataset_yaml).read_text())
    ds_root = Path(ds["path"])
    split_dir = ds_root  # images under images/<split>, labels under labels/<split>

    model = YOLO(weights)
    labels = _load_labels(split_dir)
    images = sorted((split_dir / "images" / split).glob("*.png"))
    if not images:
        images = sorted((split_dir / "images" / split).glob("*.jpg"))

    review_dir = Path(out_root) / Path(weights).parent.parent.name / split
    for sub in ("false_positives", "false_negatives", "low_confidence", "confused_pairs"):
        (review_dir / sub).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    pair_counts: Counter = Counter()
    exported = 0

    results = model.predict(
        source=[str(p) for p in images], conf=conf, iou=iou_thr, imgsz=640, verbose=False
    )
    for res, img_path in zip(results, images):
        stem = img_path.stem
        gt = labels.get(stem, [])
        preds = []
        if res.boxes is not None and res.boxes.conf is not None:
            for i in range(len(res.boxes.conf)):
                cls_id = int(res.boxes.cls[i].item())
                cx, cy, w, h = (v / 640.0 for v in res.boxes.xywh[i].tolist())
                preds.append((cls_id, cx, cy, w, h, float(res.boxes.conf[i].item())))

        matched_gt = set()
        matched_pred = set()
        # match predictions to ground truth
        for pi, p in enumerate(preds):
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gt):
                iou = _iou(p[1:5], g[1:5])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_gi >= 0 and best_iou >= iou_thr and gt[best_gi][0] == p[0]:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                if p[5] < low_conf_thr and exported < max_export:
                    shutil.copy(img_path, review_dir / "low_confidence" / img_path.name)
                    stats["low_confidence"] += 1
                    exported += 1
            elif best_gi >= 0 and best_iou >= iou_thr and gt[best_gi][0] != p[0]:
                # right box, wrong class -> confusion pair candidate
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                got, predicted = id2name.get(gt[best_gi][0]), id2name.get(p[0])
                if (got, predicted) in _CONFUSION_SET:
                    pair_counts[(got, predicted)] += 1
                if exported < max_export:
                    shutil.copy(img_path, review_dir / "confused_pairs" / img_path.name)
                    stats["confused_pairs"] += 1
                    exported += 1
        # unmatched predictions = false positives
        for pi, p in enumerate(preds):
            if pi not in matched_pred and exported < max_export:
                shutil.copy(img_path, review_dir / "false_positives" / img_path.name)
                stats["false_positives"] += 1
                exported += 1
                break  # one copy per image is enough
        # unmatched ground truth = false negatives
        if gt and not matched_gt and exported < max_export:
            shutil.copy(img_path, review_dir / "false_negatives" / img_path.name)
            stats["false_negatives"] += 1
            exported += 1

    report = {
        "split": split,
        "n_images": len(images),
        "stats": dict(stats),
        "confusion_pairs": {f"{a} -> {b}": c for (a, b), c in pair_counts.most_common(20)},
        "review_dir": str(review_dir),
        "note": "Copy review images into the training split (or upweight) to focus the next run.",
    }
    report_path = review_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    dataset = args.dataset
    if dataset is None:
        root = Path("artifacts/datasets")
        cands = sorted((p for p in root.iterdir() if (p / "data.yaml").exists()), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("no dataset found")
        dataset = str(cands[-1] / "data.yaml")
    mine(args.weights, dataset, split=args.split, conf=args.conf, max_export=args.max)


if __name__ == "__main__":
    main()
