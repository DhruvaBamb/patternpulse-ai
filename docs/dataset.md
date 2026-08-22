# Dataset

## Synthetic generation

```bash
python -m ml.datasets.synthetic.generate --samples 200        # ~77 patterns × 200
```

- Procedural OHLC paths per pattern family with controlled variation: scale,
  duration, noise, preceding trend, gaps (via candle open noise), volume,
  price range, and partial completion (truncated spans).
- **Labels are verified, not assumed**: after generation the production
  geometry validator runs on each positive; only detectable samples are kept.
  `manifest.json` reports `stats.failed_verification` transparently.
- Negatives: trend + noise random walks; *hard negatives* are near-miss
  structures (e.g. double top with 9% peak mismatch).
- Output: YOLO format (`images/{train,valid,test}`, `labels/...`, `data.yaml`)
  plus `manifest.json` (registry version, render config, splits, hashes).

## Splits & leakage

Each generated series is independent; split assignment hashes the sample id
(70/15/15) so the same series can never appear in two splits. For real data,
split by underlying time period *before* rendering sliding windows
(see `ml/datasets/real/` — annotation workflow pending).

## Partial-completion labels

A fraction of positives (`partial_frac`, default 0.35) is cut mid-formation;
the label stores the true completion so the partial-detection experiment
(spec §94) can measure Mean Absolute Completion Error.

## Real data (workflow)

1. `python -m scripts.seed_market_data` — caches OHLCV for the universe.
2. Render sliding windows with the same `configs/rendering.yaml`.
3. Annotate in Roboflow/CVAT; export YOLO/COCO into `artifacts/datasets/<v>/`.
4. Merge with synthetic data (class ids come from the registry — do not re-map).

## Known limitations (honest)

- The synthetic→real domain gap is unmeasured until real annotations exist.
- Verify pass rate is ~50% at default tolerances: generators are idealised,
  validators demand exact pivot structure. Kept positives are guaranteed
  detectable; raising generator fidelity is a tracked improvement task.
