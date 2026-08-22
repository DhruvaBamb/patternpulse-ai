# Training

## Commands

```bash
python -m ml.datasets.synthetic.generate --samples 200   # dataset first
python -m ml.training.train                              # fresh (or auto-resume per config)
python -m ml.training.train --auto-resume                # resume if checkpoint exists
python -m ml.training.train --resume                     # force resume from latest
python -m ml.training.train --fresh                      # archive old run, start epoch 0
```

Configuration: `configs/training.yaml` (model arch, epochs, batch, device,
early stopping, checkpoint retention). Device auto-detects CUDA → MPS → CPU,
so the same command works on an Apple-Silicon Mac and a GPU box.

## Checkpoint & resume (critical requirement)

`ml/training/checkpointing.py` guarantees an interrupted run continues where
it stopped — never restarting from epoch 0 unless `--fresh` is explicit:

- Ultralytics persists weights + optimizer + scheduler inside `last.pt`;
  the wrapper mirrors them into `artifacts/runs/<run>/checkpoints/{last,best}.pt`.
- Sidecar state: `run_meta.json` (config, dataset, RNG snapshot, environment
  fingerprint: Python/torch/CUDA/ultralytics versions, git commit),
  `metrics/history.json` (per-epoch metric history).
- **Signals**: SIGINT/SIGTERM handlers save `emergency.pt` + history +
  `EMERGENCY.txt` before exiting. The next `--auto-resume` picks the
  emergency checkpoint first.
- Epoch checkpoints are pruned to `checkpointing.keep_epoch_checkpoints`;
  prior runs are archived (`<run>_archive_<timestamp>`), never deleted.

## Evaluation

```bash
python -m ml.evaluation.evaluate --weights artifacts/runs/<run>/best.pt
python -m ml.evaluation.evaluate --calibrate preds.json --method isotonic
```

Reports mAP@0.5, mAP@0.5:0.95, precision/recall, per-class AP and the
confusion matrix to `artifacts/eval/`. Calibration fits isotonic/Platt maps
on validation confidences and reports Expected Calibration Error before/after.
Store both `raw_model_confidence` and `calibrated_confidence` downstream.

## Ablation / research experiments

- `ml/evaluation/evaluate.py:compare_systems` — CV-only vs geometry-only vs
  hybrid on a window (System A/B/C per spec §93).
- Partial-reveal experiment scaffold (spec §94) — progressively reveal
  completed patterns and measure recognition accuracy + completion error.

## Weights in production

Point `model.weights` in `configs/app.yaml` at the trained `best.pt`; the
backend hot-loads it at startup and `/api/health` then reports
`cv_detector.available: true` (UI switches from "geometry-only" to
"CV + geometry" badges). Weights are git-ignored — train locally.
