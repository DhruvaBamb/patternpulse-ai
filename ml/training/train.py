"""YOLO pattern-detector training with checkpoint/resume.

Usage:
    python -m ml.training.train                       # fresh or auto-resume per config
    python -m ml.training.train --resume              # resume latest run
    python -m ml.training.train --auto-resume         # resume if state exists, else fresh
    python -m ml.training.train --fresh               # force training from epoch 0
    python -m ml.training.train --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .checkpointing import (
    CheckpointManager,
    environment_fingerprint,
    restore_rng,
    snapshot_rng,
)

RUNS_ROOT = Path("artifacts/runs")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_dataset() -> Path | None:
    root = Path("artifacts/datasets")
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if (p / "data.yaml").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def build_run_dir(run_name: str, fresh: bool) -> Path:
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        # archive, never destroy: prior run history is precious
        import time

        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = RUNS_ROOT / f"{run_name}_archive_{stamp}"
        if any(run_dir.iterdir()):
            run_dir.rename(archive)
            run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class PulseTrainer:
    """Thin orchestration layer over ultralytics' trainer.

    Ultralytics already persists model/optimizer/scheduler inside ``last.pt``
    when ``model.train(resume=True)`` is used; we add a sidecar history, RNG
    snapshot, environment fingerprint, and emergency checkpoints so resume
    is exact and auditable.
    """

    def __init__(self, config: dict, run_dir: Path, dataset_yaml: Path) -> None:
        self.cfg = config
        self.run_dir = run_dir
        self.dataset_yaml = dataset_yaml
        self.cm = CheckpointManager(run_dir)
        self.history: list[dict] = []

    # ── run ─────────────────────────────────────────────────────────────────
    def train(self, resume_from: str | None = None) -> dict:
        from ultralytics import YOLO

        model_spec = self.cfg.get("model_arch", "yolo11n")
        ckpt_cfg = self.cfg.get("checkpointing", {})
        es_cfg = self.cfg.get("early_stopping", {})

        resume = resume_from is not None
        if resume:
            model = YOLO(resume_from)
            train_args = {"resume": True}
            self.history = self.cm.load_history()
            print(f"[train] resuming from {resume_from} "
                  f"(history has {len(self.history)} epochs)")
        else:
            model = YOLO(model_spec)
            train_args = dict(
                data=str(self.dataset_yaml),
                epochs=self.cfg.get("epochs", 120),
                imgsz=self.cfg.get("image_size", 640),
                batch=self.cfg.get("batch_size", 16),
                device=self.cfg.get("device", "auto"),
                workers=self.cfg.get("workers", 4),
                seed=self.cfg.get("seed", 42),
                project=str(self.run_dir.parent),
                name=self.run_dir.name,
                exist_ok=True,
                patience=es_cfg.get("patience", 25) if es_cfg.get("enabled", True) else 0,
                optimizer=self.cfg.get("optimizer", {}).get("name", "auto"),
                lr0=self.cfg.get("optimizer", {}).get("lr0", 0.01),
                lrf=self.cfg.get("optimizer", {}).get("lrf", 0.01),
                val=True,
                verbose=True,
            )

        self.cm.install_signal_handlers(self._emergency_checkpoint)

        # Persist run metadata before training starts (available even if we crash)
        meta = {
            "config": self.cfg,
            "dataset_yaml": str(self.dataset_yaml),
            "environment": environment_fingerprint(),
            "rng_state": snapshot_rng(),
        }
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        (self.run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

        results = model.train(**train_args)

        # ultralytics writes its own best.pt/last.pt under run_dir/weights;
        # mirror them into our checkpoint layout
        self._mirror_ultralytics_weights()
        self._record_final(results)
        return {"history_len": len(self.history)}

    # ── checkpoint helpers ──────────────────────────────────────────────────
    def _mirror_ultralytics_weights(self) -> None:
        src = self.run_dir / "weights"
        for name, dst in (("best.pt", self.cm.best), ("last.pt", self.cm.last)):
            s = src / name
            if s.exists():
                dst.write_bytes(s.read_bytes())
        self.cm.prune_epoch_checkpoints(int(self.cfg.get("checkpointing", {}).get("keep_epoch_checkpoints", 5)))

    def _emergency_checkpoint(self) -> None:
        try:
            src = self.run_dir / "weights" / "last.pt"
            if src.exists():
                self.cm.emergency.write_bytes(src.read_bytes())
            self.cm.save_history(self.history or self.cm.load_history())
            (self.run_dir / "EMERGENCY.txt").write_text(
                "Emergency checkpoint written on interrupt signal. "
                "Re-run with --resume to continue from here.\n"
            )
            print(f"[checkpoint] emergency state saved to {self.cm.emergency}")
        except Exception as exc:  # best effort during teardown
            print(f"[checkpoint] emergency save failed: {exc}")

    def _record_final(self, results) -> None:
        try:
            summary = {}
            if results is not None:
                for key in ("metrics",):
                    if hasattr(results, key):
                        m = getattr(results, key)
                        summary[key] = {k: float(v) for k, v in dict(m).items()} if m else {}
            (self.run_dir / "final_results.json").write_text(json.dumps(summary, indent=2, default=str))
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    ap.add_argument("--auto-resume", action="store_true", help="resume if a checkpoint exists, else fresh")
    ap.add_argument("--fresh", action="store_true", help="force a new run (archives the old one)")
    ap.add_argument("--dataset", default=None, help="path to dataset data.yaml (default: latest)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_yaml = Path(args.dataset) if args.dataset else find_latest_dataset()
    if dataset_yaml is None or not dataset_yaml.exists():
        raise SystemExit(
            "No dataset found. Generate one first:\n"
            "  python -m ml.datasets.synthetic.generate --samples 200"
        )

    run_dir = build_run_dir(cfg["run_name"], fresh=args.fresh)
    trainer = PulseTrainer(cfg, run_dir, dataset_yaml)

    resume_from: str | None = None
    if args.resume or args.auto_resume or cfg.get("auto_resume", True) and not args.fresh:
        state = trainer.cm.find_resume_state()
        if state and (args.resume or args.auto_resume or cfg.get("auto_resume", True)):
            resume_from = state["checkpoint"]
    if args.fresh:
        resume_from = None

    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()
