"""Training checkpointing with full state persistence and auto-resume.

CRITICAL REQUIREMENT (spec §35–39): an interrupted run must resume exactly
where it stopped — never restart from epoch 0 unless explicitly requested.

We wrap the Ultralytics trainer and, in addition to the standard checkpoint
(weights + optimizer + EMA), persist a rich sidecar ``state.json`` +
``training_state.pt`` holding: epoch, global step, RNG states (python,
numpy, torch), scheduler state, best metric, metric history, early-stopping
state, config and dataset manifest hashes. Signals (SIGINT/SIGTERM) trigger
an emergency checkpoint before exit.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np


class CheckpointManager:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.metrics_dir / "history.json"
        self._installed = False

    # ── paths ────────────────────────────────────────────────────────────────
    @property
    def last(self) -> Path:
        return self.ckpt_dir / "last.pt"

    @property
    def best(self) -> Path:
        return self.ckpt_dir / "best.pt"

    @property
    def emergency(self) -> Path:
        return self.ckpt_dir / "emergency.pt"

    def epoch_path(self, epoch: int) -> Path:
        return self.ckpt_dir / f"epoch_{epoch:03d}.pt"

    # ── history ──────────────────────────────────────────────────────────────
    def load_history(self) -> list[dict]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text())
        return []

    def save_history(self, history: list[dict]) -> None:
        tmp = self.history_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(history, indent=2))
        tmp.replace(self.history_path)

    def prune_epoch_checkpoints(self, keep: int) -> None:
        epoch_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        for p in epoch_ckpts[:-keep] if keep > 0 else []:
            p.unlink(missing_ok=True)

    # ── resume detection ─────────────────────────────────────────────────────
    def find_resume_state(self) -> dict[str, Any] | None:
        """Return the richest available resume descriptor, or None."""
        candidates = [self.emergency, self.last]
        for p in candidates:
            if p.exists():
                return {"checkpoint": str(p), "kind": p.stem}
        # fall back to newest epoch checkpoint
        epoch_ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if epoch_ckpts:
            return {"checkpoint": str(epoch_ckpts[-1]), "kind": "epoch"}
        return None

    # ── signal handling ──────────────────────────────────────────────────────
    def install_signal_handlers(self, on_emergency) -> None:
        """``on_emergency()`` must save an emergency checkpoint quickly."""
        if self._installed:
            return

        def handler(signum, frame):
            print(
                f"\n[checkpoint] signal {signal.Signals(signum).name} received — "
                "saving emergency checkpoint before exit...",
                file=sys.stderr,
            )
            try:
                on_emergency()
            finally:
                # restore default and re-raise so the process exits
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:
                pass
        self._installed = True


def snapshot_rng() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    import torch

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def environment_fingerprint() -> dict[str, str]:
    """Reproducibility metadata (spec §40)."""
    import platform

    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "none"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = "not installed"
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        info["git_commit"] = commit or "unknown"
    except Exception:
        info["git_commit"] = "unknown"
    return info
