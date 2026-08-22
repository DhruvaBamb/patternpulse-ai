"""CV detector wrapper (Ultralytics YOLO).

Loads trained weights when available; otherwise reports *unavailable*
instead of pretending. Device auto-detection prefers CUDA > MPS > CPU so
the same code runs on this Mac and on a GPU box.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Detection:
    pattern_id: str
    confidence: float
    bbox_px: list[float]     # x0,y0,x1,y1 in image pixels
    bbox_bars: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "confidence": round(self.confidence, 4),
            "bbox_px": [round(v, 1) for v in self.bbox_px],
        }


class PatternDetector:
    """Thread-safe wrapper around a trained YOLO model."""

    def __init__(
        self,
        weights: str | Path | None,
        registry,
        device: str = "auto",
        conf_threshold: float = 0.25,
        image_size: int = 640,
    ) -> None:
        self._registry = registry
        self._conf_threshold = conf_threshold
        self._image_size = image_size
        self._device = pick_device(device)
        self._model = None
        self._lock = threading.Lock()
        self.weights_path = str(weights) if weights else None
        self.available = False
        self.unavailable_reason = "no weights configured"
        if weights and Path(weights).exists():
            self._load(Path(weights))

    def _load(self, weights: Path) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(weights))
            self.available = True
            self.unavailable_reason = ""
        except Exception as exc:
            self.available = False
            self.unavailable_reason = f"failed to load weights: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.unavailable_reason or "ready",
            "weights": self.weights_path,
            "device": self._device,
            "conf_threshold": self._conf_threshold,
            "registry_version": self._registry.version,
        }

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on an RGB uint8 image; returns detections sorted by confidence."""
        if not self.available or self._model is None:
            return []
        with self._lock:
            try:
                results = self._model.predict(
                    source=image,
                    conf=self._conf_threshold,
                    imgsz=self._image_size,
                    device=self._device,
                    verbose=False,
                )
            except Exception:
                return []
        out: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.conf is None:
                continue
            for i in range(len(boxes.conf)):
                cls_id = int(boxes.cls[i].item())
                defn = self._registry.cv_id_to_pattern(cls_id)
                if defn is None:
                    continue
                xyxy = boxes.xyxy[i].tolist()
                out.append(
                    Detection(
                        pattern_id=defn.id,
                        confidence=float(boxes.conf[i].item()),
                        bbox_px=[float(v) for v in xyxy],
                    )
                )
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
