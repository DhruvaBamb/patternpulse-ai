"""Model export (PyTorch -> ONNX) with inference latency benchmark (spec §44).

Usage:
    python -m ml.export.export_onnx --weights artifacts/runs/<run>/best.pt [--benchmark] [--n 50]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def export_onnx(weights: str, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    out = model.export(format="onnx", imgsz=imgsz, opset=12, dynamic=False, simplify=True)
    return Path(out)


def benchmark(weights_or_onnx: str, n: int = 50, imgsz: int = 640, device: str = "auto") -> dict:
    """Latency benchmark across a sample of synthetic chart images."""
    import numpy as np

    from ..inference.detector import pick_device

    path = Path(weights_or_onnx)
    if path.suffix == ".onnx":
        import cv2  # opencv headless

        cap = cv2.VideoCapture  # noqa: F841  (presence check only)
        # use onnxruntime if available
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            dummy = np.random.default_rng(0).integers(0, 255, (1, 3, imgsz, imgsz)).astype(np.float32)
            times: list[float] = []
            for i in range(n):
                t0 = time.perf_counter()
                sess.run(None, {inp_name: dummy})
                times.append((time.perf_counter() - t0) * 1000)
            return _summarise(times, backend="onnxruntime-cpu", n=n)
        except ImportError:
            return {"error": "onnxruntime not installed — pip install onnxruntime for ONNX benchmarks"}

    from ultralytics import YOLO

    dev = pick_device(device)
    model = YOLO(str(path))
    dummy = np.random.default_rng(0).integers(0, 255, (imgsz, imgsz, 3)).astype(np.uint8)
    # warmup
    model.predict(source=dummy, imgsz=imgsz, device=dev, verbose=False)
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, device=dev, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return _summarise(times, backend=f"ultralytics-{dev}", n=n)


def _summarise(times_ms: list[float], backend: str, n: int) -> dict:
    return {
        "backend": backend,
        "n_runs": n,
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(statistics.quantiles(times_ms, n=20)[-1], 2) if n >= 20 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    onnx_path = export_onnx(args.weights)
    print(f"[export] ONNX written: {onnx_path}")
    result: dict = {"onnx_path": str(onnx_path)}
    if args.benchmark:
        result["pt_benchmark"] = benchmark(args.weights, n=args.n)
        result["onnx_benchmark"] = benchmark(str(onnx_path), n=args.n)
    out_file = onnx_path.parent / "export_report.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
