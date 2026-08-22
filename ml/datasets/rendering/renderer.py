"""Deterministic candlestick chart renderer.

The exact same renderer + config is used for synthetic dataset generation,
training, validation and live inference, so the CV model always sees the
same image distribution. Rendering is pure: OHLCV in -> RGBA array out,
no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Pillow is the primary path; a numpy fallback keeps CI light
    from PIL import Image, ImageDraw

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


@dataclass
class RenderConfig:
    image_width: int = 640
    image_height: int = 640
    window_bars: int = 96
    background: str = "#14161f"
    bull_body: str = "#26a69a"
    bear_body: str = "#ef5350"
    chart_padding_x: int = 8
    chart_padding_y: int = 14
    candle_gap_ratio: float = 0.35
    wick_width_px: int = 1
    show_volume: bool = False
    volume_height_ratio: float = 0.18
    y_range_padding_ratio: float = 0.02

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RenderConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())["rendering"]
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in allowed})


DEFAULT_CONFIG = RenderConfig()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class RenderResult:
    image: np.ndarray            # uint8 HxWx3
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        if _HAS_PIL:
            Image.fromarray(self.image).save(Path(path))
        else:  # pragma: no cover
            np.save(Path(path).with_suffix(".npy"), self.image)


class CandlestickRenderer:
    """Renders the trailing ``window_bars`` of a frame into a fixed-size image."""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    # ── coordinate helpers ───────────────────────────────────────────────────
    def _scale(self, df: pd.DataFrame) -> tuple[float, float, float, float]:
        """Return (x0, dx, y_min, y_max) used to map bars to pixels."""
        cfg = self.cfg
        n = len(df)
        usable_w = cfg.image_width - 2 * cfg.chart_padding_x
        dx = usable_w / max(n, 1)
        lo = float(df["low"].min())
        hi = float(df["high"].max())
        pad = (hi - lo) * cfg.y_range_padding_ratio if hi > lo else hi * 0.01 + 1e-9
        return cfg.chart_padding_x, dx, lo - pad, hi + pad

    def y_to_px(self, price: float, y_min: float, y_max: float, plot_h: int, top: int) -> float:
        if y_max <= y_min:
            return top + plot_h / 2
        frac = (price - y_min) / (y_max - y_min)
        return top + (1.0 - frac) * plot_h

    # ── main entry ───────────────────────────────────────────────────────────
    def render(self, df: pd.DataFrame, config: RenderConfig | None = None) -> RenderResult:
        cfg = config or self.cfg
        if not _HAS_PIL:  # pragma: no cover
            raise RuntimeError("Pillow required for chart rendering")

        window = df.tail(cfg.window_bars).reset_index(drop=True)
        if window.empty:
            raise ValueError("cannot render an empty frame")

        show_volume = cfg.show_volume and "volume" in window.columns
        vol_h = int(cfg.image_height * cfg.volume_height_ratio) if show_volume else 0
        top = cfg.chart_padding_y
        plot_h = cfg.image_height - 2 * cfg.chart_padding_y - vol_h

        x0, dx, y_min, y_max = self._scale(window)
        candle_w = max(1, int(dx * (1.0 - cfg.candle_gap_ratio)))

        img = Image.new("RGB", (cfg.image_width, cfg.image_height), _hex_to_rgb(cfg.background))
        draw = ImageDraw.Draw(img)

        bull = _hex_to_rgb(cfg.bull_body)
        bear = _hex_to_rgb(cfg.bear_body)

        vol_max = float(window["volume"].max()) if show_volume and window["volume"].max() > 0 else 1.0

        for i, row in window.iterrows():
            cx = int(x0 + dx * (i + 0.5))
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            y_h = self.y_to_px(h, y_min, y_max, plot_h, top)
            y_l = self.y_to_px(l, y_min, y_max, plot_h, top)
            y_o = self.y_to_px(o, y_min, y_max, plot_h, top)
            y_c = self.y_to_px(c, y_min, y_max, plot_h, top)

            color = bull if c >= o else bear
            # wick
            draw.rectangle(
                [cx - max(1, cfg.wick_width_px // 2), y_h, cx + max(1, cfg.wick_width_px // 2), y_l],
                fill=color,
            )
            # body (ensure at least 1px so dojis stay visible)
            body_top, body_bot = min(y_o, y_c), max(y_o, y_c)
            if body_bot - body_top < 1:
                body_bot = body_top + 1
            draw.rectangle(
                [cx - candle_w // 2, body_top, cx + candle_w // 2, body_bot],
                fill=color,
            )
            # volume sub-panel
            if show_volume:
                v = float(row.get("volume", 0.0) or 0.0)
                vh = int((v / vol_max) * (vol_h - 4)) if vol_max > 0 else 0
                vy = cfg.image_height - cfg.chart_padding_y - vh
                draw.rectangle([cx - candle_w // 2, vy, cx + candle_w // 2, cfg.image_height - cfg.chart_padding_y], fill=color)

        arr = np.asarray(img).copy()
        meta = {
            "n_bars": len(window),
            "y_min": y_min,
            "y_max": y_max,
            "x0": x0,
            "dx": dx,
            "plot_h": plot_h,
            "top": top,
            "first_ts": str(window.index[0]) if len(window) else None,
            # mapping helpers for overlays / label conversion
            "last_ts": str(window.iloc[-1:].index[-1]) if len(window) else None,
            "time_index": [str(t) for t in window.index],
        }
        return RenderResult(image=arr, meta=meta)

    # ── bbox helpers (shared by labelling & overlays) ────────────────────────
    def bbox_px(self, df: pd.DataFrame, i0: int, i1: int, p_low: float, p_high: float) -> list[int]:
        """Pixel bbox [x0,y0,x1,y1] for bar-range [i0,i1] within the rendered window.

        Indices are relative to the *window* (0 == oldest rendered bar).
        """
        cfg = self.cfg
        window = df.tail(cfg.window_bars).reset_index(drop=True)
        offset = len(df) - len(window)
        i0 = max(0, i0 - offset)
        i1 = min(len(window) - 1, i1 - offset)
        vol_h = int(cfg.image_height * cfg.volume_height_ratio) if cfg.show_volume else 0
        top = cfg.chart_padding_y
        plot_h = cfg.image_height - 2 * cfg.chart_padding_y - vol_h
        x0, dx, y_min, y_max = self._scale(window)
        x_lo = int(x0 + dx * i0)
        x_hi = int(x0 + dx * (i1 + 1))
        y_hi_px = self.y_to_px(p_high, y_min, y_max, plot_h, top)
        y_lo_px = self.y_to_px(p_low, y_min, y_max, plot_h, top)
        return [max(0, x_lo), max(0, int(y_hi_px)), min(cfg.image_width - 1, x_hi), min(cfg.image_height - 1, int(y_lo_px))]
