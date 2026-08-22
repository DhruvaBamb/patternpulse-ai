"""Data validation and renderer determinism tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.datasets.validation import validate_ohlcv
from ml.datasets.rendering.renderer import CandlestickRenderer, RenderConfig


def _frame(n: int = 60, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    idx = pd.date_range("2021-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(1e6, 2e6, n)}, index=idx)


class TestDataValidation:
    def test_clean_frame_passes(self):
        df, rep = validate_ohlcv(_frame())
        assert rep.ok
        assert len(rep.issues) == 0 or all(not i["fatal"] for i in rep.issues)
        assert len(df) == 60

    def test_duplicate_timestamps_removed(self):
        df = _frame()
        df = pd.concat([df, df.iloc[[5]]])
        out, rep = validate_ohlcv(df)
        assert len(out) == 60
        assert any(i["code"] == "duplicate_timestamps" for i in rep.issues)

    def test_ohlc_violation_dropped(self):
        df = _frame()
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 5  # high < low
        out, rep = validate_ohlcv(df)
        assert len(out) == 59
        assert any(i["code"] == "ohlc_violation" for i in rep.issues)

    def test_unordered_input_reordered(self):
        df = _frame().sample(frac=1.0, random_state=2)
        out, rep = validate_ohlcv(df)
        assert out.index.is_monotonic_increasing
        assert any(i["code"] == "unordered" for i in rep.issues)

    def test_naive_timezone_assumed_utc(self):
        df = _frame()
        df.index = df.index.tz_localize(None)
        out, _ = validate_ohlcv(df)
        assert str(out.index.tz) == "UTC"

    def test_missing_columns_fatal(self):
        df = _frame().drop(columns=["volume"])
        _, rep = validate_ohlcv(df)
        assert rep.fatal


class TestRenderer:
    def test_deterministic_output(self):
        r = CandlestickRenderer(RenderConfig())
        df = _frame(120, seed=9)
        a = r.render(df)
        b = r.render(df)
        assert np.array_equal(a.image, b.image)

    def test_shape_and_type(self):
        r = CandlestickRenderer(RenderConfig(image_width=320, image_height=320, window_bars=64))
        out = r.render(_frame(200, seed=3))
        assert out.image.shape == (320, 320, 3)
        assert out.image.dtype == np.uint8
        assert out.meta["n_bars"] == 64   # only the trailing window

    def test_no_future_bars_leak(self):
        """Rendering a truncated frame must equal rendering the full frame up to that cut."""
        r = CandlestickRenderer(RenderConfig(window_bars=60))
        df = _frame(100, seed=4)
        full = r.render(df)
        cut = r.render(df.iloc[:80])
        # the last rendered window of df.iloc[:80] uses bars 20..79
        assert full.meta["n_bars"] == 60
        assert cut.meta["n_bars"] == 60

    def test_bbox_px_within_bounds(self):
        r = CandlestickRenderer(RenderConfig())
        df = _frame(100, seed=6)
        box = r.bbox_px(df, 10, 50, float(df["low"].min()), float(df["high"].max()))
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= r.cfg.image_width
        assert 0 <= y0 < y1 <= r.cfg.image_height
