"""OHLCV validation & sanitization.

A corrupted candle must never silently propagate into rendering, geometry
or model inference. ``validate_ohlcv`` returns a cleaned frame plus a
structured issue report; callers decide whether issues are fatal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class ValidationReport:
    issues: list[dict] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    fatal: bool = False

    @property
    def ok(self) -> bool:
        return not self.fatal

    def add(self, code: str, detail: str, fatal: bool = False) -> None:
        self.issues.append({"code": code, "detail": detail, "fatal": fatal})
        if fatal:
            self.fatal = True

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "n_issues": len(self.issues),
            "issues": self.issues,
        }


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and sanitize an OHLCV frame (index = datetime).

    Returns (cleaned_df, report). The cleaned frame keeps only rows that pass
    hard OHLC sanity; duplicates and disorder are repaired, timezone is
    normalized to UTC, missing candles are only *reported* (interpolation of
    price history would fabricate data).
    """
    report = ValidationReport(rows_before=len(df))
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as exc:
            report.add("bad_index", f"index not datetime-convertible: {exc}", fatal=True)
            return out, report

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing_cols:
        report.add("missing_columns", f"missing {missing_cols}", fatal=True)
        return out, report
    out = out[REQUIRED_COLUMNS]

    # timezone normalization
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
        report.add("tz_assumed", "naive timestamps assumed UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    # duplicates
    dup = out.index.duplicated(keep="last")
    if dup.any():
        report.add("duplicate_timestamps", f"removed {int(dup.sum())} duplicate timestamps")
        out = out[~dup]

    # chronology
    if not out.index.is_monotonic_increasing:
        report.add("unordered", "timestamps reordered to chronological order")
        out = out.sort_index()

    # numeric coercion
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # non-finite prices
    bad_na = out[["open", "high", "low", "close"]].isna().any(axis=1)
    if bad_na.any():
        report.add("non_finite_price", f"dropped {int(bad_na.sum())} rows with NaN/inf price")
        out = out[~bad_na]

    # OHLC relationships
    bad_ohlc = (
        (out["high"] < out[["open", "close", "low"]].max(axis=1) - 1e-12)
        | (out["low"] > out[["open", "close", "high"]].min(axis=1) + 1e-12)
        | (out["high"] < out["low"])
    )
    if bad_ohlc.any():
        report.add("ohlc_violation", f"dropped {int(bad_ohlc.sum())} bars violating high>=max(o,c,l)")
        out = out[~bad_ohlc]

    # non-positive prices
    nonpos = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if nonpos.any():
        report.add("non_positive_price", f"dropped {int(nonpos.sum())} bars with price <= 0")
        out = out[~nonpos]

    # suspicious zero volume
    if "volume" in out:
        zero_vol = out["volume"].fillna(0) == 0
        if zero_vol.any():
            report.add(
                "zero_volume",
                f"{int(zero_vol.sum())} bars with zero/absent volume (kept, flagged)",
            )

    # abnormal gaps (> expected interval * 5 with no data) — information only
    if len(out) > 2:
        gaps = out.index.to_series().diff().dt.total_seconds()
        typical = gaps.median()
        if typical and typical > 0:
            abnormal = gaps > typical * 5
            if abnormal.any():
                first = out.index[np.argmax(abnormal.values)]
                report.add(
                    "abnormal_gap",
                    f"{int(abnormal.sum())} gaps > 5x median interval; first at {first}",
                )

    if len(out) < 30:
        report.add(
            "insufficient_bars",
            f"only {len(out)} usable bars for {symbol or 'series'} (need >= 30 for pattern work)",
            fatal=False,
        )

    report.rows_after = len(out)
    return out, report
