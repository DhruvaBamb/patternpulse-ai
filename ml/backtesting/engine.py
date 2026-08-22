"""Backtesting engine (spec §45–47).

Re-runs the detection pipeline historically (walk-forward, chronological),
simulates each confirmed pattern's outcome using the pattern's own levels
(target/stop), and aggregates per-pattern statistics. No look-ahead: at each
step the pipeline only sees candles up to that bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..inference.pipeline import DetectionPipeline, PatternFinding


@dataclass
class TradeOutcome:
    pattern_id: str
    symbol: str
    interval: str
    entry_ts: str
    entry: float
    target: float | None
    stop: float | None
    exit_ts: str | None
    exit_price: float | None
    outcome: str            # "target" | "stop" | "timeout"
    return_pct: float
    bars_held: int
    confidence: float
    completion: float

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class BacktestConfig:
    window_bars: int = 200
    step_bars: int = 3
    max_hold_bars: int = 60
    min_confidence: float = 0.55
    require_breakout: bool = True
    cooldown_same_pattern: int = 20   # bars before same pattern can re-enter


@dataclass
class BacktestResult:
    trades: list[TradeOutcome] = field(default_factory=list)

    def summary_by_pattern(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        by_pat: dict[str, list[TradeOutcome]] = {}
        for t in self.trades:
            by_pat.setdefault(t.pattern_id, []).append(t)
        for pid, trades in by_pat.items():
            rets = np.array([t.return_pct for t in trades])
            wins = [t for t in trades if t.return_pct > 0]
            losses = [t for t in trades if t.return_pct <= 0]
            gross_win = sum(t.return_pct for t in wins)
            gross_loss = abs(sum(t.return_pct for t in losses))
            out[pid] = {
                "observations": len(trades),
                "confirmed": len(trades),
                "successful": len(wins),
                "hit_rate": round(len(wins) / len(trades), 3) if trades else 0.0,
                "avg_return_pct": round(float(rets.mean()), 2) if len(rets) else 0.0,
                "median_return_pct": round(float(np.median(rets)), 2) if len(rets) else 0.0,
                "expectancy_pct": round(float(rets.mean()), 2) if len(rets) else 0.0,
                "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
                "max_drawdown_pct": round(float(self._max_dd(rets)), 2) if len(rets) else 0.0,
                "avg_confidence": round(float(np.mean([t.confidence for t in trades])), 3),
                "avg_completion": round(float(np.mean([t.completion for t in trades])), 1),
            }
        return out

    @staticmethod
    def _max_dd(rets: np.ndarray) -> float:
        equity = np.cumprod(1 + rets / 100)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return float(dd.min() * 100) if len(dd) else 0.0

    def overall(self) -> dict:
        rets = np.array([t.return_pct for t in self.trades]) if self.trades else np.array([])
        by_timeframe: dict[str, list[float]] = {}
        for t in self.trades:
            by_timeframe.setdefault(t.interval, []).append(t.return_pct)
        return {
            "n_trades": len(self.trades),
            "hit_rate": round(float((rets > 0).mean()), 3) if len(rets) else 0.0,
            "avg_return_pct": round(float(rets.mean()), 2) if len(rets) else 0.0,
            "median_return_pct": round(float(np.median(rets)), 2) if len(rets) else 0.0,
            "by_timeframe": {
                tf: {"n": len(r), "avg_return_pct": round(float(np.mean(r)), 2)}
                for tf, r in by_timeframe.items()
            },
        }


class BacktestEngine:
    def __init__(self, pipeline: DetectionPipeline, config: BacktestConfig | None = None) -> None:
        self.pipeline = pipeline
        self.cfg = config or BacktestConfig()

    def run(self, df: pd.DataFrame, symbol: str, interval: str) -> list[TradeOutcome]:
        """Walk-forward over the history; returns completed trade outcomes."""
        outcomes: list[TradeOutcome] = []
        last_entry_bar: dict[str, int] = {}
        n = len(df)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)

        for end in range(self.cfg.window_bars, n, self.cfg.step_bars):
            window = df.iloc[: end + 1]
            try:
                findings = self.pipeline.analyze(window, max_findings=3)
            except Exception:
                continue
            for f in findings:
                if f.fused_confidence < self.cfg.min_confidence:
                    continue
                if self.cfg.require_breakout and not f.detected_breakout:
                    continue
                if end - last_entry_bar.get(f.pattern_id, -10**9) < self.cfg.cooldown_same_pattern:
                    continue
                last_entry_bar[f.pattern_id] = end
                outcomes.append(self._simulate(f, window, highs, lows, closes, end, symbol, interval))
        return [o for o in outcomes if o is not None]

    def _simulate(
        self, f: PatternFinding, window: pd.DataFrame, highs, lows, closes, end: int,
        symbol: str, interval: str,
    ) -> TradeOutcome | None:
        entry = float(closes[end])
        target = f.levels.get("target")
        stop = f.levels.get("invalidation") or f.levels.get("neckline")
        if stop is not None:
            # stop must be on the risking side of entry for the direction
            if f.direction == "bullish" and stop >= entry:
                stop = entry * 0.97
            if f.direction == "bearish" and stop <= entry:
                stop = entry * 1.03
        if stop is None:
            stop = entry * (0.97 if f.direction != "bearish" else 1.03)
        if target is None or (f.direction == "bullish" and target <= entry) or (
            f.direction == "bearish" and target >= entry
        ):
            return None

        ts_index = window.index
        hold = 0
        for j in range(end + 1, min(end + 1 + self.cfg.max_hold_bars, len(closes))):
            hold = j - end
            if f.direction == "bullish":
                if highs[j] >= target:
                    exit_price, outcome = target, "target"
                    break
                if lows[j] <= stop:
                    exit_price, outcome = stop, "stop"
                    break
            else:
                if lows[j] <= target:
                    exit_price, outcome = target, "target"
                    break
                if highs[j] >= stop:
                    exit_price, outcome = stop, "stop"
                    break
        else:
            j = min(end + self.cfg.max_hold_bars, len(closes) - 1)
            exit_price, outcome, hold = float(closes[j]), "timeout", j - end

        ret = (exit_price - entry) / entry * 100 * (1 if f.direction != "bearish" else -1)
        return TradeOutcome(
            pattern_id=f.pattern_id, symbol=symbol, interval=interval,
            entry_ts=str(ts_index[end]), entry=entry, target=target, stop=stop,
            exit_ts=str(ts_index[j]) if j < len(ts_index) else None,
            exit_price=float(exit_price), outcome=outcome,
            return_pct=round(ret, 3), bars_held=hold,
            confidence=f.fused_confidence, completion=f.completion.completion,
        )
