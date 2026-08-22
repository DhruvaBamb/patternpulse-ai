"""Signal engine: BUY / SELL / HOLD / WATCH with pattern-specific levels.

A pattern label never directly becomes a signal: the engine requires fused
confidence, formation completion, geometry validation and (for entries) a
confirmed breakout with volume. Every signal carries its reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..geometry.pivots import atr, classify_trend
from .pipeline import PatternFinding

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
WATCH = "WATCH"


@dataclass
class Signal:
    action: str
    pattern_id: str
    pattern_name: str
    completion: float
    confidence: float
    entry: float | None
    target: float | None
    stop_loss: float | None
    invalidation: float | None
    risk_reward: float | None
    reason: str
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "pattern": self.pattern_name,
            "pattern_id": self.pattern_id,
            "completion": round(self.completion, 1),
            "confidence": round(self.confidence, 3),
            "entry": self.entry,
            "target": self.target,
            "stop_loss": self.stop_loss,
            "invalidation": self.invalidation,
            "risk_reward": round(self.risk_reward, 2) if self.risk_reward else None,
            "reason": self.reason,
            "evidence": self.evidence,
            "warnings": self.warnings,
        }


@dataclass
class SignalConfig:
    min_confidence_for_entry: float = 0.65
    min_completion_for_entry: float = 80.0
    breakout_volume_ratio: float = 1.2
    atr_stop_multiplier: float = 1.5
    min_risk_reward: float = 1.2


def _volume_ok(df: pd.DataFrame, ratio: float, lookback: int = 20) -> bool:
    if "volume" not in df.columns or len(df) < lookback:
        return True  # volume unknown => do not block, note it instead
    vol = df["volume"].tail(lookback).to_numpy(dtype=float)
    mean = vol[:-1].mean()
    return bool(vol[-1] >= ratio * mean) if mean > 0 else True


def _levels(finding: PatternFinding, df: pd.DataFrame, cfg: SignalConfig) -> Signal:
    """Compute entry/target/stop/invalidation with pattern-aware rules."""
    lv = finding.levels
    close = float(df["close"].iloc[-1])
    a = atr(df)
    bullish = finding.direction == "bullish"
    entry = None
    target = lv.get("target")
    stop = None
    invalidation = lv.get("invalidation")

    if finding.detected_breakout:
        entry = lv.get("breakout", close)
        if invalidation is None:
            invalidation = close - cfg.atr_stop_multiplier * a if bullish else close + cfg.atr_stop_multiplier * a
        stop = invalidation
    elif a > 0:
        stop = close - cfg.atr_stop_multiplier * a if bullish else close + cfg.atr_stop_multiplier * a

    rr = None
    if entry and target and stop:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else None
    return Signal(
        action=WATCH, pattern_id=finding.pattern_id, pattern_name=finding.pattern_name,
        completion=finding.completion.completion, confidence=finding.fused_confidence,
        entry=entry, target=target, stop_loss=stop, invalidation=invalidation,
        risk_reward=rr, reason="", evidence=[], warnings=[],
    )


def generate_signal(finding: PatternFinding, df: pd.DataFrame, cfg: SignalConfig | None = None) -> Signal:
    cfg = cfg or SignalConfig()
    s = _levels(finding, df, cfg)
    bullish = finding.direction == "bullish"

    ev: list[str] = []
    warn: list[str] = []

    if finding.visual_score is not None:
        ev.append(f"visual model confidence {finding.visual_score:.0%}")
    else:
        warn.append("CV detector unavailable — geometry-only evidence")
    ev.append(f"geometry validation {finding.geometry_score:.0%}")
    ev.append(f"formation completion {finding.completion.completion:.0f}%")
    if finding.detected_breakout:
        if _volume_ok(df, cfg.breakout_volume_ratio):
            ev.append("breakout registered with volume support")
        else:
            warn.append("breakout without volume confirmation")
    else:
        warn.append("breakout not yet confirmed")
    if finding.violations:
        warn.extend(finding.violations[:3])
    s.evidence, s.warnings = ev, warn

    if finding.direction == "neutral" or finding.completion.completion < cfg.min_completion_for_entry * 0.6:
        s.action = WATCH
        s.reason = (
            f"{finding.pattern_name} is {finding.completion.completion:.0f}% formed — too early to act. "
            "Watching for structural completion and breakout confirmation."
        )
        return s

    if not finding.detected_breakout:
        s.action = WATCH
        s.reason = (
            f"{finding.pattern_name} is approaching completion ({finding.completion.completion:.0f}%) "
            "but breakout has not yet been confirmed. Awaiting price acceptance beyond the pattern level."
        )
        return s

    if finding.fused_confidence < cfg.min_confidence_for_entry:
        s.action = WATCH
        s.reason = (
            f"Breakout occurred but fused confidence {finding.fused_confidence:.0%} is below the "
            f"entry threshold ({cfg.min_confidence_for_entry:.0%})."
        )
        return s

    if finding.completion.completion < cfg.min_completion_for_entry:
        s.action = WATCH
        s.reason = (
            f"Breakout with only {finding.completion.completion:.0f}% structural completion — "
            "risk of a partial-pattern failure."
        )
        return s

    action = BUY if bullish else SELL
    rr_ok = s.risk_reward is None or s.risk_reward >= cfg.min_risk_reward
    s.action = action
    s.reason = (
        f"{finding.pattern_name} {'bullish' if bullish else 'bearish'} formation confirmed by breakout "
        f"with {finding.fused_confidence:.0%} fused confidence and {finding.completion.completion:.0f}% completion. "
        + (
            f"Risk/reward ≈ {s.risk_reward:.1f}:1."
            if s.risk_reward
            else "Risk/reward not computable (missing levels)."
        )
        if rr_ok
        else f"Risk/reward {s.risk_reward:.1f}:1 below the {cfg.min_risk_reward}:1 minimum — size accordingly."
    )
    if not rr_ok:
        s.warnings.append(f"risk/reward {s.risk_reward:.1f} below minimum {cfg.min_risk_reward}")
    return s


def no_pattern_signal(df: pd.DataFrame, symbol: str = "") -> Signal:
    trend = classify_trend(df)
    return Signal(
        action=HOLD, pattern_id=None, pattern_name="No qualifying pattern",  # type: ignore[arg-type]
        completion=0.0, confidence=0.0, entry=None, target=None, stop_loss=None,
        invalidation=None, risk_reward=None,
        reason=(
            f"No pattern currently meets confidence/completion gates for {symbol or 'this symbol'} "
            f"(trend: {trend})."
        ),
        evidence=[f"trend context: {trend}"],
        warnings=[],
    )
