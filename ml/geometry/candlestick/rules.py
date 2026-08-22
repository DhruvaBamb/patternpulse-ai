"""Candlestick pattern rules (41 patterns).

Each rule inspects the final bars of the window and returns a
:class:`ValidationResult`. Candlestick patterns complete at the close of
their last candle, so partial detection does not apply — instead, the
rules include *context* checks (trend before the pattern) that feed the
signal engine.

Conventions used throughout:
  body      = |close - open|
  range     = high - low           (always > 0 for usable bars)
  upper     = high - max(open, close)
  lower     = min(open, close) - low
  bull      = close >= open
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import ValidationResult, clamp01, soft_pass
from ..pivots import classify_trend


def _bars(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Last k bars as a list of dict-like rows."""
    return df.tail(k).reset_index(drop=True)


def _c(row) -> dict:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    rng = h - l
    body = abs(c - o)
    return {
        "o": o, "h": h, "l": l, "c": c,
        "range": rng if rng > 0 else 1e-12,
        "body": body,
        "body_ratio": body / (rng if rng > 0 else 1e-12),
        "upper": h - max(o, c),
        "lower": min(o, c) - l,
        "bull": c >= o,
    }


def _avg_range(df: pd.DataFrame, k: int = 20) -> float:
    tail = df.tail(k)
    return float(np.mean(tail["high"] - tail["low"])) or 1e-9


def _trend_before(df: pd.DataFrame, k: int) -> str:
    return classify_trend(df.iloc[:-k] if len(df) > k else df)


# ── Doji family ──────────────────────────────────────────────────────────────

def rule_doji(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"body_ratio_max": 0.05, "variant": "generic", "upper_shadow_ratio_max": 0.15,
         "lower_shadow_ratio_max": 0.15, "min_range_atr": 0.5, "range_atr_max": 0.05} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    b = _c(_bars(df, 1).iloc[0])
    avg = _avg_range(df)

    variant = p["variant"]
    checks = {
        "body_small": b["body_ratio"] <= p["body_ratio_max"] * 1.6,
        "range_meaningful": b["range"] >= p["min_range_atr"] * avg if variant == "long_legged" else True,
    }
    if variant == "dragonfly":
        checks["upper_small"] = b["upper"] / b["range"] <= p["upper_shadow_ratio_max"] * 1.5
    if variant == "gravestone":
        checks["lower_small"] = b["lower"] / b["range"] <= p["lower_shadow_ratio_max"] * 1.5
    if variant == "four_price":
        checks["tiny_range"] = b["range"] <= p["range_atr_max"] * avg * 4
        checks["body_small"] = b["body_ratio"] <= p["body_ratio_max"]
    if variant == "long_legged":
        checks["both_shadows"] = min(b["upper"], b["lower"]) / b["range"] >= 0.2

    s_body = soft_pass(b["body_ratio"], 0.0, p["body_ratio_max"])
    s_shadows = 1.0
    if variant in ("dragonfly",):
        s_shadows = soft_pass(b["upper"] / b["range"], 0.0, p["upper_shadow_ratio_max"] * 1.5)
    if variant in ("gravestone",):
        s_shadows = soft_pass(b["lower"] / b["range"], 0.0, p["lower_shadow_ratio_max"] * 1.5)

    res.geometry_score = clamp01(0.6 * s_body + 0.4 * s_shadows)
    res.matched = res.geometry_score >= 0.5 and all(checks.values())
    if not res.matched:
        failed = [k for k, v in checks.items() if not v]
        if failed:
            res.violations.append(f"failed checks: {failed}")
        else:
            res.violations.append("shape too far from tolerance")
    res.keypoints = {"open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"]}
    res.levels = {"high": b["h"], "low": b["l"]}
    res.span = (len(df) - 1, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Hammer / Inverted Hammer / Hanging Man / Shooting Star ───────────────────

def rule_hammer(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"lower_shadow_body_min": 2.0, "upper_shadow_body_min": 2.0,
         "upper_shadow_ratio_max": 0.25, "lower_shadow_ratio_max": 0.25,
         "body_ratio_max": 0.35, "inverted": False, "context": "downtrend"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    b = _c(_bars(df, 1).iloc[0])
    inverted = bool(p["inverted"])

    long_shadow = b["upper"] if inverted else b["lower"]
    short_shadow = b["lower"] if inverted else b["upper"]
    ratio = long_shadow / max(b["body"], b["range"] * 0.02)

    s_body = soft_pass(b["body_ratio"], 0.0, p["body_ratio_max"])
    s_long = soft_pass(ratio, p["lower_shadow_body_min"] if not inverted else p["upper_shadow_body_min"], 4.0)
    s_short = soft_pass(short_shadow / b["range"], 0.0, (p["upper_shadow_ratio_max"] if not inverted else p["lower_shadow_ratio_max"]) * 1.5)
    trend = _trend_before(df, 1)
    want = p["context"]
    s_context = 1.0 if trend == want else (0.35 if trend == "sideways" else 0.0)

    res.geometry_score = clamp01(0.25 * s_body + 0.45 * s_long + 0.30 * s_short)
    res.matched = res.geometry_score >= 0.5 and s_context >= 0.35
    if s_context < 0.35:
        res.violations.append(f"expected {want} context, found {trend}")
    if s_long < 0.5:
        res.violations.append("shadow not at least twice the body")
    res.keypoints = {"open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"]}
    # invalidation/reference levels
    if inverted:
        res.levels = {"high": b["h"], "low": b["l"], "invalidation": b["h"], "trigger": b["h"]}
    else:
        res.levels = {"high": b["h"], "low": b["l"], "invalidation": b["l"], "trigger": b["h"]}
    res.span = (len(df) - 1, len(df) - 1)
    res.stages_present = ["complete"]
    res.meta = {"trend_before": trend, "shadow_body_ratio": ratio}
    return res


# ── Engulfing ────────────────────────────────────────────────────────────────

def rule_engulfing(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "body_engulf_required": True} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 2)
    if len(bars) < 2:
        res.violations.append("need two bars")
        return res
    prev, cur = _c(bars.iloc[0]), _c(bars.iloc[1])
    bullish = p["direction"] == "bullish"

    prev_opposite = prev["bull"] != bullish
    cur_aligned = cur["bull"] == bullish
    engulfs = (
        cur["o"] <= prev["c"] and cur["c"] >= prev["o"]
    ) if bullish else (
        cur["o"] >= prev["c"] and cur["c"] <= prev["o"]
    )
    body_dom = cur["body"] > prev["body"]

    score = 0.4 * float(prev_opposite) + 0.3 * float(cur_aligned) + 0.3 * float(engulfs and body_dom)
    res.geometry_score = clamp01(score)
    res.matched = prev_opposite and cur_aligned and engulfs and body_dom
    if not res.matched:
        if not prev_opposite:
            res.violations.append("prior candle not opposite color")
        if not engulfs:
            res.violations.append("current body does not engulf prior body")
    trend = _trend_before(df, 2)
    res.keypoints = {"prev_body": [prev["o"], prev["c"]], "cur_body": [cur["o"], cur["c"]]}
    res.levels = {"engulf_high": max(cur["h"], prev["h"]), "engulf_low": min(cur["l"], prev["l"])}
    res.levels["invalidation"] = cur["l"] if bullish else cur["h"]
    res.levels["trigger"] = cur["h"] if bullish else cur["l"]
    res.span = (len(df) - 2, len(df) - 1)
    res.stages_present = ["complete"]
    res.meta = {"trend_before": trend}
    return res


# ── Harami ───────────────────────────────────────────────────────────────────

def rule_harami(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "second_body_ratio_max": 0.5} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 2)
    if len(bars) < 2:
        res.violations.append("need two bars")
        return res
    prev, cur = _c(bars.iloc[0]), _c(bars.iloc[1])
    direction = p["direction"]
    if direction == "cross":
        inside = max(cur["o"], cur["c"]) <= max(prev["o"], prev["c"]) and min(cur["o"], cur["c"]) >= min(prev["o"], prev["c"])
        small = cur["body_ratio"] <= p["second_body_ratio_max"]
        res.matched = inside and small
        res.geometry_score = clamp01(0.6 * float(inside) + 0.4 * float(small))
    else:
        bullish = direction == "bullish"
        prev_big = prev["body_ratio"] >= 0.5
        prev_opposite = prev["bull"] != bullish
        cur_aligned = cur["bull"] == bullish
        inside = max(cur["o"], cur["c"]) < max(prev["o"], prev["c"]) and min(cur["o"], cur["c"]) > min(prev["o"], prev["c"])
        res.matched = prev_big and prev_opposite and cur_aligned and inside
        res.geometry_score = clamp01(
            0.3 * float(prev_big) + 0.2 * float(prev_opposite) + 0.2 * float(cur_aligned) + 0.3 * float(inside)
        )
    if not res.matched:
        res.violations.append("second body not contained in first / colors wrong")
    res.keypoints = {"prev_body": [prev["o"], prev["c"]], "cur_body": [cur["o"], cur["c"]]}
    res.levels = {"pattern_high": max(prev["h"], cur["h"]), "pattern_low": min(prev["l"], cur["l"])}
    res.span = (len(df) - 2, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Morning / Evening Star ───────────────────────────────────────────────────

def rule_star(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "middle_doji": False, "third_close_into_body_min": 0.5} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 3)
    if len(bars) < 3:
        res.violations.append("need three bars")
        return res
    b1, b2, b3 = (_c(bars.iloc[0]), _c(bars.iloc[1]), _c(bars.iloc[2]))
    bullish = p["direction"] == "bullish"

    first_long = b1["body_ratio"] >= 0.55
    first_aligned = b1["bull"] != bullish  # first candle continues prior trend
    second_small = b2["body_ratio"] <= (0.12 if p["middle_doji"] else 0.30)
    third_aligned = b3["bull"] == bullish
    third_long = b3["body_ratio"] >= 0.5

    if bullish:
        gap_down = max(b2["o"], b2["c"]) < min(b1["o"], b1["c"])
        penetration = (b3["c"] - min(b1["o"], b1["c"])) / max(b1["body"], 1e-9)
    else:
        gap_up = min(b2["o"], b2["c"]) > max(b1["o"], b1["c"])
        penetration = (max(b1["o"], b1["c"]) - b3["c"]) / max(b1["body"], 1e-9)
    gap_ok = gap_down if bullish else gap_up
    s_pen = soft_pass(penetration, p["third_close_into_body_min"], 1.2)

    trend = _trend_before(df, 3)
    s_trend = 1.0 if trend == ("down" if bullish else "up") else 0.4

    res.geometry_score = clamp01(
        0.25 * float(first_long and first_aligned) + 0.25 * float(second_small)
        + 0.30 * float(third_aligned and third_long) + 0.20 * s_pen
    )
    res.matched = (
        res.geometry_score >= 0.5 and first_long and second_small and third_aligned and s_pen >= 0.5
    )
    if not res.matched:
        if not second_small:
            res.violations.append("middle candle body not small")
        if not third_aligned:
            res.violations.append("third candle not in reversal direction")
        if s_pen < 0.5:
            res.violations.append("third close does not reach far enough into first body")
    res.keypoints = {"first_body": [b1["o"], b1["c"]], "star_body": [b2["o"], b2["c"]],
                     "third_body": [b3["o"], b3["c"]]}
    res.levels = {
        "pattern_high": max(b1["h"], b2["h"], b3["h"]),
        "pattern_low": min(b1["l"], b2["l"], b3["l"]),
        "trigger": b3["h"] if bullish else b3["l"],
        "invalidation": (min(b1["l"], b2["l"]) if bullish else max(b1["h"], b2["h"])),
    }
    res.span = (len(df) - 3, len(df) - 1)
    res.stages_present = ["complete"]
    res.meta = {"gap_between_first_second": bool(gap_ok), "trend_before": trend}
    return res


# ── Piercing Line / Dark Cloud Cover ─────────────────────────────────────────

def rule_piercing(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "min_penetration": 0.5, "max_penetration": 1.0} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 2)
    if len(bars) < 2:
        res.violations.append("need two bars")
        return res
    prev, cur = _c(bars.iloc[0]), _c(bars.iloc[1])
    bullish = p["direction"] == "bullish"
    trend = _trend_before(df, 2)

    prev_aligned = prev["bull"] != bullish
    cur_aligned = cur["bull"] == bullish
    if bullish:
        gap = cur["o"] < prev["c"]
        mid = (prev["o"] + prev["c"]) / 2
        penetration = (cur["c"] - mid) / max(prev["body"], 1e-9) + 0.5
    else:
        gap = cur["o"] > prev["c"]
        mid = (prev["o"] + prev["c"]) / 2
        penetration = (mid - cur["c"]) / max(prev["body"], 1e-9) + 0.5
    s_pen = soft_pass(penetration, p["min_penetration"], p["max_penetration"])

    res.geometry_score = clamp01(0.3 * float(prev_aligned) + 0.3 * float(cur_aligned and gap) + 0.4 * s_pen)
    res.matched = res.geometry_score >= 0.5 and prev_aligned and cur_aligned and s_pen >= 0.5
    if s_pen < 0.5:
        res.violations.append("close does not pierce the midpoint of prior body")
    if not gap:
        res.violations.append("no opening gap in reversal direction")
    res.keypoints = {"prev_body": [prev["o"], prev["c"]], "cur_body": [cur["o"], cur["c"]]}
    res.levels = {"midpoint": float(mid), "trigger": cur["h"] if bullish else cur["l"],
                  "invalidation": cur["l"] if bullish else cur["h"]}
    res.span = (len(df) - 2, len(df) - 1)
    res.stages_present = ["complete"]
    res.meta = {"penetration": penetration, "trend_before": trend}
    return res


# ── Three White Soldiers / Black Crows ──────────────────────────────────────

def rule_three_soldiers(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 3)
    if len(bars) < 3:
        res.violations.append("need three bars")
        return res
    c1, c2, c3 = _c(bars.iloc[0]), _c(bars.iloc[1]), _c(bars.iloc[2])
    bullish = p["direction"] == "bullish"

    strong = all(x["body_ratio"] >= 0.55 for x in (c1, c2, c3))
    aligned = all(x["bull"] == bullish for x in (c1, c2, c3))
    if bullish:
        progression = c1["c"] < c2["c"] < c3["c"] and c2["o"] > c1["o"] and c3["o"] > c2["o"]
        opens_inside = c2["o"] < c1["c"] and c3["o"] < c2["c"]
    else:
        progression = c1["c"] > c2["c"] > c3["c"] and c2["o"] < c1["o"] and c3["o"] < c2["o"]
        opens_inside = c2["o"] > c1["c"] and c3["o"] > c2["c"]

    res.geometry_score = clamp01(0.35 * float(strong) + 0.35 * float(aligned) + 0.30 * float(progression))
    res.matched = strong and aligned and progression
    if not res.matched:
        if not aligned:
            res.violations.append("candles not uniformly colored")
        if not progression:
            res.violations.append("closes not progressing in pattern direction")
    res.keypoints = {"close1": c1["c"], "close2": c2["c"], "close3": c3["c"]}
    res.levels = {"pattern_high": max(c1["h"], c2["h"], c3["h"]),
                  "pattern_low": min(c1["l"], c2["l"], c3["l"]),
                  "trigger": c3["h"] if bullish else c3["l"],
                  "invalidation": c1["l"] if bullish else c1["h"]}
    res.span = (len(df) - 3, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Tweezers ─────────────────────────────────────────────────────────────────

def rule_tweezers(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"side": "top", "match_tol": 0.0015} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 2)
    if len(bars) < 2:
        res.violations.append("need two bars")
        return res
    prev, cur = _c(bars.iloc[0]), _c(bars.iloc[1])
    top = p["side"] == "top"

    a, b = (prev["h"], cur["h"]) if top else (prev["l"], cur["l"])
    tol = p["match_tol"] * max(abs(a), abs(b), 1e-9)
    s_match = soft_pass(abs(a - b), 0.0, tol * 2)
    colors_opposite = prev["bull"] != cur["bull"] if top else prev["bull"] != cur["bull"]
    want_colors = (not prev["bull"] and cur["bull"]) if not top else (prev["bull"] and not cur["bull"])
    trend = _trend_before(df, 2)

    res.geometry_score = clamp01(0.6 * s_match + 0.2 * float(want_colors) + 0.2 * float(trend == ("up" if top else "down")))
    res.matched = s_match >= 0.6 and want_colors
    if not res.matched:
        if s_match < 0.6:
            res.violations.append(f"{'highs' if top else 'lows'} do not match closely")
        else:
            res.violations.append("candle colors not in required order")
    res.keypoints = {"level": float(np.mean([a, b])), "first_extreme": float(a), "second_extreme": float(b)}
    res.levels = {"match_level": float(np.mean([a, b])), "trigger": float(min(prev["l"], cur["l"]) if top else max(prev["h"], cur["h"])),
                  "invalidation": float(np.mean([a, b]))}
    res.span = (len(df) - 2, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Marubozu ─────────────────────────────────────────────────────────────────

def rule_marubozu(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "shadow_ratio_max": 0.05, "min_body_atr": 0.8} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    b = _c(_bars(df, 1).iloc[0])
    avg = _avg_range(df)
    bullish = p["direction"] == "bullish"

    s_shadow = soft_pass(max(b["upper"], b["lower"]) / b["range"], 0.0, p["shadow_ratio_max"] * 2)
    s_body = soft_pass(b["body"] / avg, p["min_body_atr"], p["min_body_atr"] * 3)
    aligned = b["bull"] == bullish

    res.geometry_score = clamp01(0.5 * s_shadow + 0.5 * s_body)
    res.matched = res.geometry_score >= 0.5 and aligned
    if not aligned:
        res.violations.append("candle color opposite to pattern direction")
    if s_shadow < 0.5:
        res.violations.append("shadows too large for marubozu")
    res.keypoints = {"open": b["o"], "close": b["c"]}
    res.levels = {"high": b["h"], "low": b["l"], "trigger": b["h"] if bullish else b["l"],
                  "invalidation": b["o"]}
    res.span = (len(df) - 1, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Spinning Top ─────────────────────────────────────────────────────────────

def rule_spinning_top(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"body_ratio_max": 0.3, "body_ratio_min": 0.06, "shadow_symmetry_tol": 0.5} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    b = _c(_bars(df, 1).iloc[0])

    s_body = soft_pass(b["body_ratio"], p["body_ratio_min"], p["body_ratio_max"])
    up, lo = b["upper"] / b["range"], b["lower"] / b["range"]
    s_sym = soft_pass(abs(up - lo), 0.0, p["shadow_symmetry_tol"])
    res.geometry_score = clamp01(0.5 * s_body + 0.5 * s_sym)
    res.matched = res.geometry_score >= 0.5
    if not res.matched:
        res.violations.append("body/shadow proportions outside spinning-top range")
    res.keypoints = {"open": b["o"], "close": b["c"]}
    res.levels = {"high": b["h"], "low": b["l"]}
    res.span = (len(df) - 1, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Belt Hold ────────────────────────────────────────────────────────────────

def rule_belt_hold(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "context": "downtrend"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    b = _c(_bars(df, 1).iloc[0])
    avg = _avg_range(df)
    bullish = p["direction"] == "bullish"

    no_shadow = (b["lower"] / b["range"] <= 0.02) if bullish else (b["upper"] / b["range"] <= 0.02)
    long_body = b["body_ratio"] >= 0.6
    closes_far = (b["c"] >= b["h"] - b["range"] * 0.1) if bullish else (b["c"] <= b["l"] + b["range"] * 0.1)
    aligned = b["bull"] == bullish
    trend = _trend_before(df, 1)
    s_trend = 1.0 if trend == p["context"] else 0.3

    res.geometry_score = clamp01(0.3 * float(no_shadow) + 0.3 * float(long_body) + 0.2 * float(closes_far) + 0.2 * s_trend)
    res.matched = no_shadow and long_body and aligned and s_trend >= 0.3
    if not res.matched:
        if not no_shadow:
            res.violations.append("candle has a shadow on the open side")
        if not long_body:
            res.violations.append("body not long enough")
    res.keypoints = {"open": b["o"], "close": b["c"]}
    res.levels = {"open": b["o"], "trigger": b["h"] if bullish else b["l"],
                  "invalidation": b["o"]}
    res.span = (len(df) - 1, len(df) - 1)
    res.stages_present = ["complete"]
    res.meta = {"trend_before": trend, "avg_range": avg}
    return res


# ── Three Inside Up/Down ────────────────────────────────────────────────────

def rule_three_inside(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 3)
    if len(bars) < 3:
        res.violations.append("need three bars")
        return res
    b1, b2, b3 = _c(bars.iloc[0]), _c(bars.iloc[1]), _c(bars.iloc[2])
    bullish = p["direction"] == "bullish"

    harami_ok = (
        b1["bull"] != bullish
        and b2["bull"] == bullish
        and max(b2["o"], b2["c"]) < max(b1["o"], b1["c"])
        and min(b2["o"], b2["c"]) > min(b1["o"], b1["c"])
    )
    confirm = b3["bull"] == bullish and (b3["c"] > b1["h"] if bullish else b3["c"] < b1["l"])
    res.geometry_score = clamp01(0.5 * float(harami_ok) + 0.5 * float(confirm))
    res.matched = harami_ok and confirm
    if not harami_ok:
        res.violations.append("first two bars not a harami")
    if not confirm:
        res.violations.append("third candle does not confirm beyond first candle's extreme")
    res.levels = {"trigger": b3["c"], "invalidation": (min(b1["l"], b2["l"]) if bullish else max(b1["h"], b2["h"]))}
    res.span = (len(df) - 3, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Three Outside Up/Down ───────────────────────────────────────────────────

def rule_three_outside(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 3)
    if len(bars) < 3:
        res.violations.append("need three bars")
        return res
    b1, b2, b3 = _c(bars.iloc[0]), _c(bars.iloc[1]), _c(bars.iloc[2])
    bullish = p["direction"] == "bullish"

    engulf_ok = (
        b1["bull"] != bullish and b2["bull"] == bullish and b2["body"] > b1["body"]
        and (b2["o"] <= b1["c"] and b2["c"] >= b1["o"]) if bullish else (b2["o"] >= b1["c"] and b2["c"] <= b1["o"])
    )
    confirm = b3["bull"] == bullish and (b3["c"] > b2["h"] if bullish else b3["c"] < b2["l"])
    res.geometry_score = clamp01(0.5 * float(engulf_ok) + 0.5 * float(confirm))
    res.matched = bool(engulf_ok) and confirm
    if not engulf_ok:
        res.violations.append("first two bars not an engulfing")
    if not confirm:
        res.violations.append("third candle does not confirm beyond the engulfing body")
    res.levels = {"trigger": b3["c"], "invalidation": (b1["l"] if bullish else b1["h"])}
    res.span = (len(df) - 3, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Rising / Falling Three Methods ──────────────────────────────────────────

def rule_three_methods(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 5)
    if len(bars) < 5:
        res.violations.append("need five bars")
        return res
    bs = [_c(bars.iloc[i]) for i in range(5)]
    bullish = p["direction"] == "bullish"

    big1 = bs[0]["body_ratio"] >= 0.6 and bs[0]["bull"] == bullish
    big2 = bs[4]["body_ratio"] >= 0.5 and bs[4]["bull"] == bullish
    middles_ok = all(
        x["body_ratio"] <= 0.35 and x["bull"] != bullish for x in bs[1:4]
    )
    if bullish:
        contained = (
            max(max(x["h"] for x in bs[1:4]), bs[4]["o"]) < bs[0]["c"] + bs[0]["range"] * 0.1
            and min(x["l"] for x in bs[1:4]) > bs[0]["o"]
            and bs[4]["c"] > bs[0]["c"]
        )
    else:
        contained = (
            min(min(x["l"] for x in bs[1:4]), bs[4]["o"]) > bs[0]["c"] - bs[0]["range"] * 0.1
            and max(x["h"] for x in bs[1:4]) < bs[0]["o"]
            and bs[4]["c"] < bs[0]["c"]
        )
    res.geometry_score = clamp01(0.3 * float(big1) + 0.2 * float(middles_ok) + 0.3 * float(big2) + 0.2 * float(contained))
    res.matched = big1 and big2 and middles_ok and contained
    if not res.matched:
        res.violations.append("candles do not satisfy three-methods structure")
    res.levels = {"trigger": bs[4]["c"], "invalidation": (bs[0]["o"] if bullish else bs[0]["o"])}
    res.span = (len(df) - 5, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Tasuki Gap ───────────────────────────────────────────────────────────────

def rule_tasuki_gap(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 3)
    if len(bars) < 3:
        res.violations.append("need three bars")
        return res
    b1, b2, b3 = _c(bars.iloc[0]), _c(bars.iloc[1]), _c(bars.iloc[2])
    bullish = p["direction"] == "bullish"

    if bullish:
        gap = min(b2["o"], b2["c"]) > max(b1["o"], b1["c"])
        aligned12 = b1["bull"] and b2["bull"]
        third_opens_inside = b3["o"] > min(b2["o"], b2["c"]) and b3["o"] < max(b2["o"], b2["c"])
        closes_into_gap = b3["c"] < b2["o"] and b3["c"] > b1["c"]
        third_color = not b3["bull"]
    else:
        gap = max(b2["o"], b2["c"]) < min(b1["o"], b1["c"])
        aligned12 = (not b1["bull"]) and (not b2["bull"])
        third_opens_inside = b3["o"] < max(b2["o"], b2["c"]) and b3["o"] > min(b2["o"], b2["c"])
        closes_into_gap = b3["c"] > b2["o"] and b3["c"] < b1["c"]
        third_color = b3["bull"]

    checks = {"gap": gap, "aligned12": aligned12, "third_opens_inside": third_opens_inside,
              "closes_into_gap_without_filling": closes_into_gap, "third_color": third_color}
    res.geometry_score = clamp01(sum(checks.values()) / len(checks))
    res.matched = all(checks.values())
    if not res.matched:
        res.violations.append(f"failed: {[k for k, v in checks.items() if not v]}")
    res.levels = {"gap_top": float(max(b1["c"], b2["o"]) if bullish else max(b2["c"], b1["o"])),
                  "gap_bottom": float(min(b2["o"], b1["c"]) if bullish else min(b2["c"], b1["o"]))}
    res.span = (len(df) - 3, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Abandoned Baby ──────────────────────────────────────────────────────────

def rule_abandoned_baby(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 3)
    if len(bars) < 3:
        res.violations.append("need three bars")
        return res
    b1, b2, b3 = _c(bars.iloc[0]), _c(bars.iloc[1]), _c(bars.iloc[2])
    bullish = p["direction"] == "bullish"
    doji = b2["body_ratio"] <= 0.08

    if bullish:
        gap1 = b2["h"] < b1["l"]
        gap2 = b3["l"] > b2["h"]
        colors = (not b1["bull"]) and b3["bull"]
    else:
        gap1 = b2["l"] > b1["h"]
        gap2 = b3["h"] < b2["l"]
        colors = b1["bull"] and (not b3["bull"])

    res.geometry_score = clamp01((float(gap1) + float(gap2) + float(doji) + float(colors)) / 4)
    res.matched = gap1 and gap2 and doji and colors
    if not res.matched:
        res.violations.append("doji not isolated by gaps on both sides")
    res.levels = {"baby_high": b2["h"], "baby_low": b2["l"],
                  "trigger": b3["h"] if bullish else b3["l"],
                  "invalidation": b2["h"] if bullish else b2["l"]}
    res.span = (len(df) - 3, len(df) - 1)
    res.stages_present = ["complete"]
    return res


# ── Kicking ──────────────────────────────────────────────────────────────────

def rule_kicking(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bars = _bars(df, 2)
    if len(bars) < 2:
        res.violations.append("need two bars")
        return res
    prev, cur = _c(bars.iloc[0]), _c(bars.iloc[1])
    bullish = p["direction"] == "bullish"
    maru = lambda x: max(x["upper"], x["lower"]) / x["range"] <= 0.05  # noqa: E731

    prev_marubozu = maru(prev) and prev["bull"] != bullish
    cur_marubozu = maru(cur) and cur["bull"] == bullish
    gap = (cur["l"] > prev["h"]) if bullish else (cur["h"] < prev["l"])

    res.geometry_score = clamp01((float(prev_marubozu) + float(cur_marubozu) + float(gap)) / 3)
    res.matched = prev_marubozu and cur_marubozu and gap
    if not res.matched:
        res.violations.append("requires two opposite marubozu separated by a gap")
    res.levels = {"gap_level": float(prev["h"] if bullish else prev["l"]),
                  "trigger": cur["h"] if bullish else cur["l"],
                  "invalidation": prev["h"] if bullish else prev["l"]}
    res.span = (len(df) - 2, len(df) - 1)
    res.stages_present = ["complete"]
    return res


CANDLESTICK_RULES = {
    "doji": rule_doji,
    "hammer": rule_hammer,
    "engulfing": rule_engulfing,
    "harami": rule_harami,
    "star": rule_star,
    "piercing": rule_piercing,
    "three_soldiers": rule_three_soldiers,
    "tweezers": rule_tweezers,
    "marubozu": rule_marubozu,
    "spinning_top": rule_spinning_top,
    "belt_hold": rule_belt_hold,
    "three_inside": rule_three_inside,
    "three_outside": rule_three_outside,
    "three_methods": rule_three_methods,
    "tasuki_gap": rule_tasuki_gap,
    "abandoned_baby": rule_abandoned_baby,
    "kicking": rule_kicking,
}
