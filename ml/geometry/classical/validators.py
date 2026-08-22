"""Classical chart-pattern geometry validators.

Each validator receives the OHLCV window (pandas DataFrame) and the pattern's
``validator_params`` from the registry, and returns a :class:`ValidationResult`.
Validators deliberately support *partial* structures: they enumerate the
stages already present so the completion estimator can convert structure
into a formation percentage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import ValidationResult, clamp01, near, soft_pass
from ..pivots import alternate_pivots, atr, classify_trend, find_pivots, fit_line, Pivot


def _pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    return alternate_pivots(find_pivots(df, left=left, right=right))


def _kp(p: Pivot) -> dict:
    return {"index": int(p.index), "price": float(p.price)}


# ── Head & Shoulders ─────────────────────────────────────────────────────────

def validate_head_and_shoulders(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {  # defaults; registry overrides
        "inverse": False,
        "shoulder_symmetry_tol": 0.05,
        "head_prominence_min": 0.015,
        "neckline_fit_max_rmse": 0.02,
    } | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)

    piv = _pivots(df)
    want = "low" if p["inverse"] else "high"
    extrema = [q for q in piv if q.kind == want]
    troughs = [q for q in piv if q.kind != want]
    if len(extrema) < 3 or len(troughs) < 2:
        res.violations.append("insufficient pivots for 5-point structure")
        return res

    best: dict | None = None
    triples: list[tuple[Pivot, Pivot, Pivot, Pivot, Pivot]] = []
    # build (LS, T1, HEAD, T2, RS) candidate structures from alternating pivots
    seq = piv
    for i in range(len(seq) - 4):
        window5 = seq[i : i + 5]
        kinds = [q.kind for q in window5]
        if kinds == [want, "low" if want == "high" else "high", want, "low" if want == "high" else "high", want]:
            triples.append(tuple(window5))  # type: ignore[arg-type]
    if not triples and len(extrema) >= 3:
        # fall back: take last three extrema with intervening troughs
        if len(piv) >= 5:
            window5 = piv[-5:]
            kinds = [q.kind for q in window5]
            if kinds[0] == kinds[2] == kinds[4] == want:
                triples.append(tuple(window5))  # type: ignore[arg-type]

    for ls, t1, head, t2, rs in reversed(triples):
        sym_tol = p["shoulder_symmetry_tol"]
        head_ref = head.price
        if p["inverse"]:
            # inverted: head is the lowest; shoulders above it
            prominence = (min(ls.price, rs.price) - head_ref) / abs(head_ref)
            symmetry = abs(ls.price - rs.price) / abs(head_ref)
        else:
            prominence = (head_ref - max(ls.price, rs.price)) / abs(head_ref)
            symmetry = abs(ls.price - rs.price) / abs(head_ref)
        if prominence < 0.0:
            continue
        slope, intercept, rmse = fit_line([(t1.index, t1.price), (t2.index, t2.price)])
        norm_rmse = rmse / abs(head_ref)
        s_sym = soft_pass(symmetry, 0.0, sym_tol)
        s_prom = soft_pass(prominence, p["head_prominence_min"], p["head_prominence_min"] * 4)
        s_neck = soft_pass(norm_rmse, 0.0, p["neckline_fit_max_rmse"])
        # spacing balance between shoulders/head
        d1, d2 = head.index - ls.index, rs.index - head.index
        s_space = soft_pass(abs(d1 - d2) / max(d1, d2, 1), 0.0, 0.8)
        score = 0.35 * s_sym + 0.30 * s_prom + 0.20 * s_neck + 0.15 * s_space
        if best is None or score > best["score"]:
            best = {
                "score": score, "ls": ls, "t1": t1, "head": head, "t2": t2, "rs": rs,
                "slope": slope, "intercept": intercept, "prominence": prominence,
            }

    if best is None:
        res.violations.append("no LS-HEAD-RS pivot structure found")
        return res

    neckline_at = lambda idx: best["slope"] * idx + best["intercept"]  # noqa: E731
    last_close = float(df["close"].iloc[-1])
    neck_now = neckline_at(len(df) - 1)

    stages: list[str] = []
    trend_before = classify_trend(df.iloc[: max(best["ls"].index, 5)])
    if (trend_before == "up") != p["inverse"] and trend_before != "sideways":
        stages.append("preceding_uptrend" if not p["inverse"] else "preceding_downtrend")
    stages += ["left_shoulder", "head", "right_shoulder"]
    broke = last_close < neck_now if not p["inverse"] else last_close > neck_now
    if broke:
        stages.append("neckline_break")

    res.geometry_score = clamp01(best["score"])
    res.matched = res.geometry_score >= 0.5
    if res.geometry_score < 0.5:
        res.violations.append("structure too distorted (symmetry/prominence/neckline)")
    res.keypoints = {
        "left_shoulder": _kp(best["ls"]), "head": _kp(best["head"]),
        "right_shoulder": _kp(best["rs"]),
        "neckline_left": _kp(best["t1"]), "neckline_right": _kp(best["t2"]),
    }
    res.lines = {"neckline": {"slope": best["slope"], "intercept": best["intercept"], "level": float(neck_now)}}
    res.levels = {
        "neckline": float(neck_now),
        "head": float(best["head"].price),
        "breakout": float(neck_now),  # breakout = neckline break
    }
    # classical measured move: head-to-neckline distance projected from neckline
    head_dist = abs(best["head"].price - neck_now)
    res.levels["target"] = float(neck_now - head_dist) if not p["inverse"] else float(neck_now + head_dist)
    res.span = (best["ls"].index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"trend_before": trend_before, "prominence": best["prominence"], "broke_neckline": bool(broke)}
    return res


# ── Double / Triple Top & Bottom ─────────────────────────────────────────────

def validate_multiples(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"kind": "top", "n_peaks": 2, "peak_equality_tol": 0.04, "min_peak_separation": 8} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    top = p["kind"] == "top"

    piv = _pivots(df)
    want = "high" if top else "low"
    extrema = [q for q in piv if q.kind == want]
    inters = [q for q in piv if q.kind != want]
    n = int(p["n_peaks"])
    if len(extrema) < n or len(inters) < n - 1:
        res.violations.append(f"need >= {n} {p['kind']} pivots with {n-1} intervening troughs")
        return res

    peaks = extrema[-n:]
    # most recent n extrema with the intervening pivots
    last_idx = peaks[-1].index
    inter = [q for q in inters if peaks[0].index < q.index < last_idx]
    if len(inter) < n - 1:
        res.violations.append("intervening pivots missing")
        return res

    prices = [q.price for q in peaks]
    ref = float(np.mean(prices))
    spread = (max(prices) - min(prices)) / abs(ref)
    s_eq = soft_pass(spread, 0.0, p["peak_equality_tol"])

    separations = [peaks[i + 1].index - peaks[i].index for i in range(n - 1)]
    min_sep = min(separations)
    s_sep = soft_pass(min_sep, p["min_peak_separation"], p["min_peak_separation"] * 2)

    # intervening depth (for tops, trough must sit meaningfully below the peaks)
    depths = []
    for i in range(n - 1):
        seg = [q for q in inter if peaks[i].index < q.index < peaks[i + 1].index]
        if not seg:
            continue
        extreme = min(seg, key=lambda q: q.price) if top else max(seg, key=lambda q: q.price)
        depths.append(abs(ref - extreme.price) / abs(ref))
    depth = float(np.mean(depths)) if depths else 0.0
    s_depth = soft_pass(depth, 0.01, 0.10)

    score = 0.45 * s_eq + 0.25 * s_sep + 0.30 * s_depth
    res.geometry_score = clamp01(score)
    res.matched = res.geometry_score >= 0.5
    if s_eq < 0.5:
        res.violations.append("peaks/troughs not at equivalent levels")
    if s_sep < 0.5:
        res.violations.append("peaks too close together")

    neckline_pivot = min(inter, key=lambda q: q.price) if top else max(inter, key=lambda q: q.price)
    last_close = float(df["close"].iloc[-1])
    if classify_trend(df.iloc[: max(peaks[0].index, 5)]) == ("up" if top else "down"):
        stages = ["preceding_uptrend" if top else "preceding_downtrend"]
    else:
        stages = ["trend_unclear"]
    # registry stage names: first_peak / valley / second_peak (+ third_peak)
    if n == 2:
        stages += ["first_peak" if top else "first_trough", "valley" if top else "intervening_peak",
                   "second_peak" if top else "second_trough"]
    else:
        stages += [f"peak_{i+1}" if top else f"trough_{i+1}" for i in range(n)]
    broke = last_close < neckline_pivot.price if top else last_close > neckline_pivot.price
    if broke:
        stages.append("neckline_break")

    kp = {f"peak_{i+1}": _kp(q) for i, q in enumerate(peaks)}
    kp["neckline"] = _kp(neckline_pivot)
    res.keypoints = kp
    res.levels = {"neckline": float(neckline_pivot.price), "breakout": float(neckline_pivot.price)}
    # measured move
    extreme_pivot = min(peaks, key=lambda q: q.price) if not top else max(peaks, key=lambda q: q.price)
    height = abs(ref - neckline_pivot.price)
    res.levels["target"] = float(neckline_pivot.price - height) if top else float(neckline_pivot.price + height)
    res.levels["pattern_height"] = float(height)
    res.span = (peaks[0].index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"spread": spread, "depth": depth, "broke_neckline": bool(broke), "extreme": _kp(extreme_pivot)}
    return res


# ── Cup & Handle / Rounded ───────────────────────────────────────────────────

def validate_cup(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {
        "variant": "cup_and_handle", "inverse": False,
        "rim_alignment_tol": 0.05, "handle_depth_max": 0.35,
        "roundness_fit_max_rmse": 0.035,
    } | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    inverse = bool(p["inverse"])
    with_handle = p["variant"] == "cup_and_handle"

    # candidate: significant extreme in the middle, rims (similar extremes) on both sides
    piv = _pivots(df)
    kind_rim = "low" if inverse else "high"     # rim pivots for inverted cup are lows? no:
    # for a normal cup: rims are two highs, basin is the lowest low.
    rim_kind = "high" if not inverse else "low"
    basin_kind = "low" if not inverse else "high"
    rims = [q for q in piv if q.kind == rim_kind]
    basins = [q for q in piv if q.kind == basin_kind]
    if not rims or not basins:
        res.violations.append("need rim and basin pivots")
        return res

    best = None
    for i in range(len(rims) - 1):
        for b in basins:
            if not (rims[i].index < b.index < rims[i + 1].index):
                continue
            left_r, right_r = rims[i], rims[i + 1]
            rim_ref = (left_r.price + right_r.price) / 2
            if inverse:
                depth = (b.price - rim_ref) / abs(rim_ref)
            else:
                depth = (rim_ref - b.price) / abs(rim_ref)
            if depth <= 0:
                continue
            align = abs(left_r.price - right_r.price) / abs(rim_ref)
            # parabolic fit of the cup segment closes
            seg = df.iloc[left_r.index : right_r.index + 1]
            xs = np.arange(len(seg), dtype=float)
            ys = (rim_ref - seg["close"].to_numpy()) if not inverse else (seg["close"].to_numpy() - rim_ref)
            try:
                coeffs = np.polyfit(xs, ys, 2)
                pred = np.polyval(coeffs, xs)
                rmse = float(np.sqrt(np.mean((ys - pred) ** 2))) / abs(rim_ref)
            except Exception:
                rmse = 1.0
            s_align = soft_pass(align, 0.0, p["rim_alignment_tol"])
            s_depth = soft_pass(depth, 0.05, 0.6)
            s_round = soft_pass(rmse, 0.0, p["roundness_fit_max_rmse"])
            # U-shape check: quadratic coefficient negative (opens downward for ys>0 measure)
            s_shape = 1.0 if coeffs[0] < 0 else 0.3
            score = 0.30 * s_align + 0.25 * s_depth + 0.30 * s_round + 0.15 * s_shape
            if best is None or score > best["score"]:
                best = {"score": score, "left": left_r, "right": right_r, "basin": b,
                        "rim_ref": rim_ref, "depth": depth, "rmse": rmse}

    if best is None:
        # Partial-cup fallback: a left rim + basin exist but the right rim has
        # not formed yet. Report the partial structure (not a match) so the
        # completion estimator can express it as a forming pattern.
        if rims and basins:
            left_r = rims[-1]
            after = [b for b in basins if b.index > left_r.index]
            if after:
                basin = min(after, key=lambda q: q.price) if not inverse else max(after, key=lambda q: q.price)
                rim = left_r.price
                depth = (rim - basin.price) / abs(rim) if not inverse else (basin.price - rim) / abs(rim)
                if depth > 0.05:
                    stages = ["preceding_trend", "left_rim",
                              "cup_decline" if not inverse else "cup_advance",
                              "rounded_bottom" if not inverse else "rounded_top"]
                    res.geometry_score = clamp01(0.35 + 0.3 * soft_pass(depth, 0.05, 0.5))
                    res.violations.append("right rim not yet formed (partial cup)")
                    res.keypoints = {"left_rim": _kp(left_r), "basin": _kp(basin)}
                    res.levels = {"rim": float(rim), "breakout": float(rim), "basin": float(basin.price)}
                    res.span = (left_r.index, len(df) - 1)
                    res.stages_present = stages
                    res.meta = {"partial": True, "depth": depth}
                    return res
        res.violations.append("no rounded basin between similar rims")
        return res

    last_close = float(df["close"].iloc[-1])
    rim = best["rim_ref"]

    # handle analysis (only for cup_and_handle)
    handle_stage, handle_score = "absent", 1.0
    if with_handle:
        tail = df.iloc[best["right"].index :]
        if len(tail) >= 3:
            if not inverse:
                handle_low = float(tail["low"].min())
                hd = (rim - handle_low) / max(best["depth"] * abs(rim), 1e-9)
            else:
                handle_high = float(tail["high"].max())
                hd = (handle_high - rim) / max(best["depth"] * abs(rim), 1e-9)
            handle_score = soft_pass(hd, 0.0, p["handle_depth_max"])
            handle_stage = "ok" if handle_score >= 0.5 else "too_deep"
        else:
            handle_score = 0.4
            handle_stage = "not_started"

    stages = ["preceding_trend"]
    if classify_trend(df.iloc[: max(best["left"].index, 5)]) != "sideways":
        stages[0] = "preceding_trend"
    stages += ["left_rim", "cup_decline" if not inverse else "cup_advance",
               "rounded_bottom" if not inverse else "rounded_top", "right_rim"]
    broke = last_close > rim if not inverse else last_close < rim
    if with_handle and handle_stage == "ok":
        stages.append("handle")
    if broke:
        stages.append("breakout" if not inverse else "breakdown")

    res.geometry_score = clamp01(best["score"] * (0.7 + 0.3 * handle_score) if with_handle else best["score"])
    res.matched = res.geometry_score >= 0.5
    if not res.matched:
        res.violations.append("basin not round/aligned enough")
    if with_handle and handle_stage == "too_deep":
        res.violations.append("handle deeper than allowed")
    res.keypoints = {"left_rim": _kp(best["left"]), "right_rim": _kp(best["right"]),
                     "basin": _kp(best["basin"])}
    res.levels = {"rim": float(rim), "breakout": float(rim), "basin": float(best["basin"].price)}
    depth_price = abs(rim - best["basin"].price)
    res.levels["target"] = float(rim + depth_price) if not inverse else float(rim - depth_price)
    res.span = (best["left"].index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"depth": best["depth"], "roundness_rmse": best["rmse"], "handle_stage": handle_stage,
                "broke_rim": bool(broke)}
    return res


# ── Triangles / Wedges (trendline patterns) ─────────────────────────────────

def validate_trendline_pattern(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"upper": "falling", "lower": "rising", "convergence": True,
         "slope_tol": 0.0015, "touch_min": 2, "flat_tol": 0.01} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)

    piv = _pivots(df)
    highs = [q for q in piv if q.kind == "high"][-4:]
    lows = [q for q in piv if q.kind == "low"][-4:]
    if len(highs) < p["touch_min"] or len(lows) < p["touch_min"]:
        res.violations.append("not enough pivot touches for both boundaries")
        return res

    n = len(df)
    up_slope, up_int, up_rmse = fit_line([(q.index, q.price) for q in highs])
    lo_slope, lo_int, lo_rmse = fit_line([(q.index, q.price) for q in lows])
    ref = float(df["close"].mean())

    def classify(slope: float) -> str:
        s = slope * n / ref  # normalized total drift across window
        if abs(s) < p["flat_tol"]:
            return "flat"
        return "rising" if s > 0 else "falling"

    upper_cls, lower_cls = classify(up_slope), classify(lo_slope)
    ok_classes = {upper_cls, lower_cls} == {p["upper"], p["lower"]} or (
        upper_cls == p["upper"] and lower_cls == p["lower"]
    )
    if not ok_classes:
        res.violations.append(f"boundaries are {upper_cls}/{lower_cls}, expected {p['upper']}/{p['lower']}")
        return res

    range_start = max(highs[0].index, lows[0].index)
    up_at = lambda i: up_slope * i + up_int  # noqa: E731
    lo_at = lambda i: lo_slope * i + lo_int  # noqa: E731
    w0 = up_at(range_start) - lo_at(range_start)
    w1 = up_at(n - 1) - lo_at(n - 1)
    conv_ratio = w1 / w0 if w0 > 0 else 1.0
    if p["convergence"]:
        s_conv = soft_pass(conv_ratio, 0.0, 0.85)
    else:
        s_conv = soft_pass(conv_ratio, 1.15, 5.0)
    if w1 <= 0:
        res.violations.append("boundaries crossed (apex passed)")
        return res

    s_fit = soft_pass((up_rmse + lo_rmse) / (2 * ref), 0.0, 0.02)
    # containment: fraction of bars whose close sits between the two lines at that bar
    closes = df["close"].to_numpy()
    between = 0.0
    for i in range(range_start, n):
        u, l = up_at(i), lo_at(i)
        if l - 1e-9 <= closes[i] <= u + 1e-9 + 3 * max(up_rmse, lo_rmse):
            between += 1
    s_contain = between / max(n - range_start, 1)

    score = 0.35 * s_conv + 0.30 * s_fit + 0.35 * s_contain
    res.geometry_score = clamp01(score)
    res.matched = res.geometry_score >= 0.5
    if s_conv < 0.4:
        res.violations.append("boundaries do not converge as required")

    last_close = float(df["close"].iloc[-1])
    up_now, lo_now = up_at(n - 1), lo_at(n - 1)
    # breakout for the bullish variant = close above upper; bearish = below lower; neutral either
    direction = {"rising": "bearish", "falling": "bullish", "flat": "bullish"}.get(p["upper"], "neutral")
    stages = ["boundary_touches", "range_established"]
    if conv_ratio < 0.6:
        stages.append("contraction")
    stages.append("apex_approach" if conv_ratio < 0.35 else "consolidation")
    broke_up, broke_down = last_close > up_now, last_close < lo_now
    if broke_up or broke_down:
        stages.append("breakout" if broke_up else "breakdown")

    res.keypoints = {f"upper_{i+1}": _kp(q) for i, q in enumerate(highs)}
    res.keypoints.update({f"lower_{i+1}": _kp(q) for i, q in enumerate(lows)})
    res.lines = {
        "upper": {"slope": up_slope, "intercept": up_int, "level": float(up_now)},
        "lower": {"slope": lo_slope, "intercept": lo_int, "level": float(lo_now)},
    }
    res.levels = {"upper": float(up_now), "lower": float(lo_now),
                  "breakout": float(up_now if direction == "bullish" else lo_now)}
    width = abs(up_now - lo_now) or ref * 0.02
    if direction == "bullish":
        res.levels["target"] = float(up_now + w0)   # measured move: widest part
    else:
        res.levels["target"] = float(lo_now - w0)
    res.span = (range_start, n - 1)
    res.stages_present = stages
    res.meta = {"upper_class": upper_cls, "lower_class": lower_cls,
                "convergence_ratio": conv_ratio, "broke_up": bool(broke_up), "broke_down": bool(broke_down),
                "contained_frac": s_contain}
    return res


# ── Flags / Pennants ─────────────────────────────────────────────────────────

def validate_flag(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"direction": "bullish", "pennant": False, "pole_min_bars": 3, "pole_min_slope": 0.02,
         "retrace_max": 0.5, "flag_slope_tol": 0.002} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    bullish = p["direction"] == "bullish"

    n = len(df)
    closes = df["close"].to_numpy(dtype=float)
    best_pole = None
    # choose split point: pole = segment before the final consolidation
    for split in range(max(p["pole_min_bars"] + 3, 10), n - 2):
        pole_seg = closes[: n - split]
        flag_seg = closes[n - split :]
        pole_slope = (pole_seg[-1] - pole_seg[0]) / abs(pole_seg[0]) if len(pole_seg) else 0.0
        if bullish and pole_slope < p["pole_min_slope"]:
            continue
        if not bullish and pole_slope > -p["pole_min_slope"]:
            continue
        pole_move = pole_seg[-1] - pole_seg[0]
        retrace = (flag_seg[0] - flag_seg.min()) / abs(pole_move) if (bullish and pole_move != 0) else (
            (flag_seg.max() - flag_seg[0]) / abs(pole_move) if (not bullish and pole_move != 0) else 1.0
        )
        if retrace > p["retrace_max"]:
            continue
        # flag tightness: range vs pole move
        flag_range = flag_seg.max() - flag_seg.min()
        tight = flag_range / max(abs(pole_move), 1e-9)
        score = soft_pass(tight, 0.05, 0.6)
        if p["pennant"]:
            # converging flag
            fhigh = df["high"].to_numpy()[n - split :]
            flow = df["low"].to_numpy()[n - split :]
            xs = np.arange(len(fhigh))
            hi_slope = np.polyfit(xs, fhigh, 1)[0] if len(xs) > 2 else 0.0
            lo_slope = np.polyfit(xs, flow, 1)[0] if len(xs) > 2 else 0.0
            converging = (hi_slope < 0 < lo_slope) if bullish else (hi_slope > 0 > lo_slope)
            if converging:
                score = min(1.0, score + 0.15)
            else:
                score *= 0.75
        if best_pole is None or score > best_pole[0]:
            best_pole = (score, n - split, float(pole_move), retrace, tight)

    if best_pole is None:
        res.violations.append("no impulse pole followed by tight consolidation")
        return res

    score, split, pole_move, retrace, tight = best_pole
    s_pole = soft_pass(abs(pole_move) / abs(closes[0]) if closes[0] else 0.0, p["pole_min_slope"], p["pole_min_slope"] * 3)
    s_retr = soft_pass(retrace, 0.0, p["retrace_max"])
    total = 0.5 * score + 0.3 * s_pole + 0.2 * s_retr
    res.geometry_score = clamp01(total)
    res.matched = res.geometry_score >= 0.5

    flag_df = df.iloc[split:]
    flag_hi = float(flag_df["high"].max())
    flag_lo = float(flag_df["low"].min())
    last_close = float(df["close"].iloc[-1])
    stages = ["impulse_pole"]
    if len(flag_df) >= 3:
        stages.append("consolidation_flag")
    broke = last_close > flag_hi - (flag_hi - flag_lo) * 0.05 if bullish else last_close < flag_lo + (flag_hi - flag_lo) * 0.05
    if broke:
        stages.append("breakout" if bullish else "breakdown")

    pole_start_price = float(df["open"].iloc[0])
    res.keypoints = {
        "pole_start": {"index": 0, "price": pole_start_price},
        "pole_end": {"index": int(split), "price": float(closes[split])},
        "flag_high": {"index": int(split + int(np.argmax(flag_df["high"].to_numpy()))), "price": flag_hi},
        "flag_low": {"index": int(split + int(np.argmin(flag_df["low"].to_numpy()))), "price": flag_lo},
    }
    res.levels = {
        "flag_high": flag_hi, "flag_low": flag_lo,
        "breakout": flag_hi if bullish else flag_lo,
    }
    # measured move: flag breakout + pole length
    res.levels["target"] = float(flag_hi + abs(pole_move)) if bullish else float(flag_lo - abs(pole_move))
    res.levels["pole_length"] = float(abs(pole_move))
    res.span = (0, n - 1)
    res.stages_present = stages
    res.meta = {"pole_move": pole_move, "retrace": retrace, "tightness": tight, "split": split,
                "broke_flag": bool(broke)}
    return res


# ── Rectangle ────────────────────────────────────────────────────────────────

def validate_rectangle(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"flat_tol": 0.015, "touch_min": 2, "min_height_atr": 1.0} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)

    piv = _pivots(df)
    highs = [q for q in piv if q.kind == "high"][-5:]
    lows = [q for q in piv if q.kind == "low"][-5:]
    if len(highs) < p["touch_min"] or len(lows) < p["touch_min"]:
        res.violations.append("not enough touches on both rails")
        return res

    resistance = float(np.mean([q.price for q in highs]))
    support = float(np.mean([q.price for q in lows]))
    ref = (resistance + support) / 2
    height = resistance - support
    a = atr(df)
    if height <= 0 or (a > 0 and height < p["min_height_atr"] * a):
        res.violations.append("range too narrow relative to volatility")
        return res

    hi_spread = max(q.price for q in highs) - min(q.price for q in highs)
    lo_spread = max(q.price for q in lows) - min(q.price for q in lows)
    s_flat = soft_pass(max(hi_spread, lo_spread) / ref, 0.0, p["flat_tol"])
    closes = df["close"].to_numpy()
    n = len(df)
    start = max(highs[0].index, lows[0].index)
    inside = float(np.mean((closes[start:] >= support - height * 0.05) & (closes[start:] <= resistance + height * 0.05)))
    s_inside = clamp01(inside * 1.1)
    score = 0.5 * s_flat + 0.5 * s_inside
    res.geometry_score = clamp01(score)
    res.matched = res.geometry_score >= 0.5
    if s_flat < 0.4:
        res.violations.append("rails not horizontal")

    last_close = float(df["close"].iloc[-1])
    broke_up = last_close > resistance
    broke_down = last_close < support
    stages = ["range_highs", "range_lows"]
    if len(highs) + len(lows) >= 5:
        stages.append("oscillation")
    if broke_up:
        stages.append("breakout")
    elif broke_down:
        stages.append("breakdown")

    res.keypoints = {f"high_{i+1}": _kp(q) for i, q in enumerate(highs)}
    res.keypoints.update({f"low_{i+1}": _kp(q) for i, q in enumerate(lows)})
    res.lines = {
        "resistance": {"slope": 0.0, "intercept": resistance, "level": resistance},
        "support": {"slope": 0.0, "intercept": support, "level": support},
    }
    res.levels = {"resistance": resistance, "support": support,
                  "breakout": resistance if broke_up else support,
                  "target": resistance + height if broke_up else support - height,
                  "pattern_height": height}
    res.span = (start, n - 1)
    res.stages_present = stages
    res.meta = {"broke_up": bool(broke_up), "broke_down": bool(broke_down)}
    return res


# ── Channel ──────────────────────────────────────────────────────────────────

def validate_channel(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"slope_sign": 1, "parallel_tol": 0.15, "touch_min": 2, "slope_tol": 0.0012} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)

    piv = _pivots(df)
    highs = [q for q in piv if q.kind == "high"][-4:]
    lows = [q for q in piv if q.kind == "low"][-4:]
    if len(highs) < p["touch_min"] or len(lows) < p["touch_min"]:
        res.violations.append("not enough rail touches")
        return res
    n = len(df)
    ref = float(df["close"].mean())
    up_slope, up_int, up_rmse = fit_line([(q.index, q.price) for q in highs])
    lo_slope, lo_int, lo_rmse = fit_line([(q.index, q.price) for q in lows])
    norm_drift = lambda s: s * n / ref  # noqa: E731
    s_slope = soft_pass(norm_drift(up_slope) * p["slope_sign"], p["slope_tol"], p["slope_tol"] * 8)
    s_parallel = soft_pass(abs(norm_drift(up_slope - lo_slope)), 0.0, p["parallel_tol"])
    s_fit = soft_pass((up_rmse + lo_rmse) / (2 * ref), 0.0, 0.02)
    up_at = lambda i: up_slope * i + up_int  # noqa: E731
    lo_at = lambda i: lo_slope * i + lo_int  # noqa: E731
    start = max(highs[0].index, lows[0].index)
    closes = df["close"].to_numpy()
    between = np.mean([lo_at(i) - 1e-9 - 2 * lo_rmse <= closes[i] <= up_at(i) + 1e-9 + 2 * up_rmse for i in range(start, n)])
    s_between = clamp01(float(between))
    score = 0.25 * s_slope + 0.25 * s_parallel + 0.2 * s_fit + 0.3 * s_between
    res.geometry_score = clamp01(score)
    res.matched = res.geometry_score >= 0.5
    if s_parallel < 0.4:
        res.violations.append("rails not parallel")
    if s_slope < 0.4:
        res.violations.append("channel slope does not match direction")

    last_close = float(df["close"].iloc[-1])
    up_now, lo_now = up_at(n - 1), lo_at(n - 1)
    broke_lower = last_close < lo_now
    stages = ["lower_rail", "upper_rail", "parallel_advance"]
    if broke_lower:
        stages.append("rail_break")
    res.keypoints = {f"high_{i+1}": _kp(q) for i, q in enumerate(highs)}
    res.keypoints.update({f"low_{i+1}": _kp(q) for i, q in enumerate(lows)})
    res.lines = {"upper": {"slope": up_slope, "intercept": up_int, "level": float(up_now)},
                 "lower": {"slope": lo_slope, "intercept": lo_int, "level": float(lo_now)}}
    res.levels = {"upper": float(up_now), "lower": float(lo_now), "breakout": float(lo_now if broke_lower else up_now)}
    res.span = (start, n - 1)
    res.stages_present = stages
    res.meta = {"broke_lower": bool(broke_lower)}
    return res


# ── Broadening ───────────────────────────────────────────────────────────────

def validate_broadening(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"divergence_min": 0.01, "touch_min": 2} | params
    # diverging boundaries: reuse trendline code with convergence=false, opposite classes
    merged = {"upper": "rising", "lower": "falling", "convergence": False}
    return validate_trendline_pattern(df, merged | params, pattern_id)


# ── Diamond ──────────────────────────────────────────────────────────────────

def validate_diamond(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"position": "top"} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)

    piv = _pivots(df)
    if len(piv) < 4:
        res.violations.append("insufficient pivots for diamond")
        return res
    seq = piv[-6:]
    # envelope width at quartiles: should expand then contract
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    start = seq[0].index
    span = max(n - start, 4)
    q1, q2, q3 = start + span // 4, start + span // 2, start + 3 * span // 4
    w = lambda i: float(np.max(highs[start : i + 1]) - np.min(lows[start : i + 1]))  # noqa: E731
    try:
        w1, w2, w3 = w(q1), w(q2), w(q3)
    except ValueError:
        res.violations.append("window too short")
        return res
    expand = w2 > w1
    contract = w3 <= w2
    s_exp = soft_pass((w2 - w1) / max(w1, 1e-9), 0.0, 0.3)
    s_con = soft_pass((w2 - w3) / max(w2, 1e-9), 0.0, 0.3)
    score = 0.5 * s_exp + 0.5 * s_con if (expand and contract) else 0.0
    res.geometry_score = clamp01(score)
    res.matched = res.geometry_score >= 0.5
    if not (expand and contract):
        res.violations.append("envelope does not expand then contract")
    top = p["position"] == "top"
    last_close = float(df["close"].iloc[-1])
    res.levels = {
        "breakout": float(np.min(lows[q2:])) if top else float(np.max(highs[q2:])),
        "apex_low": float(np.min(lows[start:])),
        "apex_high": float(np.max(highs[start:])),
    }
    res.stages_present = ["broadening_phase", "narrowing_phase"]
    if (top and last_close < res.levels["breakout"]) or (not top and last_close > res.levels["breakout"]):
        res.stages_present.append("breakdown" if top else "breakout")
    res.span = (start, n - 1)
    res.keypoints = {f"pivot_{i+1}": _kp(q) for i, q in enumerate(seq)}
    return res


CLASSICAL_VALIDATORS = {
    "head_and_shoulders": validate_head_and_shoulders,
    "multiples": validate_multiples,
    "cup": validate_cup,
    "trendline_pattern": validate_trendline_pattern,
    "flag": validate_flag,
    "rectangle": validate_rectangle,
    "channel": validate_channel,
    "broadening": validate_broadening,
    "diamond": validate_diamond,
}
