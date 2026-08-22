"""Harmonic pattern validators (XABCD Fibonacci structures).

Direction is an attribute resolved from structure orientation (X->A up =
bearish completion at D for retracement patterns), not a separate class.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..common import ValidationResult, clamp01, soft_pass
from ..pivots import alternate_pivots, find_pivots, Pivot


def _pivots(df: pd.DataFrame) -> list[Pivot]:
    return alternate_pivots(find_pivots(df, left=3, right=3))


def _kp(p: Pivot) -> dict:
    return {"index": int(p.index), "price": float(p.price)}


def _score_window(ratios: list[tuple[float, float, float]]) -> float:
    """Mean soft membership over (value, lo, hi) ratio constraints."""
    if not ratios:
        return 0.0
    return float(np.mean([soft_pass(v, lo, hi) for v, lo, hi in ratios]))


def validate_harmonic(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    variant = params.get("variant", "gartley")
    dispatch = {
        "gartley": _gartley_family, "bat": _gartley_family, "butterfly": _gartley_family,
        "crab": _gartley_family, "deep_crab": _gartley_family,
        "shark": _shark, "cypher": _cypher, "abcd": _abcd,
    }
    handler = dispatch.get(variant, _gartley_family)
    return handler(df, params, pattern_id)


def _orientation(x: float, a: float) -> str:
    return "up" if a > x else "down"


def _gartley_family(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {
        "xab_min": 0.618, "xab_max": 0.65, "ad_min": 0.786, "ad_max": 0.82,
        "bc_projection": [1.13, 1.618], "xd_max": 1.0,
    } | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)

    piv = _pivots(df)
    best = None
    # search alternating X-A-B-C-D quintuples among recent pivots
    quints: list[tuple[Pivot, Pivot, Pivot, Pivot, Pivot]] = []
    kinds_ok = {"high", "low"}
    for i in range(max(0, len(piv) - 12), len(piv) - 4):
        q5 = piv[i : i + 5]
        kinds = [q.kind for q in q5]
        if q5[0].kind != q5[2].kind or q5[1].kind != q5[3].kind or q5[2].kind != q5[4].kind:
            continue
        if kinds[0] in kinds_ok:
            quints.append(tuple(q5))

    for X, A, B, C, D in quints:
        xa = abs(A.price - X.price)
        if xa <= 0:
            continue
        xab = abs(A.price - B.price) / xa   # retracement depth of B into XA
        ad = abs(D.price - A.price) / xa
        # classic semantics: C retraces AB (0.382–0.886); the registry's
        # bc_projection window [1.13, 1.618] is the CD extension of BC.
        bc_retrace = abs(C.price - B.price) / max(abs(A.price - B.price), 1e-9)
        cd_of_bc = abs(D.price - C.price) / max(abs(C.price - B.price), 1e-9)
        xd = abs(D.price - X.price) / xa

        lo_bc, hi_bc = p["bc_projection"]
        pad = 0.08
        score = _score_window([
            (xab, p["xab_min"] - pad, p["xab_max"] + pad),
            (ad, p["ad_min"] - pad, p["ad_max"] + pad),
            (bc_retrace, 0.382 - pad, 0.886 + pad),
            (cd_of_bc, max(lo_bc - pad, 0.0), hi_bc + pad),
            (xd, 0.0, p["xd_max"] + pad),
        ])
        strict = (
            p["xab_min"] - 0.02 <= xab <= p["xab_max"] + 0.02
            and p["ad_min"] - 0.02 <= ad <= p["ad_max"] + 0.02
            and 0.382 - 0.05 <= bc_retrace <= 0.886 + 0.05
            and lo_bc * 0.8 <= cd_of_bc <= hi_bc * 1.25
            and xd <= p["xd_max"] + 0.02
        )
        if best is None or score > best[0]:
            best = (score, strict, (X, A, B, C, D), xab, ad, cd_of_bc, xd)

    if best is None:
        res.violations.append("no XABCD alternating pivot structure found")
        return res

    score, strict, (X, A, B, C, D), xab, ad, bc_proj, xd = best
    res.geometry_score = clamp01(score)
    res.matched = strict and score >= 0.55
    if not res.matched:
        res.violations.append("Fibonacci ratios outside pattern tolerance")

    orient = _orientation(X.price, A.price)
    # bullish completion: XA down then D low (retracement patterns), or extension beyond X low
    bullish = orient == "down"
    if p.get("ad_min", 0) > 1.0:      # extension patterns invert the rule
        bullish = not bullish

    stages = ["xa_leg", "b_retracement", "bc_projection"]
    if res.matched or score >= 0.4:
        stages.append("d_completion_prz")
    last_close = float(df["close"].iloc[-1])
    reversal_started = (last_close > D.price) if bullish else (last_close < D.price)
    if reversal_started:
        stages.append("reversal")

    res.keypoints = {"X": _kp(X), "A": _kp(A), "B": _kp(B), "C": _kp(C), "D": _kp(D)}
    res.levels = {
        "D": float(D.price), "X": float(X.price), "A": float(A.price),
        "prz_center": float(A.price - (A.price - X.price) * (p["ad_min"] + p["ad_max"]) / 2 * (1 if orient == "down" else -1)),
        "invalidation": float(X.price - xa * 0.02) if bullish else float(X.price + xa * 0.02),
    }
    res.levels["target"] = float(B.price) if bullish else float(B.price)
    res.span = (X.index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"xab": xab, "ad": ad, "bc_projection": bc_proj, "xd": xd,
                "orientation": orient, "bullish": bullish,
                "reversal_started": bool(reversal_started)}
    return res


def _shark(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"variant": "shark", "xb_min": 1.13, "xb_max": 1.618,
         "xc_min": 0.886, "xc_max": 1.13} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    piv = _pivots(df)

    best = None
    for i in range(max(0, len(piv) - 10), len(piv) - 3):
        O, X, B, C = piv[i : i + 4]
        if O.kind != B.kind or X.kind != C.kind:
            continue
        ox = abs(X.price - O.price)
        if ox <= 0:
            continue
        xb = abs(B.price - O.price) / ox      # B beyond X
        xc = abs(C.price - O.price) / ox      # C in OX-projected zone
        score = _score_window([
            (xb, p["xb_min"] - 0.1, p["xb_max"] + 0.1),
            (xc, p["xc_min"] - 0.05, p["xc_max"] + 0.05),
        ])
        strict = (p["xb_min"] - 0.02 <= xb <= p["xb_max"] + 0.02
                  and p["xc_min"] - 0.02 <= xc <= p["xc_max"] + 0.02)
        if best is None or score > best[0]:
            best = (score, strict, (O, X, B, C), xb, xc)

    if best is None:
        res.violations.append("no O-X-B-C shark structure found")
        return res
    score, strict, (O, X, B, C), xb, xc = best
    res.geometry_score = clamp01(score)
    res.matched = strict and score >= 0.55
    if not res.matched:
        res.violations.append("shark ratio windows not satisfied")

    bullish = X.price < O.price  # initial leg down => bullish reversal at C
    stages = ["ox_leg", "b_extension"]
    if score >= 0.4:
        stages.append("c_completion_prz")
    res.keypoints = {"O": _kp(O), "X": _kp(X), "B": _kp(B), "C": _kp(C)}
    res.levels = {"C": float(C.price), "invalidation": float(O.price)}
    res.span = (O.index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"xb": xb, "xc": xc, "bullish": bullish}
    return res


def _cypher(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"variant": "cypher", "xab_min": 0.382, "xab_max": 0.618,
         "xc_xa_min": 1.272, "xc_xa_max": 1.414, "d_of_xc": 0.786} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    piv = _pivots(df)

    best = None
    for i in range(max(0, len(piv) - 12), len(piv) - 4):
        X, A, B, C, D = piv[i : i + 5]
        xa = abs(A.price - X.price)
        if xa <= 0:
            continue
        xab = abs(A.price - B.price) / xa
        # C extends beyond A
        xc_xa = (abs(C.price - X.price)) / xa
        d_of_xc = abs(C.price - D.price) / max(abs(C.price - X.price), 1e-9)
        score = _score_window([
            (xab, p["xab_min"] - 0.08, p["xab_max"] + 0.08),
            (xc_xa, p["xc_xa_min"] - 0.08, p["xc_xa_max"] + 0.08),
            (d_of_xc, p["d_of_xc"] - 0.08, p["d_of_xc"] + 0.08),
        ])
        c_beyond_a = (C.price > A.price) if A.price > X.price else (C.price < A.price)
        strict = (p["xab_min"] - 0.02 <= xab <= p["xab_max"] + 0.02
                  and p["xc_xa_min"] - 0.02 <= xc_xa <= p["xc_xa_max"] + 0.02
                  and abs(d_of_xc - p["d_of_xc"]) <= 0.05 and c_beyond_a)
        if best is None or score > best[0]:
            best = (score, strict, (X, A, B, C, D), xab, xc_xa, d_of_xc)

    if best is None:
        res.violations.append("no cypher XABCD structure found")
        return res
    score, strict, (X, A, B, C, D), xab, xc_xa, d_of_xc = best
    res.geometry_score = clamp01(score)
    res.matched = strict and score >= 0.55
    if not res.matched:
        res.violations.append("cypher ratio windows not satisfied")

    bullish = A.price < X.price
    stages = ["xa_leg", "b_retracement", "c_extension"]
    if score >= 0.4:
        stages.append("d_completion_prz")
    res.keypoints = {"X": _kp(X), "A": _kp(A), "B": _kp(B), "C": _kp(C), "D": _kp(D)}
    res.levels = {"D": float(D.price), "C": float(C.price), "invalidation": float(X.price)}
    res.span = (X.index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"xab": xab, "xc_xa": xc_xa, "d_of_xc": d_of_xc, "bullish": bullish}
    return res


def _abcd(df: pd.DataFrame, params: dict, pattern_id: str) -> ValidationResult:
    p = {"variant": "abcd", "bc_min": 0.618, "bc_max": 0.786, "cd_tolerance": 0.10} | params
    res = ValidationResult(pattern_id=pattern_id, matched=False, geometry_score=0.0)
    piv = _pivots(df)

    best = None
    for i in range(max(0, len(piv) - 10), len(piv) - 3):
        A, B, C, D = piv[i : i + 4]
        if A.kind != C.kind:
            continue
        ab = abs(B.price - A.price)
        if ab <= 0:
            continue
        bc = abs(C.price - B.price) / ab
        cd = abs(D.price - C.price)
        rel = abs(cd - ab) / ab
        score = _score_window([
            (bc, p["bc_min"] - 0.1, p["bc_max"] + 0.1),
            (rel, 0.0, p["cd_tolerance"] + 0.1),
        ])
        strict = (p["bc_min"] - 0.05 <= bc <= p["bc_max"] + 0.05 and rel <= p["cd_tolerance"] * 1.3)
        # time symmetry bonus
        t1, t2 = B.index - A.index, D.index - C.index
        t_sym = soft_pass(abs(t1 - t2) / max(t1, t2, 1), 0.0, 0.8)
        score = 0.8 * score + 0.2 * t_sym
        if best is None or score > best[0]:
            best = (score, strict, (A, B, C, D), bc, rel, t_sym)

    if best is None:
        res.violations.append("no AB=CD pivot structure found")
        return res
    score, strict, (A, B, C, D), bc, rel, t_sym = best
    res.geometry_score = clamp01(score)
    res.matched = strict and score >= 0.55
    if not res.matched:
        res.violations.append("CD length not within tolerance of AB, or BC retrace off")

    bullish = B.price < A.price
    stages = ["ab_leg", "bc_retracement", "cd_leg_completion"]
    res.keypoints = {"A": _kp(A), "B": _kp(B), "C": _kp(C), "D": _kp(D)}
    res.levels = {"D": float(D.price), "invalidation": float(D.price - abs(D.price - C.price) * 0.1 if bullish else D.price + abs(D.price - C.price) * 0.1)}
    res.span = (A.index, len(df) - 1)
    res.stages_present = stages
    res.meta = {"bc": bc, "cd_vs_ab": rel, "time_symmetry": t_sym, "bullish": bullish}
    return res
