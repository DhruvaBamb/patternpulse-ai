# Pattern Library

All 77 patterns live in `ml/registry/patterns/*.yaml` and are served through
`PatternRegistry` (content-hashed version, stable CV class ids).

| Category | Count | Validators |
|---|---|---|
| Classical chart patterns | 28 | `ml/geometry/classical/validators.py` (9 parametrised families) |
| Candlestick patterns | 41 | `ml/geometry/candlestick/rules.py` (deterministic 1–5 bar rules) |
| Harmonic patterns | 8 | `ml/geometry/harmonic/validators.py` (XABCD + Fibonacci windows) |

## Classical validator families

`head_and_shoulders` (incl. inverse), `multiples` (double/triple top/bottom),
`cup` (cup & handle / rounded, incl. partial-cup fallback), `trendline_pattern`
(triangles, wedges), `flag` (flags/pennants), `rectangle`, `channel`,
`broadening`, `diamond`. Direction variants are registry attributes, not
duplicate classes.

## Every definition carries

- identity: id, name, category, direction
- geometry: min/max bars, `validator_params` tolerances (timeframe-agnostic,
  normalised ratios)
- behaviour: confirmation rules, invalidation rules
- formation `stages` powering the completion estimator
- `supports_partial_detection` (false for candlesticks — they complete at
  the close of their final candle)
- `confused_with` pairs for hard-example mining and education

## Verification

`tests/geometry/test_known_answers.py` holds deterministic fixtures with
known outcomes: perfect/imperfect/invalid double top, valid/invalid H&S,
partial cup & handle, bullish/bearish engulfing, Gartley pass/fail ratios.
The synthetic dataset additionally self-verifies every generated positive
with the production validators.

## Timeframes

The same registry serves 1m–1wk. Geometry ratios are expressed relative to
price scale and bar counts (not absolute ticks), so tolerances transfer
across intervals; the scanner's timeframe list is configured in
`configs/app.yaml`.
