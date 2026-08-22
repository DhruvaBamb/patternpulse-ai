"""Adapters for optional future data providers.

Each becomes functional once its credentials exist in the environment.
They intentionally fail with actionable messages rather than returning
fake data.
"""

from __future__ import annotations

import os

from .base import UnconfiguredProvider


def _configured(env_keys: list[str]) -> bool:
    return all(os.environ.get(k) for k in env_keys)


def build_alpha_vantage() -> UnconfiguredProvider | None:
    if _configured(["ALPHAVANTAGE_API_KEY"]):
        return None  # real implementation pending — do not pretend
    return UnconfiguredProvider(
        "alpha_vantage", ["ALPHAVANTAGE_API_KEY"]
    )


def build_twelve_data() -> UnconfiguredProvider | None:
    if _configured(["TWELVEDATA_API_KEY"]):
        return None
    return UnconfiguredProvider("twelve_data", ["TWELVEDATA_API_KEY"])


def build_polygon() -> UnconfiguredProvider | None:
    if _configured(["POLYGON_API_KEY"]):
        return None
    return UnconfiguredProvider("polygon", ["POLYGON_API_KEY"])


def build_binance() -> UnconfiguredProvider | None:
    if _configured(["BINANCE_API_KEY"]):
        return None
    return UnconfiguredProvider("binance", ["BINANCE_API_KEY"])


def build_zerodha_kite() -> UnconfiguredProvider | None:
    if _configured(["ZERODHA_API_KEY", "ZERODHA_ACCESS_TOKEN"]):
        return None
    return UnconfiguredProvider("zerodha_kite", ["ZERODHA_API_KEY", "ZERODHA_ACCESS_TOKEN"])


FUTURE_PROVIDERS = {
    "alpha_vantage": build_alpha_vantage,
    "twelve_data": build_twelve_data,
    "polygon": build_polygon,
    "binance": build_binance,
    "zerodha_kite": build_zerodha_kite,
}
