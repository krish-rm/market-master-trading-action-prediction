from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureParams:
    window_small_ma: int = 5
    window_large_ma: int = 20
    window_volatility: int = 20
    window_rsi: int = 14
    window_atr: int = 14
    window_bb: int = 20
    rsi_eps: float = 1e-9


@dataclass
class LabelParams:
    threshold_t1: float = 0.001
    threshold_t2: float = 0.003


ACTION_LABELS: List[str] = [
    "strong_sell",
    "sell",
    "hold",
    "buy",
    "strong_buy",
]


def compute_rsi_like(close: pd.Series, window: int, eps: float) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = (avg_gain + eps) / (avg_loss + eps)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def compute_bollinger_band_width(close: pd.Series, window: int) -> pd.Series:
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = (upper - lower) / (ma.replace(0, np.nan))
    return width


def build_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    df = df.copy()
    # Basic checks
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort and set index for rolling ops
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Price-based features
    df["log_return"] = np.log(df["close"]).diff()
    df["range_rel"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["body_rel"] = (df["close"] - df["open"]) / df["close"].replace(0, np.nan)

    # Moving averages and momentum
    df["ma_s"] = (
        df["close"]
        .rolling(params.window_small_ma, min_periods=params.window_small_ma)
        .mean()
    )
    df["ma_l"] = (
        df["close"]
        .rolling(params.window_large_ma, min_periods=params.window_large_ma)
        .mean()
    )
    df["ma_cross"] = (df["ma_s"] - df["ma_l"]) / df["ma_l"].replace(0, np.nan)

    # Volatility
    df["volatility"] = (
        df["close"]
        .rolling(params.window_volatility, min_periods=params.window_volatility)
        .std(ddof=0)
    )

    # RSI-like
    df["rsi"] = compute_rsi_like(df["close"], params.window_rsi, params.rsi_eps)

    # ATR
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], params.window_atr)
    df["atr_rel"] = df["atr"] / df["close"].replace(0, np.nan)

    # Bollinger Band width
    df["bb_width"] = compute_bollinger_band_width(df["close"], params.window_bb)

    return df


def build_labels(df: pd.DataFrame, label_params: LabelParams) -> pd.DataFrame:
    df = df.copy()
    # Forward return for next bar
    df["r_next"] = df["close"].shift(-1).sub(df["close"]).div(df["close"])
    t1, t2 = label_params.threshold_t1, label_params.threshold_t2
    bins = [-np.inf, -t2, -t1, t1, t2, np.inf]
    df["action"] = pd.cut(
        df["r_next"],
        bins=bins,
        labels=ACTION_LABELS,
        right=True,
        include_lowest=True
    )
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    return df


def build_dataset(
    raw_df: pd.DataFrame, feature_params: FeatureParams, label_params: LabelParams
) -> Tuple[pd.DataFrame, List[str]]:
    """Return feature frame with labels and the list of feature column names.

    Drops initial warm-up rows where rolling features are NaN and the last row
    with no label.
    """
    df = build_features(raw_df, feature_params)
    # Determine warm-up based on the largest window
    warmup = max(
        feature_params.window_large_ma,
        feature_params.window_volatility,
        feature_params.window_rsi,
        feature_params.window_atr,
        feature_params.window_bb,
    )
    df = df.iloc[warmup:].reset_index(drop=True)
    df = build_labels(df, label_params)

    feature_columns = [
        "log_return",
        "range_rel",
        "body_rel",
        "ma_s",
        "ma_l",
        "ma_cross",
        "volatility",
        "rsi",
        "atr_rel",
        "bb_width",
        "volume",
    ]
    # Ensure no NaNs remain in features
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df, feature_columns
