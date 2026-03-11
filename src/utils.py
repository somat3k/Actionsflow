"""
Quantum Trading System – Utility Functions
Shared helpers used across all modules.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%SZ"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ── Time utilities ─────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_now_ms() -> int:
    """Current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def parse_snapshot_end_ms(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def interval_to_seconds(interval: str) -> int:
    """Convert interval string (e.g. '15m', '1h', '4h') to seconds."""
    unit = interval[-1]
    value = int(interval[:-1])
    mapping = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return value * mapping.get(unit, 60)


def interval_to_ms(interval: str) -> int:
    return interval_to_seconds(interval) * 1000


# ── Data utilities ─────────────────────────────────────────────────────────────

def candles_to_dataframe(candles: List[Dict]) -> pd.DataFrame:
    """Convert raw Hyperliquid candle list to a pandas DataFrame."""
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    # Hyperliquid candle fields: t (open time ms), T (close time ms),
    # o (open), h (high), l (low), c (close), v (volume), n (num trades)
    rename = {
        "t": "open_time", "T": "close_time",
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "n": "num_trades",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open_time" in df.columns:
        df["open_time"] = pd.to_numeric(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("datetime")
    return df


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ── Technical indicators ───────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    tr = compute_atr(df, period=1)
    tr_smooth = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / tr_smooth.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx.fillna(0.0)


def compute_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k.fillna(50.0)
    d = k.rolling(window=d_period).mean()
    return k, d.fillna(50.0)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"]
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return (cum_tp_vol / cum_vol.replace(0, np.nan)).fillna(typical_price)


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to a OHLCV DataFrame."""
    df = df.copy()
    close = df["close"]

    df["rsi_14"] = compute_rsi(close, 14)
    df["rsi_7"] = compute_rsi(close, 7)

    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)

    for span in [9, 21, 50, 200]:
        df[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

    df["bb_upper"], df["bb_middle"], df["bb_lower"] = compute_bollinger_bands(close)
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)

    df["atr_14"] = compute_atr(df, 14)
    df["adx_14"] = compute_adx(df, 14)
    df["stochastic_k"], df["stochastic_d"] = compute_stochastic(df)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["cci_20"] = (typical - typical.rolling(20).mean()) / (
        0.015 * typical.rolling(20).std().replace(0, np.nan)
    )

    df["williams_r"] = (
        -100
        * (df["high"].rolling(14).max() - close)
        / (df["high"].rolling(14).max() - df["low"].rolling(14).min()).replace(0, np.nan)
    )

    df["vwap"] = compute_vwap(df)
    df["obv"] = compute_obv(df)

    # Ichimoku components
    df["ichimoku_tenkan"] = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    df["ichimoku_kijun"] = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    df["ichimoku_senkou_a"] = ((df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2).shift(26)
    df["ichimoku_senkou_b"] = (
        (df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2
    ).shift(26)

    # Price changes at various lags
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    # Volatility
    df["vol_20"] = close.pct_change().rolling(20).std()
    df["vol_5"] = close.pct_change().rolling(5).std()

    return df.dropna()


# ── State persistence ──────────────────────────────────────────────────────────

def save_state(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def fmt_summary(metrics: Dict[str, Any]) -> str:
    lines = ["=" * 60, "  PERFORMANCE SUMMARY", "=" * 60]
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            if "pct" in key or "rate" in key or "ratio" in key:
                lines.append(f"  {label:<35} {val:>10.4f}")
            else:
                lines.append(f"  {label:<35} {val:>10.2f}")
        else:
            lines.append(f"  {label:<35} {str(val):>10}")
    lines.append("=" * 60)
    return "\n".join(lines)
