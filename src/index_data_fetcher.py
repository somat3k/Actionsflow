"""
Quantum Trading System – Index / Equity Data Fetcher

Fetches OHLCV data for traditional market indices and equities (GOOGL, AAPL,
NVDA, US30/^DJI, S&P500/^GSPC, JPM, SPY, NASDAQ/QQQ) via Yahoo Finance.

Features
--------
* Rate-limited requests with configurable delay and exponential-backoff retry.
* Missing-period gap detection: only fetches the absent date range when a
  cached CSV already covers earlier history.
* Long-range historical CSV download (up to ``max_years`` of daily data).
* Returns DataFrames in the same format as ``HyperliquidDataFetcher`` (OHLCV
  with technical indicators added via ``add_all_features``).

Set ``TRADING_MODE=test`` to skip real network calls and receive synthetic
data (compatible with CI/unit-test environments that have no internet access).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.utils import add_all_features, get_logger

log = get_logger(__name__)

# Yahoo Finance interval strings that map to our internal interval names.
_YF_INTERVAL_MAP: Dict[str, str] = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "1d":  "1d",
    "1wk": "1wk",
}

# Minimum rows needed before features can be reliably computed.
_MIN_ROWS_FOR_FEATURES = 50


class IndexDataFetcher:
    """Fetch equity / index OHLCV data from Yahoo Finance with rate limiting.

    Parameters
    ----------
    config:
        Application configuration (used for CSV dir, rate-limit delay, etc.).
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.csv_dir = Path(config.data.historical_csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = config.data.rate_limit_delay_s

    # ── Public interface ───────────────────────────────────────────────────────

    def fetch_ohlcv_history(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_candles: Optional[int] = None,
        yf_ticker: Optional[str] = None,
        include_features: bool = True,
    ) -> pd.DataFrame:
        """Fetch extended OHLCV history for *symbol* at the given *interval*.

        Strategy
        --------
        1. Check whether a CSV cache exists covering the required date range.
        2. If the cached file is complete, load and return it.
        3. If the cache is partial (missing recent data), fetch only the gap.
        4. If no cache exists, download the full ``lookback_candles`` range.

        Parameters
        ----------
        symbol:
            Internal symbol name (e.g. ``"GOOGL"``).
        interval:
            OHLCV interval; one of ``1m, 5m, 15m, 1h, 1d``.
        lookback_candles:
            Number of candles requested (default: ``training_lookback_candles``).
        yf_ticker:
            Yahoo Finance ticker override (e.g. ``"^DJI"`` for US30).
            Falls back to *symbol* when not provided.
        include_features:
            Whether to run ``add_all_features`` on the result.
        """
        if os.environ.get("TRADING_MODE") == "test":
            return self._synthetic_df(
                symbol, interval, lookback_candles or 400,
                include_features=include_features,
            )

        ticker = yf_ticker or symbol
        n = lookback_candles or self.cfg.data.training_lookback_candles

        # Derive the required start date from lookback_candles and interval.
        now = datetime.now(tz=timezone.utc)
        interval_seconds = _interval_to_seconds(interval)
        required_start = now - timedelta(seconds=interval_seconds * n)

        csv_path = self._csv_path(symbol, interval)

        # ── Try incremental load from existing CSV ─────────────────────────
        if csv_path.exists():
            existing = self._load_csv(csv_path)
            if not existing.empty:
                latest_ts = existing.index[-1]
                if latest_ts.tzinfo is None:
                    latest_ts = latest_ts.replace(tzinfo=timezone.utc)
                gap_start = latest_ts + timedelta(seconds=interval_seconds)
                if gap_start < now - timedelta(seconds=interval_seconds * 2):
                    log.info(
                        "Fetching missing %s@%s gap: %s → %s",
                        symbol, interval, gap_start.date(), now.date(),
                    )
                    gap_df = self._download(ticker, interval, gap_start, now)
                    if not gap_df.empty:
                        combined = pd.concat([existing, gap_df])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        self._save_csv(combined, csv_path)
                        existing = combined
                # Trim to lookback window.
                cutoff = now - timedelta(seconds=interval_seconds * n)
                existing = existing[existing.index >= cutoff]
                if len(existing) >= _MIN_ROWS_FOR_FEATURES:
                    if include_features:
                        existing = self._add_features_safe(existing, symbol, interval)
                    return existing

        # ── Full download ──────────────────────────────────────────────────
        # For daily/weekly intervals use max_years cap as the outer bound, but
        # always ensure the window is at least as wide as lookback_candles.
        max_years = self.cfg.data.historical_csv_max_years
        if interval in ("1d", "1wk"):
            cap_start = now - timedelta(days=365 * max_years)
            # If lookback_candles requests a longer window, honour it (capped by
            # max_years so we never request more than the configured limit).
            start = min(cap_start, required_start)
            if start < cap_start:
                log.warning(
                    "%s@%s lookback_candles=%d requests %s but historical_csv_max_years"
                    " =%d caps download to %s; trimmed result may have fewer rows.",
                    symbol, interval, n, required_start.date(), max_years, cap_start.date(),
                )
                start = cap_start
        else:
            start = required_start

        log.info("Downloading %s@%s history from %s", symbol, interval, start.date())
        df = self._download(ticker, interval, start, now)

        if df.empty:
            log.warning("No data returned for %s@%s from Yahoo Finance", symbol, interval)
            return df

        self._save_csv(df, csv_path)

        # Trim to lookback window before returning.
        cutoff = now - timedelta(seconds=interval_seconds * n)
        df = df[df.index >= cutoff]

        if len(df) < _MIN_ROWS_FOR_FEATURES:
            log.warning(
                "Insufficient rows (%d) for %s@%s after trimming", len(df), symbol, interval
            )
            return df

        if include_features:
            df = self._add_features_safe(df, symbol, interval)

        return df

    def download_historical_csv(
        self,
        symbol: str,
        yf_ticker: Optional[str] = None,
        interval: str = "1d",
        max_years: Optional[int] = None,
    ) -> Path:
        """Download the maximum available history for *symbol* and save as CSV.

        Returns the path to the saved file.  Useful for seeding the dataset
        cache before the first training run.
        """
        if os.environ.get("TRADING_MODE") == "test":
            path = self._csv_path(symbol, interval)
            df = self._synthetic_df(symbol, interval, 500)
            self._save_csv(df, path)
            return path

        ticker = yf_ticker or symbol
        years = max_years or self.cfg.data.historical_csv_max_years
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(days=365 * years)

        log.info("Downloading full %s@%s CSV (%d years)", symbol, interval, years)
        df = self._download(ticker, interval, start, now)
        if df.empty:
            log.warning("No data downloaded for %s", symbol)
            return self._csv_path(symbol, interval)

        path = self._csv_path(symbol, interval)
        self._save_csv(df, path)
        log.info("Saved %d rows → %s", len(df), path)
        return path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _download(
        self,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
        retries: int = 4,
    ) -> pd.DataFrame:
        """Download OHLCV from Yahoo Finance with retry + rate-limit backoff."""
        try:
            import yfinance as yf  # optional dep – imported lazily
        except ImportError:
            log.error(
                "yfinance is not installed. Install it with: pip install yfinance"
            )
            return pd.DataFrame()

        yf_interval = _YF_INTERVAL_MAP.get(interval, interval)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        for attempt in range(retries):
            try:
                time.sleep(self._rate_limit_delay)
                raw = yf.download(
                    ticker,
                    start=start_str,
                    end=end_str,
                    interval=yf_interval,
                    auto_adjust=True,
                    progress=False,
                )
                if raw is None or raw.empty:
                    log.warning(
                        "Empty response for %s@%s (attempt %d/%d)",
                        ticker, interval, attempt + 1, retries,
                    )
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                    continue
                df = self._normalise(raw)
                log.info(
                    "Downloaded %d rows for %s@%s (%s → %s)",
                    len(df), ticker, interval, df.index[0], df.index[-1],
                )
                return df
            except Exception as exc:
                wait = 2 ** attempt
                log.warning(
                    "Yahoo Finance attempt %d/%d for %s failed (%s). Retrying in %ds",
                    attempt + 1, retries, ticker, exc, wait,
                )
                time.sleep(wait)

        log.error("All %d attempts failed for %s@%s", retries, ticker, interval)
        return pd.DataFrame()

    @staticmethod
    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise a yfinance DataFrame into our standard OHLCV format."""
        # yfinance may return a MultiIndex or flat columns.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower() for c in raw.columns]

        rename = {
            "open":   "open",
            "high":   "high",
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        }
        available = {k: v for k, v in rename.items() if k in raw.columns}
        df = raw.rename(columns=available)[list(available.values())].copy()
        df.index.name = "timestamp"
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        return df

    def _add_features_safe(
        self, df: pd.DataFrame, symbol: str, interval: str
    ) -> pd.DataFrame:
        """Apply add_all_features, logging and skipping on failure."""
        try:
            return add_all_features(df)
        except Exception as exc:
            log.warning(
                "Feature engineering failed for %s@%s: %s", symbol, interval, exc
            )
            return df

    def _csv_path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("^", "_").replace("/", "_")
        return self.csv_dir / f"{safe}_{interval}.csv"

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index.name = "timestamp"
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.sort_index(inplace=True)
            return df
        except Exception as exc:
            log.warning("Failed to load CSV %s: %s", path, exc)
            return pd.DataFrame()

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    # ── Test / synthetic helpers ───────────────────────────────────────────────

    @staticmethod
    def _synthetic_df(
        symbol: str,
        interval: str,
        n: int = 400,
        include_features: bool = True,
    ) -> pd.DataFrame:
        """Return deterministic synthetic OHLCV for test mode (no network calls).

        Uses ``zlib.crc32`` for a stable, process-independent seed and a fixed
        anchor timestamp so the output is reproducible across runs and
        ``PYTHONHASHSEED`` settings.
        """
        import zlib

        seed = zlib.crc32(symbol.encode()) & 0x7FFF_FFFF
        rng = np.random.default_rng(seed)
        price = 100.0
        interval_s = _interval_to_seconds(interval)
        # Fixed anchor so timestamps don't change between runs.
        _ANCHOR_MS = 1_700_000_000_000
        anchor = datetime.fromtimestamp(_ANCHOR_MS / 1000.0, tz=timezone.utc)
        timestamps = [anchor + timedelta(seconds=interval_s * i) for i in range(n)]

        rows = []
        for _ in range(n):
            price *= 1.0 + float(rng.normal(0, 0.001))
            price = max(1.0, price)
            rows.append(
                {
                    "open":   round(price * 0.999, 4),
                    "high":   round(price * 1.002, 4),
                    "low":    round(price * 0.997, 4),
                    "close":  round(price, 4),
                    "volume": round(abs(float(rng.normal(1_000_000, 200_000))), 0),
                }
            )

        df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, name="timestamp"))
        if include_features:
            try:
                df = add_all_features(df)
            except Exception:
                pass
        return df


# ── Utility ────────────────────────────────────────────────────────────────────

def _interval_to_seconds(interval: str) -> int:
    """Convert interval string to approximate seconds."""
    mapping = {
        "1m":  60,
        "5m":  300,
        "15m": 900,
        "1h":  3600,
        "4h":  14400,
        "1d":  86400,
        "1wk": 604800,
    }
    return mapping.get(interval, 86400)
