"""
Quantum Trading System – Hyperliquid Data Fetcher
Fetches OHLCV candles, order book snapshots, funding rates, and open interest
from the Hyperliquid public REST API.

Set TRADING_MODE=test to use synthetic data instead of live API calls.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from src.config import AppConfig, DataConfig
from src.utils import (
    add_all_features,
    candles_to_dataframe,
    get_logger,
    interval_to_ms,
    utc_now_ms,
)

import pandas as pd

log = get_logger(__name__)

# Maximum candles per single API request (Hyperliquid limit)
_MAX_CANDLES_PER_REQUEST = 5000
_REQUEST_TIMEOUT = 30  # seconds

# ── Symbol registry (loaded once from JSON) ───────────────────────────────────
_SYMBOLS_FILE = Path(__file__).resolve().parent.parent / "config" / "symbols.json"


def _load_symbols_registry() -> List[Dict[str, Any]]:
    """Load the full symbol list from config/symbols.json."""
    try:
        with open(_SYMBOLS_FILE) as fh:
            data = json.load(fh)
        return data.get("symbols", [])
    except Exception as exc:
        log.warning("Failed to load symbols.json (%s); using empty fallback", exc)
        return []


_SYMBOLS_REGISTRY: List[Dict[str, Any]] = _load_symbols_registry()


class HyperliquidDataFetcher:
    """Fetches all market data needed by the trading system from Hyperliquid."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.api_url = config.data.hyperliquid_api_url
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Public interface ───────────────────────────────────────────────────────

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        lookback_candles: Optional[int] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and enrich with technical indicators."""
        n = lookback_candles or self.cfg.data.lookback_candles
        end_ms = self._resolve_end_ms(end_ms)
        start_ms = start_ms or (end_ms - n * interval_to_ms(interval))

        raw = self._fetch_candle_snapshot(symbol, interval, start_ms, end_ms)
        if not raw:
            log.warning("No candle data returned for %s@%s", symbol, interval)
            return pd.DataFrame()

        df = candles_to_dataframe(raw)
        df = add_all_features(df)
        log.info(
            "Fetched %d candles for %s@%s (%s → %s)",
            len(df),
            symbol,
            interval,
            df.index[0] if len(df) else "N/A",
            df.index[-1] if len(df) else "N/A",
        )
        return df

    def fetch_ohlcv_history(
        self,
        symbol: str,
        interval: str,
        lookback_candles: Optional[int] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        include_features: bool = True,
    ) -> pd.DataFrame:
        """Fetch extended-range OHLCV history for dataset creation."""
        n = lookback_candles or self.cfg.data.lookback_candles
        end_ms = self._resolve_end_ms(end_ms)
        start_ms = start_ms or (end_ms - n * interval_to_ms(interval))

        log.info(
            "Downloading OHLCV history for %s (%s, lookback=%s)",
            symbol,
            interval,
            n,
        )
        raw = self._fetch_candle_snapshot(symbol, interval, start_ms, end_ms)
        if not raw:
            log.warning("No candle history returned for %s@%s", symbol, interval)
            return pd.DataFrame()

        df = candles_to_dataframe(raw)
        if include_features:
            df = add_all_features(df)
        return df

    def fetch_multi_timeframe(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch candles for all configured timeframes (1m, 5m, 15m, 1h, 1d)."""
        cfg = self.cfg.data
        frames: Dict[str, pd.DataFrame] = {}
        for tf in [
            cfg.primary_interval,
            cfg.secondary_interval,
            cfg.macro_interval,
            cfg.hourly_interval,
            cfg.daily_interval,
        ]:
            frames[tf] = self.fetch_candles(symbol, tf)
        return frames

    def save_ohlcv_csv(
        self,
        symbol: str,
        interval: str,
        lookback_candles: Optional[int] = None,
    ) -> Path:
        """Incrementally update the OHLCV CSV for *symbol*/*interval*.

        On first run the full ``lookback_candles`` history is fetched.  On
        subsequent runs only the gap since the last saved timestamp is fetched
        and merged, keeping the CSV small.  Rows older than
        ``data.historical_csv_max_years`` are trimmed so files stay bounded.

        Returns the path to the (updated) CSV file.
        """
        csv_dir = Path(self.cfg.data.historical_csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{symbol}_{interval}.csv"

        max_years = self.cfg.data.historical_csv_max_years
        base_lookback = lookback_candles or self.cfg.data.training_lookback_candles
        interval_ms = interval_to_ms(interval)
        now_ms = utc_now_ms()

        # ── Load existing CSV (if any) ────────────────────────────────────
        existing: Optional[pd.DataFrame] = None
        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if existing.empty:
                    existing = None
            except Exception as exc:
                log.warning("Could not read existing CSV %s (%s) – starting fresh", csv_path, exc)
                existing = None

        # ── Decide fetch window ───────────────────────────────────────────
        fetch_candles: Optional[int] = base_lookback
        if existing is not None:
            try:
                last_ts_ms = int(existing.index.max().timestamp() * 1000)
                ms_gap = max(0, now_ms - last_ts_ms)
                if ms_gap <= interval_ms:
                    # Already up-to-date; skip fetch.
                    log.info("CSV %s is up-to-date – skipping download", csv_path)
                    return csv_path
                fetch_candles = min(base_lookback, max(1, ms_gap // interval_ms + 1))
            except Exception:
                pass  # Unusable index → fall back to base_lookback

        df = self.fetch_ohlcv_history(
            symbol, interval, lookback_candles=fetch_candles, include_features=False
        )
        if df.empty:
            log.warning("No OHLCV data fetched for %s@%s – skipping CSV save", symbol, interval)
            return csv_path

        # ── Merge with existing ───────────────────────────────────────────
        if existing is not None:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        # ── Enforce retention window ──────────────────────────────────────
        if max_years > 0 and isinstance(df.index, pd.DatetimeIndex):
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=365 * max_years)
            df = df[df.index >= cutoff]

        df.to_csv(csv_path)
        log.info("Saved %d OHLCV rows → %s", len(df), csv_path)
        return csv_path

    def fetch_order_book(self, symbol: str) -> Dict[str, Any]:
        """Fetch level-2 order book snapshot."""
        payload = {"type": "l2Book", "coin": symbol}
        data = self._post(payload)
        if not data:
            return {"bids": [], "asks": [], "symbol": symbol}

        levels = data.get("levels", [[], []])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []

        result = {
            "symbol": symbol,
            "timestamp_ms": utc_now_ms(),
            "bids": [{"price": float(b["px"]), "size": float(b["sz"])} for b in bids],
            "asks": [{"price": float(a["px"]), "size": float(a["sz"])} for a in asks],
        }

        if result["bids"] and result["asks"]:
            best_bid = result["bids"][0]["price"]
            best_ask = result["asks"][0]["price"]
            mid = (best_bid + best_ask) / 2
            result["mid_price"] = mid
            result["bid_ask_spread"] = best_ask - best_bid
            result["bid_ask_spread_bps"] = (best_ask - best_bid) / mid * 10_000

            total_bid_size = sum(b["size"] for b in result["bids"][:10])
            total_ask_size = sum(a["size"] for a in result["asks"][:10])
            denom = total_bid_size + total_ask_size
            result["order_book_imbalance"] = (
                (total_bid_size - total_ask_size) / denom if denom else 0.0
            )

        return result

    def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch current funding rate for a perpetual market."""
        payload = {"type": "metaAndAssetCtxs"}
        data = self._post(payload)
        if not isinstance(data, list) or len(data) < 2:
            return {"symbol": symbol, "funding_rate": 0.0, "open_interest": 0.0}

        meta = data[0]
        asset_ctxs = data[1]
        universe = meta.get("universe", [])

        idx: Optional[int] = None
        for i, asset in enumerate(universe):
            if asset.get("name", "").upper() == symbol.upper():
                idx = i
                break

        if idx is None or idx >= len(asset_ctxs):
            return {"symbol": symbol, "funding_rate": 0.0, "open_interest": 0.0}

        ctx = asset_ctxs[idx]
        return {
            "symbol": symbol,
            "timestamp_ms": utc_now_ms(),
            "funding_rate": float(ctx.get("funding", 0.0)),
            "open_interest": float(ctx.get("openInterest", 0.0)),
            "mark_price": float(ctx.get("markPx", 0.0)),
            "oracle_price": float(ctx.get("oraclePx", 0.0)),
            "mid_price": float(ctx.get("midPx", 0.0)),
        }

    def fetch_recent_trades(self, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch recent public trades."""
        payload = {"type": "recentTrades", "coin": symbol}
        data = self._post(payload)
        if not isinstance(data, list):
            return []
        trades = []
        for t in data[-limit:]:
            trades.append(
                {
                    "timestamp_ms": int(t.get("time", 0)),
                    "price": float(t.get("px", 0)),
                    "size": float(t.get("sz", 0)),
                    "side": t.get("side", ""),
                }
            )
        return trades

    def compute_trade_flow_imbalance(
        self, trades: List[Dict[str, Any]], window: int = 100
    ) -> float:
        """Compute buy/sell volume imbalance from recent trades."""
        if not trades:
            return 0.0
        recent = trades[-window:]
        buy_vol = sum(t["size"] for t in recent if t["side"] == "B")
        sell_vol = sum(t["size"] for t in recent if t["side"] == "A")
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0.0

    def fetch_all_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch complete market snapshot for a symbol."""
        mtf = self.fetch_multi_timeframe(symbol)
        order_book = self.fetch_order_book(symbol)
        funding = self.fetch_funding_rate(symbol)
        trades = self.fetch_recent_trades(symbol)
        tfi = self.compute_trade_flow_imbalance(trades)

        return {
            "symbol": symbol,
            "timestamp_ms": utc_now_ms(),
            "candles": mtf,
            "order_book": order_book,
            "funding": funding,
            "trade_flow_imbalance": tfi,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_end_ms(self, end_ms: Optional[int]) -> int:
        if end_ms is not None:
            return end_ms
        snapshot_raw = os.environ.get("DATA_SNAPSHOT_END_MS")
        if snapshot_raw:
            try:
                return int(snapshot_raw)
            except ValueError:
                log.warning(
                    "Invalid DATA_SNAPSHOT_END_MS=%s; falling back to current time",
                    snapshot_raw,
                )
        return utc_now_ms()

    def _fetch_candle_snapshot(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> List[Dict]:
        """Fetch candles in paginated chunks if needed."""
        all_candles: List[Dict] = []
        chunk_ms = _MAX_CANDLES_PER_REQUEST * interval_to_ms(interval)
        current_start = start_ms

        while current_start < end_ms:
            current_end = min(current_start + chunk_ms, end_ms)
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": current_end,
                },
            }
            chunk = self._post(payload)
            if not isinstance(chunk, list) or not chunk:
                break
            all_candles.extend(chunk)
            current_start = current_end + interval_to_ms(interval)
            if len(chunk) < 10:
                break

        return all_candles

    def _post(self, payload: Dict) -> Any:
        """POST to Hyperliquid info endpoint with retry logic.

        When ``TRADING_MODE=test``, returns synthetic data instead of making
        real network requests so the complete pipeline can be exercised in CI
        without external API credentials.
        """
        if os.environ.get("TRADING_MODE") == "test":
            return self._synthetic_response(payload)

        retries = 3
        for attempt in range(retries):
            try:
                response = self._session.post(
                    self.api_url, json=payload, timeout=_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                if attempt == retries - 1:
                    log.error("Hyperliquid API error (payload=%s): %s", payload.get("type"), exc)
                    return None
                wait = 2 ** attempt
                log.warning(
                    "Hyperliquid API attempt %d/%d failed, retrying in %ds: %s",
                    attempt + 1,
                    retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
        return None

    # ── Test-mode helpers ──────────────────────────────────────────────────────

    def _synthetic_response(self, payload: Dict) -> Any:
        """Return deterministic synthetic data for test/CI mode (TRADING_MODE=test)."""
        ptype = payload.get("type", "")
        if ptype == "candleSnapshot":
            interval = payload.get("req", {}).get("interval", "15m")
            return self._synthetic_candles(interval=interval)
        if ptype == "l2Book":
            return {
                "levels": [
                    [{"px": "40100", "sz": "0.5"}, {"px": "40090", "sz": "1.0"}],
                    [{"px": "40110", "sz": "0.8"}, {"px": "40120", "sz": "0.3"}],
                ]
            }
        if ptype == "metaAndAssetCtxs":
            # Return entries for all symbols from the registry so fetch_funding_rate
            # works for any configured symbol in test/CI mode.
            universe = [{"name": entry["name"]} for entry in _SYMBOLS_REGISTRY]
            contexts = [
                {
                    "funding": "0.0001",
                    "openInterest": str(1_000.0),
                    "markPx": str(entry.get("synthetic_price", 1.0)),
                    "oraclePx": str(entry.get("synthetic_price", 1.0)),
                    "midPx": str(entry.get("synthetic_price", 1.0)),
                }
                for entry in _SYMBOLS_REGISTRY
            ]
            return [{"universe": universe}, contexts]
        if ptype == "recentTrades":
            return [
                {"time": 1_700_000_000_000 + i * 1_000, "px": "40000", "sz": "1.0", "side": "B"}
                for i in range(20)
            ]
        return None

    @staticmethod
    def _synthetic_candles(n: int = 400, interval: str = "15m") -> List[Dict]:
        """Generate deterministic synthetic OHLCV candle dicts for test mode.

        Uses a fixed random seed so results are reproducible across runs.
        Generates ``n`` candles (default 400), which is sufficient for all
        technical indicators produced by :func:`~src.utils.add_all_features`.

        The ``interval`` parameter controls the candle spacing so each
        timeframe has realistic timestamps in multi-timeframe tests.
        """
        rng = np.random.default_rng(42)
        candles: List[Dict] = []
        price = 40_000.0
        interval_ms = interval_to_ms(interval)
        start_ms = 1_700_000_000_000
        for i in range(n):
            price *= 1.0 + float(rng.normal(0, 0.001))
            price = max(1_000.0, price)
            candles.append(
                {
                    "t": start_ms + i * interval_ms,
                    "T": start_ms + (i + 1) * interval_ms - 1,
                    "o": str(round(price * 0.999, 2)),
                    "h": str(round(price * 1.002, 2)),
                    "l": str(round(price * 0.997, 2)),
                    "c": str(round(price, 2)),
                    "v": str(round(float(abs(rng.normal(500, 100))), 2)),
                    "n": str(int(rng.integers(100, 1_000))),  # num_trades (Hyperliquid field)
                }
            )
        return candles
