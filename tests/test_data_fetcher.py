"""Tests for Hyperliquid data fetcher."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data_fetcher import HyperliquidDataFetcher
from src.utils import (
    add_all_features,
    candles_to_dataframe,
    compute_atr,
    compute_macd,
    compute_rsi,
    interval_to_ms,
    interval_to_seconds,
    ms_to_dt,
    utc_now_ms,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def fetcher(config):
    return HyperliquidDataFetcher(config)


def _make_candle_list(n: int = 100, start_ms: int = 1_700_000_000_000) -> list:
    """Generate synthetic Hyperliquid candle dicts."""
    candles = []
    price = 40_000.0
    interval_ms = 60_000 * 15  # 15m
    for i in range(n):
        price += np.random.normal(0, 200)
        price = max(100, price)
        candles.append(
            {
                "t": start_ms + i * interval_ms,
                "T": start_ms + (i + 1) * interval_ms - 1,
                "o": str(round(price * 0.999, 2)),
                "h": str(round(price * 1.002, 2)),
                "l": str(round(price * 0.997, 2)),
                "c": str(round(price, 2)),
                "v": str(round(abs(np.random.normal(500, 100)), 2)),
                "n": str(np.random.randint(100, 1000)),
            }
        )
    return candles


# ── candles_to_dataframe ──────────────────────────────────────────────────────

class TestCandlesToDataframe:
    def test_basic_conversion(self):
        raw = _make_candle_list(50)
        df = candles_to_dataframe(raw)
        assert not df.empty
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_datetime_index(self):
        raw = _make_candle_list(10)
        df = candles_to_dataframe(raw)
        assert df.index.dtype.tz is not None  # timezone-aware

    def test_sorted_ascending(self):
        raw = _make_candle_list(20)
        # Shuffle
        import random
        random.shuffle(raw)
        df = candles_to_dataframe(raw)
        assert df.index.is_monotonic_increasing

    def test_empty_input(self):
        df = candles_to_dataframe([])
        assert df.empty


# ── Interval helpers ──────────────────────────────────────────────────────────

class TestIntervalHelpers:
    @pytest.mark.parametrize(
        "interval,expected_seconds",
        [
            ("1m", 60),
            ("5m", 300),
            ("15m", 900),
            ("1h", 3600),
            ("4h", 14400),
            ("1d", 86400),
        ],
    )
    def test_interval_to_seconds(self, interval, expected_seconds):
        assert interval_to_seconds(interval) == expected_seconds

    def test_interval_to_ms(self):
        assert interval_to_ms("1h") == 3_600_000

    def test_ms_to_dt(self):
        dt = ms_to_dt(1_700_000_000_000)
        assert dt.year == 2023


# ── Technical indicators ──────────────────────────────────────────────────────

class TestTechnicalIndicators:
    @pytest.fixture
    def close_series(self):
        np.random.seed(42)
        prices = 40_000 + np.cumsum(np.random.normal(0, 200, 200))
        return pd.Series(prices)

    @pytest.fixture
    def ohlcv_df(self, close_series):
        close = close_series
        df = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close),
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": abs(np.random.normal(500, 100, len(close))),
            }
        )
        return df

    def test_rsi_range(self, close_series):
        rsi = compute_rsi(close_series, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_returns_three_series(self, close_series):
        macd, signal, hist = compute_macd(close_series)
        assert len(macd) == len(close_series)
        assert len(signal) == len(close_series)
        assert len(hist) == len(close_series)

    def test_atr_positive(self, ohlcv_df):
        atr = compute_atr(ohlcv_df, period=14)
        valid = atr.dropna()
        assert (valid >= 0).all()

    def test_add_all_features_columns(self, ohlcv_df):
        enriched = add_all_features(ohlcv_df)
        expected_cols = ["rsi_14", "macd", "ema_21", "atr_14", "bb_upper", "vwap", "obv"]
        for col in expected_cols:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_add_all_features_no_nan(self, ohlcv_df):
        enriched = add_all_features(ohlcv_df)
        # add_all_features calls dropna(), so no NaN rows should remain
        assert not enriched.isnull().any().any()


# ── HyperliquidDataFetcher (mocked) ───────────────────────────────────────────

class TestHyperliquidDataFetcher:
    def test_fetch_candles_returns_dataframe(self, fetcher):
        mock_candles = _make_candle_list(100)
        with patch.object(fetcher, "_post", return_value=mock_candles):
            df = fetcher.fetch_candles("BTC", "15m", lookback_candles=100)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_fetch_candles_empty_response(self, fetcher):
        with patch.object(fetcher, "_post", return_value=[]):
            df = fetcher.fetch_candles("BTC", "15m", lookback_candles=50)
        assert df.empty

    def test_fetch_candles_api_error(self, fetcher):
        with patch.object(fetcher, "_post", return_value=None):
            df = fetcher.fetch_candles("BTC", "15m", lookback_candles=50)
        assert df.empty

    def test_fetch_candles_respects_snapshot_end_ms(self, fetcher, monkeypatch):
        snapshot_end_ms = 1_700_000_123_000
        monkeypatch.setenv("DATA_SNAPSHOT_END_MS", str(snapshot_end_ms))
        captured = {}

        def _fake_fetch(_symbol, _interval, _start_ms, end_ms):
            captured["end_ms"] = end_ms
            return _make_candle_list(300)

        with patch.object(fetcher, "_fetch_candle_snapshot", side_effect=_fake_fetch):
            df = fetcher.fetch_candles("BTC", "15m", lookback_candles=50)
        assert not df.empty
        assert captured["end_ms"] == snapshot_end_ms

    def test_fetch_candles_invalid_snapshot_falls_back(self, fetcher, monkeypatch):
        monkeypatch.setenv("DATA_SNAPSHOT_END_MS", "123")
        fallback_end_ms = 1_700_000_000_000
        monkeypatch.setattr("src.data_fetcher.utc_now_ms", lambda: fallback_end_ms)
        captured = {}

        def _fake_fetch(_symbol, _interval, _start_ms, end_ms):
            captured["end_ms"] = end_ms
            return _make_candle_list(300)

        with patch.object(fetcher, "_fetch_candle_snapshot", side_effect=_fake_fetch):
            df = fetcher.fetch_candles("BTC", "15m", lookback_candles=50)
        assert not df.empty
        assert captured["end_ms"] == fallback_end_ms

    def test_fetch_order_book(self, fetcher):
        mock_response = {
            "levels": [
                [{"px": "40100", "sz": "0.5"}, {"px": "40090", "sz": "1.0"}],
                [{"px": "40110", "sz": "0.8"}, {"px": "40120", "sz": "0.3"}],
            ]
        }
        with patch.object(fetcher, "_post", return_value=mock_response):
            book = fetcher.fetch_order_book("BTC")
        assert "bids" in book
        assert "asks" in book
        assert book["mid_price"] == pytest.approx(40105.0)

    def test_fetch_order_book_empty(self, fetcher):
        with patch.object(fetcher, "_post", return_value=None):
            book = fetcher.fetch_order_book("BTC")
        assert book["bids"] == []
        assert book["asks"] == []

    def test_compute_trade_flow_imbalance_all_buys(self, fetcher):
        trades = [{"side": "B", "size": 1.0, "price": 40000, "timestamp_ms": 0}] * 10
        tfi = fetcher.compute_trade_flow_imbalance(trades)
        assert tfi == pytest.approx(1.0)

    def test_compute_trade_flow_imbalance_balanced(self, fetcher):
        trades = (
            [{"side": "B", "size": 1.0, "price": 40000, "timestamp_ms": 0}] * 5
            + [{"side": "A", "size": 1.0, "price": 40000, "timestamp_ms": 0}] * 5
        )
        tfi = fetcher.compute_trade_flow_imbalance(trades)
        assert tfi == pytest.approx(0.0)

    def test_fetch_funding_rate_no_symbol(self, fetcher):
        with patch.object(fetcher, "_post", return_value=None):
            funding = fetcher.fetch_funding_rate("BTC")
        assert funding["funding_rate"] == 0.0

    def test_post_retries_on_failure(self, fetcher):
        """_post should retry up to 3 times on request exception."""
        import requests

        call_count = 0

        def failing_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.ConnectionError("timeout")

        fetcher._session.post = failing_post
        result = fetcher._post({"type": "test"})
        assert result is None
        assert call_count == 3
