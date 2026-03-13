"""
Tests for new indices/index-market features, extended training, infinity-loop
supervised learning, zero-trade hyperparameter adjustment, and stabs/pierces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, MarketConfig


# ── Config: index markets ─────────────────────────────────────────────────────

class TestIndexMarketsConfig:
    def test_index_markets_present(self):
        cfg = load_config()
        syms = [m.symbol for m in cfg.trading.index_markets]
        for expected in ["GOOGL", "AAPL", "NVDA", "US30", "SPX", "JPM", "SPY", "NASDAQ"]:
            assert expected in syms, f"{expected} missing from index_markets"

    def test_index_markets_are_training_only(self):
        cfg = load_config()
        for m in cfg.trading.index_markets:
            assert m.training_only, f"{m.symbol} should be training_only=True"

    def test_index_markets_have_yf_ticker(self):
        cfg = load_config()
        for m in cfg.trading.index_markets:
            assert m.yf_ticker, f"{m.symbol} missing yf_ticker"

    def test_index_markets_market_type(self):
        cfg = load_config()
        for m in cfg.trading.index_markets:
            assert m.market_type == "index", f"{m.symbol} market_type should be 'index'"

    def test_crypto_markets_still_present(self):
        cfg = load_config()
        syms = [m.symbol for m in cfg.trading.markets]
        for expected in ["BTC", "ETH", "SOL"]:
            assert expected in syms

    def test_market_config_has_market_type_field(self):
        m = MarketConfig(symbol="BTC", market_type="crypto")
        assert m.market_type == "crypto"
        m2 = MarketConfig(symbol="GOOGL", market_type="index", training_only=True, yf_ticker="GOOGL")
        assert m2.training_only is True
        assert m2.yf_ticker == "GOOGL"


class TestTrainingLookbackDefaults:
    def test_training_lookback_allows_limited_history(self):
        cfg = load_config()
        assert cfg.data.training_lookback_candles >= 300, (
            "Expected training_lookback_candles >= 300, "
            f"got {cfg.data.training_lookback_candles}"
        )

    def test_historical_csv_dir_configured(self):
        cfg = load_config()
        assert cfg.data.historical_csv_dir
        assert "csv" in cfg.data.historical_csv_dir.lower()

    def test_historical_csv_max_years(self):
        cfg = load_config()
        assert cfg.data.historical_csv_max_years >= 5

    def test_rate_limit_delay_configured(self):
        cfg = load_config()
        assert cfg.data.rate_limit_delay_s >= 0


# ── IndexDataFetcher ──────────────────────────────────────────────────────────

class TestIndexDataFetcherTestMode:
    """All tests run with TRADING_MODE=test to avoid network calls."""

    def test_synthetic_df_returns_ohlcv(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRADING_MODE", "test")
        from src.index_data_fetcher import IndexDataFetcher
        cfg = load_config()
        cfg.data.historical_csv_dir = str(tmp_path / "csv")
        fetcher = IndexDataFetcher(cfg)
        df = fetcher.fetch_ohlcv_history("GOOGL", interval="1d")
        assert not df.empty
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_synthetic_df_has_enough_rows(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRADING_MODE", "test")
        from src.index_data_fetcher import IndexDataFetcher
        cfg = load_config()
        cfg.data.historical_csv_dir = str(tmp_path / "csv")
        fetcher = IndexDataFetcher(cfg)
        df = fetcher.fetch_ohlcv_history("AAPL", interval="1d", lookback_candles=200)
        assert len(df) >= 50

    def test_download_historical_csv_saves_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRADING_MODE", "test")
        from src.index_data_fetcher import IndexDataFetcher
        cfg = load_config()
        cfg.data.historical_csv_dir = str(tmp_path / "csv")
        fetcher = IndexDataFetcher(cfg)
        path = fetcher.download_historical_csv("NVDA", interval="1d")
        assert path.exists()

    def test_different_symbols_produce_different_data(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRADING_MODE", "test")
        from src.index_data_fetcher import IndexDataFetcher
        cfg = load_config()
        cfg.data.historical_csv_dir = str(tmp_path / "csv")
        fetcher = IndexDataFetcher(cfg)
        df1 = fetcher.fetch_ohlcv_history("GOOGL", interval="1d", lookback_candles=100)
        df2 = fetcher.fetch_ohlcv_history("AAPL", interval="1d", lookback_candles=100)
        # Prices should differ (different RNG seeds)
        assert not df1["close"].equals(df2["close"])

    def test_incremental_load_merges_gap(self, monkeypatch, tmp_path):
        """When a CSV exists, the fetcher should detect and fill the gap.

        We stub _download() to avoid network calls while still exercising the
        real CSV load/save/merge code path (no TRADING_MODE=test early-return).
        """
        monkeypatch.delenv("TRADING_MODE", raising=False)
        from src.index_data_fetcher import IndexDataFetcher

        call_count = {"n": 0}

        def fake_download(self_inner, ticker, interval, start, end, retries=4):
            """Return a deterministic DataFrame without hitting the network.

            Data is anchored to ``end`` so rows survive any lookback trimming.
            We generate enough rows (500) to support all technical indicators.
            """
            call_count["n"] += 1
            n = 500
            # Anchor near 'end' so rows survive any lookback trimming.
            idx = pd.date_range(end=end, periods=n, freq="D", tz="UTC")
            rng = np.random.default_rng(call_count["n"])
            prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
            return pd.DataFrame(
                {
                    "open":   prices * 0.999,
                    "high":   prices * 1.002,
                    "low":    prices * 0.997,
                    "close":  prices,
                    "volume": np.full(n, 1_000_000, dtype=float),
                },
                index=idx,
            )

        monkeypatch.setattr(
            "src.index_data_fetcher.IndexDataFetcher._download", fake_download
        )

        cfg = load_config()
        csv_dir = tmp_path / "csv"
        cfg.data.historical_csv_dir = str(csv_dir)
        cfg.data.rate_limit_delay_s = 0.0  # No sleep in tests
        fetcher = IndexDataFetcher(cfg)

        # First download – creates the CSV using the stubbed _download().
        fetcher.download_historical_csv("SPY", interval="1d")
        assert (csv_dir / "SPY_1d.csv").exists(), "First download should create the CSV file"

        first_calls = call_count["n"]
        assert first_calls >= 1

        # Second call – exercises the real CSV load and (if gap exists) merge path.
        # Use lookback_candles=300 so there is enough data for feature engineering.
        df = fetcher.fetch_ohlcv_history("SPY", interval="1d", lookback_candles=300)
        assert not df.empty


# ── Supervised learning: zero-trade detection ─────────────────────────────────

class TestZeroTradeDetection:
    def test_handle_zero_trades_relaxes_thresholds(self):
        cfg = load_config()
        original_lt = cfg.ml.long_threshold
        original_st = cfg.ml.short_threshold
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        adjustments = module.handle_zero_trades()
        # At least one threshold should have been relaxed
        assert len(adjustments) > 0
        assert cfg.ml.long_threshold <= original_lt
        assert cfg.ml.short_threshold <= original_st

    def test_handle_zero_trades_increments_streak(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        module.handle_zero_trades()
        assert module._zero_trade_streak == 1
        module.handle_zero_trades()
        assert module._zero_trade_streak == 2

    def test_reset_zero_trade_streak(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        module.handle_zero_trades()
        module.handle_zero_trades()
        module.reset_zero_trade_streak()
        assert module._zero_trade_streak == 0

    def test_evaluate_and_adjust_with_ai_zero_trades(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        old_lt = cfg.ml.long_threshold
        adjustments = module.evaluate_and_adjust_with_ai(
            trade_history=[],
            current_metrics={"win_rate": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0},
        )
        assert len(adjustments) > 0
        assert cfg.ml.long_threshold <= old_lt

    def test_evaluate_and_adjust_with_ai_uses_callback(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        # Build trade history that passes the zero-trade threshold
        trades = [{"pnl": 10.0} for _ in range(5)]
        metrics = {"win_rate": 0.60, "sharpe_ratio": 1.5, "max_drawdown_pct": 0.05}

        suggested_val = 0.55
        def ai_callback(m):
            return {"ml.long_threshold": suggested_val}

        module.evaluate_and_adjust_with_ai(
            trade_history=trades,
            current_metrics=metrics,
            ai_callback=ai_callback,
        )
        # AI suggestion should have been applied
        assert abs(cfg.ml.long_threshold - suggested_val) < 0.001

    def test_ai_suggestions_clamped_to_safe_range(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        suggestions = {"ml.long_threshold": 0.99}  # Way out of range
        module._apply_ai_suggestions(suggestions)
        assert cfg.ml.long_threshold <= 0.85

    def test_ai_suggestions_unknown_param_ignored(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        original_lt = cfg.ml.long_threshold
        module._apply_ai_suggestions({"unknown.param": 0.5})
        assert cfg.ml.long_threshold == original_lt


# ── Supervised learning: infinity-loop epoch tracking ────────────────────────

class TestInfinityLoopTracking:
    def test_increment_epoch(self):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        assert module.epoch == 0
        module.increment_epoch()
        assert module.epoch == 1
        module.increment_epoch()
        assert module.epoch == 2

    def test_should_evaluate_at_interval(self):
        cfg = load_config()
        cfg.ml.infinity_evaluation_interval = 5
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        for _ in range(4):
            module.increment_epoch()
            assert not module.should_evaluate()
        module.increment_epoch()  # epoch == 5
        assert module.should_evaluate()

    def test_should_evaluate_disabled_when_interval_zero(self):
        cfg = load_config()
        cfg.ml.infinity_evaluation_interval = 0
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        for _ in range(100):
            module.increment_epoch()
        assert not module.should_evaluate()

    def test_save_and_load_state_persists_epoch(self, tmp_path):
        cfg = load_config()
        from src.supervised_learning import SupervisedLearningModule
        module = SupervisedLearningModule(cfg)
        for _ in range(7):
            module.increment_epoch()
        module._zero_trade_streak = 3

        path = tmp_path / "sl_state.json"
        module.save_state(path)

        module2 = SupervisedLearningModule(cfg)
        module2.load_state(path)
        assert module2.epoch == 7
        assert module2._zero_trade_streak == 3


# ── Evaluator: stabs/pierces ──────────────────────────────────────────────────

class TestStabsPierces:
    def _make_trades(self, n_win: int, n_loss: int) -> list:
        trades = []
        for _ in range(n_win):
            trades.append({"pnl": 100.0, "entry_time_ms": 0, "exit_time_ms": 3_600_000})
        for _ in range(n_loss):
            trades.append({"pnl": -200.0, "entry_time_ms": 0, "exit_time_ms": 3_600_000})
        return trades

    def test_stab_alert_triggered_on_low_winrate(self):
        from src.evaluator import Evaluator
        cfg = load_config()
        cfg.evaluation.stabs_enabled = True
        cfg.evaluation.stabs_window_trades = 5
        cfg.evaluation.stabs_min_win_rate = 0.35
        cfg.evaluation.evaluation_window_trades = 200  # Never triggers full eval

        # Last 5 trades all losing
        trades = self._make_trades(n_win=50, n_loss=0) + self._make_trades(n_win=0, n_loss=5)
        evaluator = Evaluator(cfg)
        metrics, _ = evaluator.evaluate(trades, 10_000.0, 9_500.0)
        assert metrics.stab_alert is True

    def test_stab_alert_not_triggered_on_good_recent_trades(self):
        from src.evaluator import Evaluator
        cfg = load_config()
        cfg.evaluation.stabs_enabled = True
        cfg.evaluation.stabs_window_trades = 5
        cfg.evaluation.stabs_min_win_rate = 0.35
        cfg.evaluation.evaluation_window_trades = 200

        # All trades winning
        trades = self._make_trades(n_win=20, n_loss=0)
        evaluator = Evaluator(cfg)
        metrics, _ = evaluator.evaluate(trades, 10_000.0, 12_000.0)
        assert metrics.stab_alert is False

    def test_stabs_disabled_no_alert(self):
        from src.evaluator import Evaluator
        cfg = load_config()
        cfg.evaluation.stabs_enabled = False
        trades = self._make_trades(n_win=0, n_loss=10)
        evaluator = Evaluator(cfg)
        metrics, _ = evaluator.evaluate(trades, 10_000.0, 8_000.0)
        assert metrics.stab_alert is False
        assert metrics.pierce_alert is False

    def test_stab_alert_fields_default_false(self):
        from src.evaluator import PerformanceMetrics
        m = PerformanceMetrics()
        assert m.stab_alert is False
        assert m.pierce_alert is False

    def test_print_report_includes_stab_alert(self):
        from src.evaluator import Evaluator, PerformanceMetrics
        cfg = load_config()
        evaluator = Evaluator(cfg)
        m = PerformanceMetrics(stab_alert=True, pierce_alert=False)
        report = evaluator.print_report(m, [])
        assert "STAB" in report

    def test_print_report_includes_pierce_alert(self):
        from src.evaluator import Evaluator, PerformanceMetrics
        cfg = load_config()
        evaluator = Evaluator(cfg)
        m = PerformanceMetrics(stab_alert=False, pierce_alert=True)
        report = evaluator.print_report(m, [])
        assert "PIERCE" in report


# ── Config: infinity-loop ML settings ────────────────────────────────────────

class TestInfinityLoopConfig:
    def test_infinity_loop_enabled_by_default(self):
        cfg = load_config()
        assert cfg.ml.infinity_loop_enabled is True

    def test_infinity_max_epochs_default_zero(self):
        cfg = load_config()
        # 0 means infinite
        assert cfg.ml.infinity_loop_max_epochs == 0

    def test_zero_trade_threshold_config(self):
        cfg = load_config()
        assert cfg.ml.infinity_zero_trade_threshold == 0

    def test_hp_adjust_step_configured(self):
        cfg = load_config()
        assert cfg.ml.infinity_hp_adjust_step_threshold > 0
        assert cfg.ml.infinity_hp_adjust_agreement_step > 0

    def test_evaluation_interval_configured(self):
        cfg = load_config()
        assert cfg.ml.infinity_evaluation_interval > 0


# ── Config: stabs evaluation settings ────────────────────────────────────────

class TestStabsConfig:
    def test_stabs_enabled_by_default(self):
        cfg = load_config()
        assert cfg.evaluation.stabs_enabled is True

    def test_stabs_window_trades(self):
        cfg = load_config()
        assert cfg.evaluation.stabs_window_trades > 0

    def test_stabs_thresholds(self):
        cfg = load_config()
        assert 0 < cfg.evaluation.stabs_min_win_rate < 1
        assert 0 < cfg.evaluation.stabs_max_drawdown_pct < 1
        assert cfg.evaluation.stabs_pierce_sharpe_threshold >= 0


# ── Main: infinity-train run type ─────────────────────────────────────────────

class TestMainInfinityTrain:
    def test_main_accepts_infinity_train_run_type(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.setenv("INFINITY_MAX_EPOCHS", "1")
        monkeypatch.setenv("TRAINING_SYMBOLS", "BTC")  # Limit to 1 symbol for speed
        monkeypatch.setenv("TRAINING_EPOCHS", "2")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "results").mkdir()
        (tmp_path / ".trading_state").mkdir()

        # Disable index training for speed by patching index_markets to empty list
        from src import config as config_module
        original_load = config_module.load_config
        def fast_load(path=None):
            cfg = original_load(path)
            cfg.trading.index_markets = []  # Skip index training in this test
            return cfg
        monkeypatch.setattr(config_module, "load_config", fast_load)

        from src import main as main_module
        result = main_module.main(["--run-type", "infinity-train", "--mode", "test"])
        assert result == 0

    def test_main_rejects_invalid_run_type(self):
        from src import main as main_module
        import sys
        with pytest.raises(SystemExit):
            main_module.main(["--run-type", "invalid-type"])
