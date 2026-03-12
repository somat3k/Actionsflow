"""Tests for new features: config updates, dual Gemini, ML models, evaluator metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfigNewSymbols:
    def test_new_symbols_present_in_markets(self):
        cfg = load_config()
        symbols = [m.symbol for m in cfg.trading.markets]
        for expected in ["ZRO", "AAVE", "FLOKI", "SHIB", "XAUT"]:
            assert expected in symbols, f"{expected} not found in markets config"

    def test_market_weights_sum_to_one(self):
        cfg = load_config()
        total = sum(m.weight for m in cfg.trading.markets if m.enabled)
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"


class TestConfigTimeframes:
    def test_five_intervals_configured(self):
        cfg = load_config()
        assert cfg.data.primary_interval == "1m"
        assert cfg.data.secondary_interval == "5m"
        assert cfg.data.macro_interval == "15m"
        assert cfg.data.hourly_interval == "1h"
        assert cfg.data.daily_interval == "1d"

    def test_hourly_interval_env_override(self, monkeypatch):
        monkeypatch.setenv("HOURLY_INTERVAL", "4H")
        cfg = load_config()
        assert cfg.data.hourly_interval == "4H"

    def test_daily_interval_env_override(self, monkeypatch):
        monkeypatch.setenv("DAILY_INTERVAL", "1W")
        cfg = load_config()
        assert cfg.data.daily_interval == "1W"


class TestConfigDualGemini:
    def test_dual_api_keys(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "key1")
        monkeypatch.setenv("GEMINI_API_KEY2", "key2")
        cfg = load_config()
        assert cfg.gemini.api_key == "key1"
        assert cfg.gemini.api_key_2 == "key2"

    def test_default_gemini_models(self):
        cfg = load_config()
        assert cfg.gemini.model == "gemini-2.5-pro"
        assert cfg.gemini.model_2 == "gemini-2.5-pro"

    def test_default_openai_model(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        cfg = load_config()
        assert cfg.openai.model == "gpt-4o-mini"


class TestConfigModelWeights:
    def test_model_weights_loaded_from_yaml(self):
        cfg = load_config()
        assert cfg.ml.model_weights["xgb"] == 0.25
        assert cfg.ml.model_weights["gb"] == 0.10
        assert cfg.ml.model_weights["rf"] == 0.15
        assert cfg.ml.model_weights["lstm"] == 0.20
        assert cfg.ml.model_weights["linear"] == 0.10
        assert cfg.ml.model_weights["tree_clf"] == 0.20


# ── Gemini orchestrator tests ─────────────────────────────────────────────────

class TestGeminiShortMessage:
    def test_build_short_message_payload(self):
        from src.gemini_orchestrator import GeminiOrchestrator
        cfg = load_config()
        orch = GeminiOrchestrator(cfg)
        payload = orch.build_short_message_payload(
            "BTC", 1, 0.85, "trending_up", 20, 42000.0
        )
        assert payload["symbol"] == "BTC"
        assert payload["signal"] == "LONG"
        assert payload["confidence"] == 0.85
        assert payload["regime"] == "trending_up"
        assert payload["leverage"] == 20
        assert "BTC LONG" in payload["message"]

    def test_short_message_flat_signal(self):
        from src.gemini_orchestrator import GeminiOrchestrator
        cfg = load_config()
        orch = GeminiOrchestrator(cfg)
        payload = orch.build_short_message_payload(
            "ETH", 0, 0.55, "ranging", 15, 3000.0
        )
        assert payload["signal"] == "FLAT"

    def test_avg_answer_time_initially_zero(self):
        from src.gemini_orchestrator import GeminiOrchestrator
        cfg = load_config()
        orch = GeminiOrchestrator(cfg)
        assert orch.avg_answer_time == 0.0


# ── ML models tests ──────────────────────────────────────────────────────────

class TestQuantumEnsembleLinear:
    def test_linear_model_trained(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        scores = ensemble.train(df, symbol="BTC")

        assert "linear" in scores, "Linear model score should be present"
        assert scores["linear"] > 0.0
        assert ensemble.linear_model is not None

    def test_predict_includes_linear(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        ensemble.train(df, symbol="BTC")
        result = ensemble.predict(df)

        assert "linear" in result.get("model_signals", {})


class TestExtraTreesClassifier:
    """Verify ExtraTreesClassifier trains, predicts, and persists correctly."""

    def test_tree_clf_trained_in_scores(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        scores = ensemble.train(df, symbol="BTC")

        assert "tree_clf" in scores, "ExtraTrees score should be present in train() output"
        assert 0.0 < scores["tree_clf"] <= 1.0
        assert ensemble.tree_clf is not None

    def test_tree_clf_in_model_signals(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        ensemble.train(df, symbol="BTC")
        result = ensemble.predict(df)

        assert "tree_clf" in result.get("model_signals", {}), (
            "ExtraTrees should appear in model_signals after predict()"
        )

    def test_tree_clf_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")

        ensemble = QuantumEnsemble(cfg)
        ensemble.train(df, symbol="BTC")
        assert ensemble.tree_clf is not None

        # Load into a fresh ensemble – tree_clf should be restored.
        ensemble2 = QuantumEnsemble(cfg)
        loaded = ensemble2.load("BTC")
        assert loaded, "load() should return True when models exist"
        assert ensemble2.tree_clf is not None, "tree_clf should be restored after load()"

    def test_tree_clf_in_train_timeframe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        scores = ensemble.train_timeframe(df, "BTC", "1m")

        assert "tree_clf" in scores, "ExtraTrees score should appear in train_timeframe()"

    def test_extra_trees_hyperparams_from_config(self, tmp_path, monkeypatch):
        """ExtraTrees hyperparameters should be read from MLConfig."""
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        assert cfg.ml.extra_trees_n_estimators == 200
        assert cfg.ml.extra_trees_max_depth == 10

        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        ensemble.train(df, symbol="BTC")

        assert ensemble.tree_clf.n_estimators == cfg.ml.extra_trees_n_estimators
        assert ensemble.tree_clf.max_depth == cfg.ml.extra_trees_max_depth


class TestPerTimeframeEpochTraining:
    def test_train_timeframe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        scores = ensemble.train_timeframe(df, "BTC", "1m")

        assert "xgb" in scores
        assert "rf" in scores
        assert "linear" in scores

    def test_predict_timeframe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)
        ensemble.train_timeframe(df, "BTC", "1m")
        result = ensemble.predict_timeframe(df, "1m")

        assert "signal" in result
        assert "confidence" in result
        assert result["timeframe"] == "1m"

    def test_combined_decision(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.data_fetcher import HyperliquidDataFetcher

        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        df = fetcher.fetch_candles("BTC", "1m")
        ensemble = QuantumEnsemble(cfg)

        for tf in ["1m", "5m", "15m"]:
            ensemble.train_timeframe(df, "BTC", tf)

        tf_preds = {}
        for tf in ["1m", "5m", "15m"]:
            tf_preds[tf] = ensemble.predict_timeframe(df, tf)

        combined = ensemble.combined_decision(tf_preds)
        assert "signal" in combined
        assert "confidence" in combined
        assert "agreement" in combined
        assert "timeframe_signals" in combined

    def test_combined_decision_empty(self):
        from src.ml_models import QuantumEnsemble
        cfg = load_config()
        ensemble = QuantumEnsemble(cfg)
        result = ensemble.combined_decision({})
        assert result["signal"] == 0
        assert result["confidence"] == 0.0


# ── Evaluator metric tests ────────────────────────────────────────────────────

class TestEvaluatorNewMetrics:
    def test_new_metrics_present_in_output(self):
        from src.evaluator import compute_metrics

        trades = [
            {
                "pnl": 100.0,
                "fee_usd": 1.0,
                "leverage": 15,
                "duration_ms": 3_600_000,
                "entry_time_ms": 1_000_000_000_000,
                "exit_time_ms": 1_000_003_600_000,
            },
            {
                "pnl": -50.0,
                "fee_usd": 1.0,
                "leverage": 10,
                "duration_ms": 1_800_000,
                "entry_time_ms": 1_000_003_700_000,
                "exit_time_ms": 1_000_005_500_000,
            },
        ]
        m = compute_metrics(trades, 10_000.0, 10_050.0)

        assert hasattr(m, "accuracy")
        assert hasattr(m, "equity_growth_pct")
        assert hasattr(m, "num_positions")
        assert hasattr(m, "gemini_answer_time_avg_s")
        assert hasattr(m, "action_time_avg_s")
        assert m.accuracy == m.win_rate
        assert abs(m.equity_growth_pct - m.total_return_pct) < 1e-6

    def test_optional_kwargs_are_forwarded(self):
        from src.evaluator import compute_metrics

        m = compute_metrics(
            [], 10_000.0, 10_000.0,
            num_positions=5,
            gemini_answer_time_avg_s=1.23,
            action_time_avg_s=0.45,
        )
        assert m.num_positions == 5
        assert m.gemini_answer_time_avg_s == 1.23
        assert m.action_time_avg_s == 0.45


# ── Data fetcher multi-timeframe tests ────────────────────────────────────────

class TestDataFetcherMultiTimeframe:
    def test_fetch_multi_timeframe_returns_five_frames(self, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")

        from src.data_fetcher import HyperliquidDataFetcher
        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)
        frames = fetcher.fetch_multi_timeframe("BTC")

        assert "1m" in frames
        assert "5m" in frames
        assert "15m" in frames
        assert "1h" in frames
        assert "1d" in frames

    def test_new_symbols_have_funding_rate(self, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "test")

        from src.data_fetcher import HyperliquidDataFetcher
        cfg = load_config()
        fetcher = HyperliquidDataFetcher(cfg)

        for symbol in ["ZRO", "AAVE", "FLOKI", "SHIB", "XAUT"]:
            result = fetcher.fetch_funding_rate(symbol)
            assert result["symbol"] == symbol
            assert "funding_rate" in result
