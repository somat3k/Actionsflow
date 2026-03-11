"""
Tests for:
  - QuantumEnsemble.export_onnx()  – ONNX export of GB/RF/Linear models
  - ModelDelegationAgent           – regime-based model selection
  - run_model_export()             – end-to-end export + CSV run-type
  - CSV data export functionality
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.ml_models import ModelDelegationAgent, QuantumEnsemble, _REGIME_MODEL_MAP


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_df(n: int = 300) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with the feature columns the ensemble expects.

    Uses zigzag price movements to ensure all three label classes (flat/long/short)
    appear in the training data.  Requires at least 200 raw rows because the
    Ichimoku Senkou-B component (52-period rolling + 26-period shift) alone
    needs ~78 warm-up bars.
    """
    n = max(n, 200)
    rng = np.random.default_rng(42)
    # Zigzag pattern guarantees both positive and negative returns
    half = n // 2
    up = np.linspace(100, 120, half)
    down = np.linspace(120, 100, n - half)
    close = np.concatenate([up, down]) + rng.uniform(-0.3, 0.3, n)
    df = pd.DataFrame(
        {
            "open": close + rng.uniform(-0.5, 0.5, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        }
    )
    from src.utils import add_all_features
    df = add_all_features(df)
    return df.reset_index(drop=True)


def _trained_ensemble(tmp_path: Path) -> QuantumEnsemble:
    """Return a QuantumEnsemble that has been trained on synthetic data."""
    import os
    os.chdir(tmp_path)
    (tmp_path / "models").mkdir(exist_ok=True)
    cfg = load_config()
    ensemble = QuantumEnsemble(cfg)
    df = _make_df(80)
    ensemble.train(df, symbol="BTC", save=True)
    return ensemble


# ── ONNX export ───────────────────────────────────────────────────────────────


class TestExportOnnx:
    def test_export_skipped_when_skl2onnx_missing(self, tmp_path, monkeypatch):
        """export_onnx returns {} and logs a warning when skl2onnx is not available."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        cfg = load_config()
        ensemble = QuantumEnsemble(cfg)
        df = _make_df(80)
        ensemble.train(df, symbol="BTC", save=True)

        import src.ml_models as ml_mod
        monkeypatch.setattr(ml_mod, "_ONNX_AVAILABLE", False)

        result = ensemble.export_onnx("BTC")
        assert result == {}

    def test_export_writes_onnx_files(self, tmp_path, monkeypatch):
        """When skl2onnx is available, .onnx files are written for GB, RF, and Linear."""
        pytest.importorskip("skl2onnx")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        cfg = load_config()
        ensemble = QuantumEnsemble(cfg)
        df = _make_df(80)
        ensemble.train(df, symbol="BTC", save=True)

        exported = ensemble.export_onnx("BTC")
        assert set(exported.keys()) == {"gb", "rf", "linear"}
        for name, path_str in exported.items():
            assert Path(path_str).exists(), f"{name}.onnx not found at {path_str}"
            assert Path(path_str).stat().st_size > 0

    def test_export_returns_subset_when_model_missing(self, tmp_path, monkeypatch):
        """export_onnx only exports models that were actually trained."""
        pytest.importorskip("skl2onnx")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        cfg = load_config()
        ensemble = QuantumEnsemble(cfg)
        df = _make_df(80)
        ensemble.train(df, symbol="ETH", save=True)
        # Forcibly remove one model after training
        ensemble.rf_model = None

        exported = ensemble.export_onnx("ETH")
        assert "rf" not in exported
        assert "gb" in exported
        assert "linear" in exported


# ── ModelDelegationAgent ─────────────────────────────────────────────────────


class TestModelDelegationAgent:
    def _make_agent(self, tmp_path: Path) -> ModelDelegationAgent:
        import os
        os.chdir(tmp_path)
        (tmp_path / "models").mkdir(exist_ok=True)
        cfg = load_config()
        ensemble = QuantumEnsemble(cfg)
        df = _make_df(80)
        ensemble.train(df, symbol="BTC", save=True)
        return ModelDelegationAgent(ensemble)

    def test_unknown_regime_falls_back_to_ensemble(self, tmp_path):
        agent = self._make_agent(tmp_path)
        df = _make_df(50)
        result = agent.predict(df, regime="unknown")
        assert "signal" in result
        assert "delegated_to" in result
        assert result["delegated_to"] == "ensemble"

    def test_trending_regime_delegates_to_xgb(self, tmp_path, monkeypatch):
        """Trending regimes should prefer XGBoost when it is available."""
        import src.ml_models as ml_mod
        monkeypatch.setattr(ml_mod, "_XGB_AVAILABLE", True)
        agent = self._make_agent(tmp_path)
        # Ensure xgb_model is loaded (requires XGBoost)
        if agent._ensemble.xgb_model is None:
            pytest.skip("XGBoost not available in this environment")
        df = _make_df(50)
        result = agent.predict(df, regime="trending_up")
        assert result["delegated_to"] == "xgb"

    def test_volatile_regime_delegates_to_rf(self, tmp_path):
        agent = self._make_agent(tmp_path)
        df = _make_df(50)
        result = agent.predict(df, regime="volatile")
        assert result["delegated_to"] == "rf"

    def test_ranging_regime_delegates_to_linear(self, tmp_path):
        agent = self._make_agent(tmp_path)
        df = _make_df(50)
        result = agent.predict(df, regime="ranging")
        assert result["delegated_to"] == "linear"

    def test_consolidating_regime_delegates_to_linear(self, tmp_path):
        agent = self._make_agent(tmp_path)
        df = _make_df(50)
        result = agent.predict(df, regime="consolidating")
        assert result["delegated_to"] == "linear"

    def test_fallback_when_preferred_model_missing(self, tmp_path):
        """Falls back to ensemble if the regime-preferred model is None."""
        agent = self._make_agent(tmp_path)
        agent._ensemble.rf_model = None  # remove RF model
        df = _make_df(50)
        result = agent.predict(df, regime="volatile")
        # RF is missing → should fall back to ensemble
        assert result["delegated_to"] == "ensemble"

    def test_output_shape_matches_ensemble(self, tmp_path):
        """Delegated prediction must have the same keys as ensemble.predict()."""
        agent = self._make_agent(tmp_path)
        df = _make_df(50)
        ensemble_result = agent._ensemble.predict(df)
        delegated_result = agent.predict(df, regime="ranging")
        expected_keys = set(ensemble_result.keys()) | {"delegated_to"}
        assert expected_keys == set(delegated_result.keys())

    def test_signal_values_are_valid(self, tmp_path):
        agent = self._make_agent(tmp_path)
        df = _make_df(50)
        for regime in ["unknown", "volatile", "ranging", "consolidating"]:
            result = agent.predict(df, regime=regime)
            assert result["signal"] in (0, 1, 2)
            assert 0.0 <= result["confidence"] <= 1.0


# ── Regime-to-model mapping ───────────────────────────────────────────────────


def test_regime_model_map_covers_all_known_regimes():
    """Ensure the mapping covers all regimes produced by the AI orchestrator."""
    ai_regimes = {
        "trending_up", "trending_down", "volatile", "ranging", "consolidating",
    }
    assert ai_regimes.issubset(set(_REGIME_MODEL_MAP.keys()))


def test_regime_model_map_targets_valid_models():
    valid_models = {"xgb", "gb", "rf", "linear", "lstm"}
    for regime, model in _REGIME_MODEL_MAP.items():
        assert model in valid_models, f"Unknown model '{model}' for regime '{regime}'"


# ── run_model_export integration ──────────────────────────────────────────────


class TestRunModelExport:
    def test_export_models_run_type_succeeds(self, tmp_path, monkeypatch):
        """run_model_export completes successfully with a pre-trained model."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        for d in ("models", "results", ".trading_state", "datasets/csv"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        # Pre-train a model
        cfg = load_config()
        df = _make_df(80)
        ensemble = QuantumEnsemble(cfg)
        ensemble.train(df, symbol="BTC", save=True)

        # Mock fetcher so we don't need API access
        with patch("src.main.HyperliquidDataFetcher") as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_candles.return_value = df

            from src.main import run_model_export
            rc = run_model_export()
        assert rc == 0

    def test_export_models_saves_csv(self, tmp_path, monkeypatch):
        """run_model_export writes OHLCV CSV files for symbols with models."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        for d in ("models", "results", ".trading_state", "datasets/csv"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        cfg = load_config()
        df = _make_df(80)

        # Only train BTC to limit scope
        ensemble = QuantumEnsemble(cfg)
        ensemble.train(df, symbol="BTC", save=True)

        with patch("src.main.HyperliquidDataFetcher") as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_candles.return_value = df

            from src.main import run_model_export
            run_model_export()

        csv_dir = tmp_path / cfg.data.historical_csv_dir
        csv_files = list(csv_dir.glob("*_ohlcv.csv"))
        assert len(csv_files) > 0, "Expected at least one CSV file to be written"
        # Verify CSV content is valid OHLCV data
        saved_df = pd.read_csv(csv_files[0])
        for col in ("open", "high", "low", "close", "volume"):
            assert col in saved_df.columns

    def test_export_models_main_entry_point(self, tmp_path, monkeypatch):
        """The --run-type export-models CLI option invokes run_model_export."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "test")
        for d in ("models", "results", ".trading_state", "datasets/csv"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        with patch("src.main.run_model_export", return_value=0) as mock_export:
            from src.main import main
            rc = main(["--run-type", "export-models", "--mode", "test"])
        assert rc == 0
        mock_export.assert_called_once()

    def test_export_handles_missing_model_gracefully(self, tmp_path, monkeypatch):
        """run_model_export logs a warning and continues when no model exists."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "test")
        for d in ("models", "results", ".trading_state", "datasets/csv"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        # No models trained – export should complete without error
        with patch("src.main.HyperliquidDataFetcher") as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_candles.return_value = pd.DataFrame()

            from src.main import run_model_export
            rc = run_model_export()
        assert rc == 0


# ── Paper signal cycle uses delegation agent ─────────────────────────────────


class TestDelegationAgentInSignalCycle:
    def test_signal_cycle_caches_regime_for_delegation(self, tmp_path, monkeypatch):
        """After a paper signal cycle, regimes are persisted in the DB cache."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "test")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        for d in ("models", "results", ".trading_state"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)

        df = _make_df(80)

        class DummyEnsemble:
            def __init__(self, config):
                self.cfg = config
                self.feature_cols = []
                self.scaler = None
                self.xgb_model = None
                self.gb_model = None
                self.rf_model = None
                self.linear_model = None
                self.nn_model = None
                self._model_weights: Dict[str, float] = {
                    "xgb": 0.30, "gb": 0.10, "rf": 0.20, "lstm": 0.25, "linear": 0.15
                }

            def load(self, symbol: str) -> bool:
                return True

            def train(self, *a, **kw) -> Dict[str, Any]:
                return {}

            def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
                return {
                    "signal": 0,
                    "confidence": 0.4,
                    "long_prob": 0.0,
                    "short_prob": 0.0,
                    "flat_prob": 1.0,
                    "agreement": 1.0,
                    "model_signals": {},
                }

        monkeypatch.setattr("src.ml_models.QuantumEnsemble", DummyEnsemble)

        from src.config import load_config as lc
        from src.database_manager import DatabaseManager
        from src.main import run_paper_signal

        cfg = lc()
        db_path = tmp_path / cfg.system.state_dir / cfg.system.database_file
        db = DatabaseManager(db_path)

        rc = run_paper_signal()
        assert rc == 0

        # Check that regime cache was persisted
        regimes = db.get_cache("signal:paper:regimes")
        assert isinstance(regimes, dict)
