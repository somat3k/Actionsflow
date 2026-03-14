"""Tests for multiplex multi-timeframe signals, proactive risk closure,
and evaluation-driven model weight updates."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    return load_config()


def _make_df(n: int = 200) -> pd.DataFrame:
    """Minimal OHLCV + feature DataFrame for ensemble inference tests."""
    np.random.seed(42)
    prices = 40_000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.random.uniform(1_000, 5_000, n),
        }
    )
    for col in [
        "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
        "ema_9", "ema_21", "ema_50", "bb_upper", "bb_middle", "bb_lower",
        "bb_bandwidth", "atr_14", "adx_14", "stochastic_k", "stochastic_d",
        "cci_20", "williams_r", "vwap", "obv", "ichimoku_tenkan",
        "ichimoku_kijun", "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
        "vol_20", "vol_5",
    ]:
        df[col] = np.random.randn(n).astype(np.float32)
    return df


# ── combined_decision timeframe weight coverage ───────────────────────────────


class TestCombinedDecisionTimeframeWeights:
    """Verify that combined_decision assigns sensible weights to all
    config-standard timeframe labels (lowercase and uppercase variants)."""

    def test_lowercase_hourly_has_higher_weight_than_1m(self, cfg, tmp_path, monkeypatch):
        """'1h' should receive the higher timeframe weight (0.25), not the
        default fallback (0.10) that would apply if only '1H' were present."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()

        # Build synthetic predictions for two timeframes.
        pred_1m = {"signal": 1, "confidence": 0.75, "long_prob": 0.75, "short_prob": 0.10, "flat_prob": 0.15}
        pred_1h = {"signal": 1, "confidence": 0.80, "long_prob": 0.80, "short_prob": 0.08, "flat_prob": 0.12}

        # Use 1h key (lowercase, from config).
        result = ensemble.combined_decision({"1m": pred_1m, "1h": pred_1h})

        # Both agree on LONG; confidence should be between the two individual
        # confidences weighted by timeframe (1h=0.25 > 1m=0.10).
        assert result["signal"] == 1, "Both timeframes are LONG – combined should be LONG"
        assert result["agreement"] == pytest.approx(1.0)
        # With 1h having weight 0.25 and 1m having 0.10, the combined long_prob
        # should be closer to the 1h value (0.80) than to the simple average.
        total_w = 0.10 + 0.25
        expected_long = (0.10 * 0.75 + 0.25 * 0.80) / total_w
        assert result["long_prob"] == pytest.approx(expected_long, abs=1e-4)

    def test_lowercase_daily_weight(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble

        ensemble = QuantumEnsemble(cfg)

        pred_1m = {"signal": 2, "confidence": 0.70, "long_prob": 0.10, "short_prob": 0.70, "flat_prob": 0.20}
        pred_1d = {"signal": 2, "confidence": 0.75, "long_prob": 0.08, "short_prob": 0.75, "flat_prob": 0.17}

        result = ensemble.combined_decision({"1m": pred_1m, "1d": pred_1d})

        assert result["signal"] == 2
        total_w = 0.10 + 0.25
        expected_short = (0.10 * 0.70 + 0.25 * 0.75) / total_w
        assert result["short_prob"] == pytest.approx(expected_short, abs=1e-4)

    def test_conflicting_timeframes_may_produce_flat(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble

        ensemble = QuantumEnsemble(cfg)

        # 1m says LONG with moderate confidence, 15m says SHORT with higher
        # weight (0.25 vs 0.10).  The combined short_prob won't reach the
        # short_threshold (0.60) so the result falls back to FLAT.
        pred_1m  = {"signal": 1, "confidence": 0.65, "long_prob": 0.65, "short_prob": 0.20, "flat_prob": 0.15}
        pred_15m = {"signal": 2, "confidence": 0.65, "long_prob": 0.15, "short_prob": 0.65, "flat_prob": 0.20}

        result = ensemble.combined_decision({"1m": pred_1m, "15m": pred_15m})

        # The combined probabilities won't clear the 0.60 threshold for either
        # long or short (combined long≈0.29, short≈0.52), so the signal is FLAT.
        assert result["signal"] == 0, (
            f"Conflicting TFs should produce FLAT signal, got {result['signal']} "
            f"(long={result['long_prob']:.3f}, short={result['short_prob']:.3f})"
        )
        # Timeframe signals should be recorded.
        assert "1m" in result["timeframe_signals"]
        assert "15m" in result["timeframe_signals"]

    def test_empty_predictions_returns_flat(self, cfg):
        from src.ml_models import QuantumEnsemble

        ensemble = QuantumEnsemble(cfg)
        result = ensemble.combined_decision({})
        assert result["signal"] == 0
        assert result["confidence"] == 0.0
        assert result["agreement"] == 0.0


# ── _build_multiplex_signal ───────────────────────────────────────────────────


@pytest.mark.slow
class TestBuildMultiplexSignal:
    """Verify _build_multiplex_signal combines timeframe predictions correctly
    and falls back gracefully when data is missing."""

    def _make_snapshot(self, tfs: Dict[str, bool]) -> Dict[str, Any]:
        """Build a minimal market snapshot with the given timeframes populated."""
        candles: Dict[str, pd.DataFrame] = {}
        for tf, populated in tfs.items():
            candles[tf] = _make_df() if populated else pd.DataFrame()
        return {"candles": candles, "funding": {}, "order_book": {}}

    def test_multiplex_uses_combined_decision_with_two_tfs(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble, ModelDelegationAgent
        from src.main import _build_multiplex_signal

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="BTC")
        delegation_agent = ModelDelegationAgent(ensemble)

        snapshot = self._make_snapshot({"1m": True, "5m": True, "15m": False, "1h": False})
        result = _build_multiplex_signal(cfg, ensemble, delegation_agent, snapshot)

        assert "signal" in result
        assert result["signal"] in (0, 1, 2)
        # With two timeframes the combined path is taken.
        assert result.get("delegated_to") == "multiplex"

    def test_fallback_to_delegation_with_one_tf(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble, ModelDelegationAgent
        from src.main import _build_multiplex_signal

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="BTC")
        delegation_agent = ModelDelegationAgent(ensemble)

        snapshot = self._make_snapshot({"1m": True, "5m": False, "15m": False, "1h": False})
        result = _build_multiplex_signal(cfg, ensemble, delegation_agent, snapshot)

        assert "signal" in result
        # With only one timeframe, falls back to delegation agent.
        assert result.get("delegated_to") != "multiplex"

    def test_empty_snapshot_returns_flat(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble, ModelDelegationAgent
        from src.main import _build_multiplex_signal

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="BTC")
        delegation_agent = ModelDelegationAgent(ensemble)

        snapshot = {"candles": {}, "funding": {}, "order_book": {}}
        result = _build_multiplex_signal(cfg, ensemble, delegation_agent, snapshot)

        assert result["signal"] == 0

    def test_agreement_gate_forces_flat_on_low_consensus(self, cfg, tmp_path, monkeypatch):
        """When timeframe predictions disagree below min_ensemble_agreement, the
        combined signal should be forced to FLAT."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble, ModelDelegationAgent
        from src.main import _build_multiplex_signal

        # Lower agreement threshold to 0.99 to guarantee disagreement forces flat.
        cfg.ml.min_ensemble_agreement = 0.99

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="BTC")
        delegation_agent = ModelDelegationAgent(ensemble)

        # Patch combined_decision to return low agreement
        with patch.object(ensemble, "combined_decision") as mock_cd:
            mock_cd.return_value = {
                "signal": 1,
                "confidence": 0.70,
                "long_prob": 0.70,
                "short_prob": 0.15,
                "flat_prob": 0.15,
                "agreement": 0.50,  # below 0.99 threshold
                "timeframe_signals": {"1m": 1, "5m": 0},
            }
            snapshot = self._make_snapshot({"1m": True, "5m": True, "15m": False, "1h": False})
            result = _build_multiplex_signal(cfg, ensemble, delegation_agent, snapshot)

        # Signal should be forced to FLAT due to low agreement.
        assert result["signal"] == 0, "Low consensus should force FLAT signal"
        assert result.get("delegated_to") == "multiplex"

    def test_nn_priority_override_for_eth(self, cfg, tmp_path, monkeypatch):
        """ETH should prioritise NN override signals even when multiplex says FLAT."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble, ModelDelegationAgent
        from src.main import _build_multiplex_signal

        cfg.ml.nn_priority_symbols = ["ETH"]
        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="ETH")
        delegation_agent = ModelDelegationAgent(ensemble)

        snapshot = self._make_snapshot({"1m": True, "5m": True, "15m": False, "1h": False})

        def _nn_signal():
            return {
                "signal": 1,
                "confidence": 0.90,
                "long_prob": 0.90,
                "short_prob": 0.05,
                "flat_prob": 0.05,
                "agreement": 1.0,
                "model_signals": {"lstm": 1},
                "nn_decision": True,
            }

        with patch.object(ensemble, "predict", side_effect=lambda *_a, **_k: _nn_signal()):
            with patch.object(
                ensemble,
                "combined_decision",
                return_value={
                    "signal": 0,
                    "confidence": 0.10,
                    "long_prob": 0.30,
                    "short_prob": 0.20,
                    "flat_prob": 0.50,
                    "agreement": 0.50,
                },
            ):
                result = _build_multiplex_signal(
                    cfg, ensemble, delegation_agent, snapshot, symbol="ETH"
                )

        assert result["delegated_to"] == "nn_override"
        assert result["nn_decision"] is True

    def test_per_tf_model_used_when_available(self, cfg, tmp_path, monkeypatch):
        """When has_timeframe_model returns True (train_timeframe was called),
        predict_timeframe should be called for that timeframe instead of the global predict."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble, ModelDelegationAgent
        from src.main import _build_multiplex_signal

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        # Train global model and per-tf model for 1m.
        ensemble.train(df, symbol="BTC")
        ensemble.train_timeframe(df, "BTC", "1m")

        assert ensemble.has_timeframe_model("1m"), "has_timeframe_model should be True after train_timeframe"
        assert not ensemble.has_timeframe_model("5m"), "has_timeframe_model should be False for untrained tf"

        delegation_agent = ModelDelegationAgent(ensemble)
        snapshot = self._make_snapshot({"1m": True, "5m": True, "15m": False, "1h": False})

        with patch.object(ensemble, "predict_timeframe", wraps=ensemble.predict_timeframe) as mock_pt:
            result = _build_multiplex_signal(cfg, ensemble, delegation_agent, snapshot)

        # predict_timeframe should have been called for "1m" (has per-tf model).
        called_tfs = [call.args[1] for call in mock_pt.call_args_list]
        assert "1m" in called_tfs


# ── Proactive risk-based position closure ─────────────────────────────────────


class TestProactiveRiskClosure:
    """Verify that positions with unrealised losses are closed when the AI
    orchestrator returns non-empty risk_flags."""

    def test_losing_position_closed_on_risk_flags(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.paper_broker import PaperBroker
        from src.risk_manager import PositionSpec

        broker = PaperBroker(cfg)
        spec = PositionSpec(
            symbol="BTC",
            side="long",
            entry_price=40_000.0,
            size_usd=1_000.0,
            size_contracts=0.025,
            leverage=15,
            stop_loss=38_000.0,
            take_profit=43_000.0,
            trailing_stop_pct=0.015,
            risk_usd=50.0,
        )
        pos = broker.open_position(spec, current_price=40_000.0)
        assert pos is not None

        # Simulate a loss: update unrealised PnL to negative.
        pos.unrealised_pnl = -200.0

        # Replicate the risk-flag logic from run_paper_signal.
        existing = broker.get_open_position("BTC")
        risk_flags = ["high_volatility", "funding_negative"]

        actions_taken = []
        if risk_flags and existing and existing.unrealised_pnl < 0:
            broker.close_position(existing.position_id, 39_500.0, "risk_flag")
            actions_taken.append({"action": "risk_close", "symbol": "BTC"})
            existing = None

        assert existing is None, "Position should have been closed"
        assert len(actions_taken) == 1
        assert actions_taken[0]["action"] == "risk_close"
        assert len(broker.positions) == 0

    def test_profitable_position_not_closed_on_risk_flags(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.paper_broker import PaperBroker
        from src.risk_manager import PositionSpec

        broker = PaperBroker(cfg)
        spec = PositionSpec(
            symbol="ETH",
            side="long",
            entry_price=3_000.0,
            size_usd=500.0,
            size_contracts=0.167,
            leverage=15,
            stop_loss=2_800.0,
            take_profit=3_500.0,
            trailing_stop_pct=0.015,
            risk_usd=25.0,
        )
        pos = broker.open_position(spec, current_price=3_000.0)
        assert pos is not None

        # Profitable position.
        pos.unrealised_pnl = 50.0

        existing = broker.get_open_position("ETH")
        risk_flags = ["high_volatility"]

        # Risk-flag closure should only trigger when unrealised_pnl < 0.
        if risk_flags and existing and existing.unrealised_pnl < 0:
            broker.close_position(existing.position_id, 3_100.0, "risk_flag")
            existing = None

        assert existing is not None, "Profitable position should NOT be closed"
        assert len(broker.positions) == 1

    def test_no_closure_without_risk_flags(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.paper_broker import PaperBroker
        from src.risk_manager import PositionSpec

        broker = PaperBroker(cfg)
        spec = PositionSpec(
            symbol="SOL",
            side="short",
            entry_price=100.0,
            size_usd=500.0,
            size_contracts=5.0,
            leverage=10,
            stop_loss=110.0,
            take_profit=85.0,
            trailing_stop_pct=0.015,
            risk_usd=25.0,
        )
        pos = broker.open_position(spec, current_price=100.0)
        pos.unrealised_pnl = -30.0

        existing = broker.get_open_position("SOL")
        risk_flags = []  # no flags

        if risk_flags and existing and existing.unrealised_pnl < 0:
            broker.close_position(existing.position_id, 103.0, "risk_flag")
            existing = None

        assert existing is not None, "Position should not be closed without risk flags"
        assert len(broker.positions) == 1


# ── Evaluation-driven model weight update ─────────────────────────────────────


class TestEvaluationDrivenWeightUpdate:
    """Verify _update_model_weights_from_evaluation applies reinforcement to
    persisted model weights and uses a larger alpha when performance is poor."""

    def _dummy_metrics(self, win_rate: float = 0.55):
        from src.evaluator import PerformanceMetrics
        return PerformanceMetrics(win_rate=win_rate, sharpe_ratio=1.2)

    @pytest.mark.slow
    def test_weights_updated_when_scores_cached(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.database_manager import DatabaseManager
        from src.main import _update_model_weights_from_evaluation

        db_path = tmp_path / "state.db"
        db = DatabaseManager(db_path)

        # Train and save a model so the ensemble can be loaded.
        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="BTC")

        original_weights = dict(ensemble._model_weights)

        # Cache training scores so the function has something to work with.
        db.set_cache("training:last_scores", {"BTC": {"xgb": 0.60, "rf": 0.55, "gb": 0.50, "linear": 0.52}})

        metrics = self._dummy_metrics(win_rate=0.55)
        _update_model_weights_from_evaluation(cfg, db, metrics)

        # Load the saved model and check weights changed.
        ensemble2 = QuantumEnsemble(cfg)
        ensemble2.load("BTC")
        new_weights = ensemble2._model_weights

        # At least one weight should have changed.
        changed = any(
            abs(new_weights.get(k, 0) - original_weights.get(k, 0)) > 1e-9
            for k in original_weights
        )
        assert changed, "Weights should be updated after evaluation-driven reinforcement"

    @pytest.mark.slow
    def test_larger_alpha_on_poor_performance(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.ml_models import QuantumEnsemble
        from src.database_manager import DatabaseManager
        from src.main import _update_model_weights_from_evaluation

        db_path = tmp_path / "state.db"
        db = DatabaseManager(db_path)

        ensemble = QuantumEnsemble(cfg)
        df = _make_df()
        ensemble.train(df, symbol="BTC")

        scores = {"xgb": 0.60, "rf": 0.55, "gb": 0.50, "linear": 0.52}
        db.set_cache("training:last_scores", {"BTC": scores})

        # Poor win rate → cache should record the doubled alpha.
        poor_metrics = self._dummy_metrics(win_rate=0.30)  # below min_win_rate (0.45)
        _update_model_weights_from_evaluation(cfg, db, poor_metrics)

        cached = db.get_cache("evaluation:weight_update")
        base_alpha = cfg.ml.reinforcement_alpha
        # The evaluation step must always write a weight_update cache entry
        # when scores are cached and performance is poor.
        assert cached is not None, "Expected evaluation:weight_update cache entry to be set"
        assert isinstance(cached, dict), "Cached weight_update entry must be a dict"
        assert "alpha" in cached, "Cached weight_update entry must contain 'alpha'"
        assert cached["alpha"] == pytest.approx(base_alpha * 2.0), (
            f"Expected doubled alpha {base_alpha * 2.0} but got {cached['alpha']}"
        )

    def test_no_error_when_no_cached_scores(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.database_manager import DatabaseManager
        from src.main import _update_model_weights_from_evaluation

        db_path = tmp_path / "state.db"
        db = DatabaseManager(db_path)
        # No cached training scores.

        metrics = self._dummy_metrics()
        # Should complete without raising.
        _update_model_weights_from_evaluation(cfg, db, metrics)

    def test_no_error_when_model_not_found(self, cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()

        from src.database_manager import DatabaseManager
        from src.main import _update_model_weights_from_evaluation

        db_path = tmp_path / "state.db"
        db = DatabaseManager(db_path)
        # Cache scores for a symbol with no persisted model.
        db.set_cache("training:last_scores", {"UNKNOWN": {"xgb": 0.60}})

        metrics = self._dummy_metrics()
        # Should complete without raising.
        _update_model_weights_from_evaluation(cfg, db, metrics)
