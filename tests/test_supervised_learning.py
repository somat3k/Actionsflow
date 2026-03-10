"""Tests for the supervised learning module."""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.supervised_learning import SupervisedLearningModule, LearningAdjustment


def test_record_prediction_tracks_accuracy():
    cfg = load_config()
    slm = SupervisedLearningModule(cfg)

    slm.record_prediction("BTC", predicted_signal=1, actual_outcome=1)
    slm.record_prediction("BTC", predicted_signal=1, actual_outcome=2)
    slm.record_prediction("BTC", predicted_signal=2, actual_outcome=2)

    assert slm.get_accuracy("BTC") == 2 / 3


def test_accuracy_returns_zero_for_unknown_symbol():
    cfg = load_config()
    slm = SupervisedLearningModule(cfg)
    assert slm.get_accuracy("UNKNOWN") == 0.0


def test_evaluate_tightens_threshold_on_low_winrate():
    cfg = load_config()
    cfg.evaluation.min_win_rate = 0.50
    old_threshold = cfg.ml.long_threshold
    slm = SupervisedLearningModule(cfg)

    trades = [{"pnl": -1.0}] * 20
    metrics = {"win_rate": 0.30, "sharpe_ratio": 0.5, "max_drawdown_pct": 0.10}
    adjustments = slm.evaluate_and_adjust(trades, metrics)

    assert len(adjustments) >= 1
    thresh_adj = [a for a in adjustments if "threshold" in a.parameter]
    assert len(thresh_adj) >= 1
    assert cfg.ml.long_threshold > old_threshold


def test_evaluate_relaxes_threshold_on_strong_performance():
    cfg = load_config()
    cfg.ml.long_threshold = 0.70
    cfg.ml.short_threshold = 0.70
    slm = SupervisedLearningModule(cfg)

    trades = [{"pnl": 10.0}] * 20
    metrics = {"win_rate": 0.70, "sharpe_ratio": 2.0, "max_drawdown_pct": 0.05}
    adjustments = slm.evaluate_and_adjust(trades, metrics)

    thresh_adj = [a for a in adjustments if "threshold" in a.parameter]
    assert len(thresh_adj) >= 1
    assert cfg.ml.long_threshold < 0.70


def test_save_and_load_state(tmp_path):
    cfg = load_config()
    slm = SupervisedLearningModule(cfg)
    slm.record_prediction("BTC", 1, 1)
    slm.record_prediction("ETH", 2, 0)
    path = tmp_path / "sl_state.json"
    slm.save_state(path)

    slm2 = SupervisedLearningModule(cfg)
    slm2.load_state(path)
    assert slm2.get_accuracy("BTC") == 1.0
    assert slm2.get_accuracy("ETH") == 0.0


def test_no_adjustments_with_insufficient_trades():
    cfg = load_config()
    slm = SupervisedLearningModule(cfg)
    metrics = {"win_rate": 0.30, "sharpe_ratio": 0.5, "max_drawdown_pct": 0.10}
    adjustments = slm.evaluate_and_adjust([], metrics)
    assert adjustments == []
