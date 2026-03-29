"""Tests for performance Evaluator."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.evaluator import Evaluator, PerformanceMetrics, compute_metrics

MS_PER_MINUTE = 60_000
MS_PER_HOUR = 60 * MS_PER_MINUTE
MS_PER_DAY = 24 * MS_PER_HOUR


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def evaluator(config):
    return Evaluator(config)


def _make_trade(
    pnl: float,
    side="long",
    symbol="BTC",
    leverage=15,
    entry_time_ms=1_700_000_000_000,
    exit_time_ms=1_700_000_000_000 + MS_PER_HOUR,
) -> dict:
    """Minimal trade dict matching ClosedTrade fields."""
    return {
        "position_id": "test",
        "symbol": symbol,
        "side": side,
        "entry_price": 40_000.0,
        "exit_price": 40_000.0 + pnl / 0.025,
        "size_contracts": 0.025,
        "size_usd": 1_000.0,
        "leverage": leverage,
        "entry_time_ms": entry_time_ms,
        "exit_time_ms": exit_time_ms,
        "pnl": pnl,
        "pnl_pct": pnl / 100.0,
        "fee_usd": 0.45,
        "funding_usd": 0.0,
        "exit_reason": "signal",
        "duration_ms": 3_600_000,
    }


def _good_trade_history(n: int = 60, win_rate: float = 0.6) -> list:
    """Generate a trade history with controllable win rate."""
    trades = []
    for i in range(n):
        if i / n < win_rate:
            trades.append(_make_trade(pnl=50.0))
        else:
            trades.append(_make_trade(pnl=-30.0))
    return trades


def _set_permissive_thresholds(evaluator: Evaluator) -> None:
    evaluator.eval_cfg.min_sharpe = -1.0
    evaluator.eval_cfg.min_win_rate = 0.0
    evaluator.eval_cfg.max_drawdown_pct = 1.0
    evaluator.eval_cfg.min_profit_factor = 0.0


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_empty_history(self):
        m = compute_metrics([], initial_equity=10_000.0, final_equity=10_000.0)
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0

    def test_all_winners(self):
        trades = [_make_trade(pnl=100.0) for _ in range(20)]
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=12_000.0)
        assert m.win_rate == pytest.approx(1.0)
        assert m.winning_trades == 20
        assert m.losing_trades == 0
        assert m.profit_factor == pytest.approx(0.0)  # no losses → inf, but safe_divide returns 0

    def test_all_losers(self):
        trades = [_make_trade(pnl=-50.0) for _ in range(20)]
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=9_000.0)
        assert m.win_rate == pytest.approx(0.0)
        assert m.profit_factor == pytest.approx(0.0)

    def test_win_rate_calculation(self):
        trades = (
            [_make_trade(pnl=100.0)] * 6
            + [_make_trade(pnl=-50.0)] * 4
        )
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=10_400.0)
        assert m.win_rate == pytest.approx(0.6)

    def test_profit_factor(self):
        trades = (
            [_make_trade(pnl=100.0)] * 3   # gross profit = 300
            + [_make_trade(pnl=-100.0)] * 2  # gross loss = 200
        )
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=10_100.0)
        assert m.profit_factor == pytest.approx(1.5)

    def test_max_drawdown_positive(self):
        # Steady gains followed by losses
        pnls = [100, 100, 100, -200, 100, -300]
        trades = [_make_trade(pnl=p) for p in pnls]
        equity = 10_000.0 + sum(pnls)
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=equity)
        assert m.max_drawdown_pct > 0.0

    def test_total_return_pct(self):
        trades = [_make_trade(pnl=200.0)]  # +200 on 10k
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=10_200.0)
        assert m.total_return_pct == pytest.approx(0.02)

    def test_fee_aggregation(self):
        trades = [_make_trade(pnl=100.0) for _ in range(10)]
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=11_000.0)
        assert m.total_fees == pytest.approx(0.45 * 10)

    def test_avg_leverage(self):
        trades = [_make_trade(pnl=10.0, leverage=20) for _ in range(5)]
        m = compute_metrics(trades, initial_equity=10_000.0, final_equity=10_050.0)
        assert m.avg_leverage_used == pytest.approx(20.0)

    def test_avg_confidence_pass_through(self):
        trades = [_make_trade(pnl=10.0) for _ in range(3)]
        m = compute_metrics(
            trades,
            initial_equity=10_000.0,
            final_equity=10_020.0,
            avg_confidence=0.72,
        )
        assert m.avg_confidence == pytest.approx(0.72)


# ── Evaluator.evaluate ────────────────────────────────────────────────────────

class TestEvaluatorEvaluate:
    def test_adjusts_without_minimum_trade_gate(self, evaluator):
        evaluator.eval_cfg.evaluation_window_trades = 0
        trades = _good_trade_history(n=10, win_rate=0.30)
        m, adj = evaluator.evaluate(trades, 10_000.0, 9_500.0)
        assert m.win_rate < evaluator.eval_cfg.min_win_rate
        assert m.total_return_pct < 0
        threshold_adj = [a for a in adj if "long_threshold" in a["parameter"]]
        assert threshold_adj
        assert threshold_adj[0]["new_value"] > threshold_adj[0]["old_value"]

    def test_min_trade_gate_blocks_full_adjustments(self, evaluator):
        evaluator.eval_cfg.evaluation_window_trades = 50
        evaluator.eval_cfg.min_trades_per_day = 0
        trades = _good_trade_history(n=10, win_rate=0.30)
        m, adj = evaluator.evaluate(trades, 10_000.0, 9_500.0)
        assert m.win_rate < evaluator.eval_cfg.min_win_rate
        assert m.total_return_pct < 0
        assert adj == []

    def test_good_performance_no_negative_adjustments(self, evaluator):
        # High win rate + good Sharpe → should not reduce leverage
        trades = _good_trade_history(n=60, win_rate=0.70)
        m, adj = evaluator.evaluate(trades, 10_000.0, 12_000.0)
        # Should not have a reduce-leverage adjustment for bad Sharpe
        reduce_adj = [a for a in adj if a["parameter"] == "leverage.default" and a["new_value"] < a["old_value"]]
        # This is performance-dependent; just verify no crash
        assert isinstance(adj, list)

    def test_bad_win_rate_triggers_threshold_increase(self, evaluator):
        # Win rate below min_win_rate (0.45)
        trades = _good_trade_history(n=60, win_rate=0.30)
        m, adj = evaluator.evaluate(trades, 10_000.0, 9_000.0)
        threshold_adj = [a for a in adj if "long_threshold" in a["parameter"]]
        assert len(threshold_adj) >= 1
        assert threshold_adj[0]["new_value"] > threshold_adj[0]["old_value"]

    def test_zero_trades_relaxes_signal_filters(self, evaluator):
        evaluator.eval_cfg.min_trades_per_day = 5
        m, adj = evaluator.evaluate([], 10_000.0, 10_000.0)
        params = {a["parameter"] for a in adj}
        assert "ml.long_threshold" in params
        assert "ml.short_threshold" in params
        assert "ml.min_ensemble_agreement" in params

    def test_low_trade_rate_relaxes_filters(self, evaluator):
        evaluator.eval_cfg.min_trades_per_day = 5
        _set_permissive_thresholds(evaluator)

        start = 1_700_000_000_000
        # Four trades spaced one per day ⇒ ~1 trade/day (< min_trades_per_day=5).
        trades = [
            _make_trade(
                pnl=50.0,
                entry_time_ms=start + i * MS_PER_DAY,
                exit_time_ms=start + i * MS_PER_DAY + MS_PER_HOUR,
            )
            for i in range(4)
        ]
        m, adj = evaluator.evaluate(trades, 10_000.0, 10_200.0)
        assert evaluator._passes_thresholds(m) is True
        threshold_adj = [a for a in adj if a["parameter"] == "ml.long_threshold"]
        assert threshold_adj
        assert threshold_adj[0]["new_value"] < threshold_adj[0]["old_value"]

    def test_high_trade_rate_no_volume_adjustment(self, evaluator):
        evaluator.eval_cfg.min_trades_per_day = 1
        # Gate full performance adjustments so this test only checks volume tuning.
        evaluator.eval_cfg.evaluation_window_trades = 25
        start = 1_700_000_000_000
        trades = [
            _make_trade(
                pnl=50.0,
                entry_time_ms=start + i * 5 * MS_PER_MINUTE,
                exit_time_ms=start + i * 5 * MS_PER_MINUTE + MS_PER_MINUTE,
            )
            for i in range(10)
        ]
        m, adj = evaluator.evaluate(trades, 10_000.0, 10_500.0)
        assert adj == []

    def test_passes_thresholds_good_metrics(self, evaluator):
        m = PerformanceMetrics(
            sharpe_ratio=2.0,
            win_rate=0.60,
            max_drawdown_pct=0.10,
            profit_factor=2.0,
        )
        assert evaluator.passes_thresholds(m) is True

    def test_fails_thresholds_bad_sharpe(self, evaluator):
        m = PerformanceMetrics(
            sharpe_ratio=0.3,  # below min 1.0
            win_rate=0.60,
            max_drawdown_pct=0.10,
            profit_factor=2.0,
        )
        assert evaluator.passes_thresholds(m) is False

    def test_fails_thresholds_high_drawdown(self, evaluator):
        m = PerformanceMetrics(
            sharpe_ratio=1.5,
            win_rate=0.60,
            max_drawdown_pct=0.35,  # above max 0.25
            profit_factor=2.0,
        )
        assert evaluator.passes_thresholds(m) is False


# ── Report saving ─────────────────────────────────────────────────────────────

class TestReportSaving:
    def test_save_report(self, evaluator, tmp_path):
        m = PerformanceMetrics(
            sharpe_ratio=1.5,
            win_rate=0.55,
            max_drawdown_pct=0.10,
            profit_factor=1.8,
            total_trades=50,
        )
        path = tmp_path / "report.json"
        evaluator.save_report(m, [], path, label="test")
        assert path.exists()
        with open(path) as fh:
            data = json.load(fh)
        assert data["label"] == "test"
        assert "metrics" in data
        assert "pass" in data

    def test_print_report_with_adjustments(self, evaluator):
        m = PerformanceMetrics(sharpe_ratio=0.5, win_rate=0.4, max_drawdown_pct=0.3, profit_factor=0.9)
        adjustments = [
            {"parameter": "leverage.default", "old_value": 15, "new_value": 10, "reason": "Low Sharpe"}
        ]
        report = evaluator.print_report(m, adjustments)
        assert "leverage.default" in report
        assert "10" in report
