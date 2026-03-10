"""Tests for dashboard data helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.dashboard_data import (
    build_model_scores_frame,
    load_dashboard_state,
    load_paper_broker_state,
)


def test_load_paper_broker_state_defaults(tmp_path: Path) -> None:
    state = load_paper_broker_state(tmp_path)
    assert state["trade_history"] == []
    assert state["positions"] == []
    assert state["equity"] == 0.0


def test_load_dashboard_state_prefers_report_metrics(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    results_dir = tmp_path / "results"
    state_dir.mkdir()
    results_dir.mkdir()

    broker_state = {
        "equity": 12_000.0,
        "balance": 11_800.0,
        "initial_equity": 10_000.0,
        "positions": {},
        "trade_history": [
            {
                "position_id": "abc",
                "symbol": "BTC",
                "side": "long",
                "entry_price": 40_000.0,
                "exit_price": 42_000.0,
                "size_contracts": 0.1,
                "size_usd": 4_000.0,
                "leverage": 10,
                "entry_time_ms": 1_700_000_000_000,
                "exit_time_ms": 1_700_003_600_000,
                "pnl": 200.0,
                "pnl_pct": 0.05,
                "fee_usd": 2.0,
                "funding_usd": 0.0,
                "exit_reason": "signal",
                "duration_ms": 3_600_000,
            }
        ],
    }
    (state_dir / "paper_broker.json").write_text(json.dumps(broker_state))

    report = {"metrics": {"sharpe_ratio": 1.5, "total_return_pct": 0.2}, "adjustments": []}
    (results_dir / "evaluation_report.json").write_text(json.dumps(report))

    cfg = load_config()
    cfg.system.state_dir = str(state_dir)
    cfg.system.results_dir = str(results_dir)

    state = load_dashboard_state(cfg)
    assert state["metrics"]["sharpe_ratio"] == 1.5
    assert len(state["trades_df"]) == 1


def test_build_model_scores_frame_handles_dict() -> None:
    scores = {
        "BTC": {"xgb": 0.7, "gb": 0.6, "rf": 0.5, "lstm": 0.4},
        "ETH": {"xgb": 0.65, "gb": 0.55, "rf": 0.45, "lstm": 0.35},
    }
    df = build_model_scores_frame(scores)
    assert set(df["symbol"]) == {"BTC", "ETH"}
    assert "xgb" in df.columns

