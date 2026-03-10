"""
Dashboard data helpers for the Streamlit UI.
Loads project state from disk and prepares pandas DataFrames for visualization.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import AppConfig
from src.evaluator import PerformanceMetrics, compute_metrics

TRADE_COLUMNS = [
    "position_id",
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "size_contracts",
    "size_usd",
    "leverage",
    "entry_time_ms",
    "exit_time_ms",
    "pnl",
    "pnl_pct",
    "fee_usd",
    "funding_usd",
    "exit_reason",
    "duration_ms",
]

POSITION_COLUMNS = [
    "position_id",
    "symbol",
    "side",
    "entry_price",
    "size_contracts",
    "size_usd",
    "leverage",
    "stop_loss",
    "take_profit",
    "trailing_stop_pct",
    "entry_time_ms",
    "margin_usd",
    "funding_accrued",
    "unrealised_pnl",
    "max_favourable_excursion",
]

MODEL_SCORE_COLUMNS = ["symbol", "xgb", "gb", "rf", "lstm"]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}


def _path_updated_at(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.isoformat()


def load_paper_broker_state(state_dir: Path) -> Dict[str, Any]:
    """Load paper broker state from disk with safe defaults."""
    path = Path(state_dir) / "paper_broker.json"
    state = _load_json(path)
    positions = []
    for pos_id, pos in state.get("positions", {}).items():
        entry = dict(pos)
        entry.setdefault("position_id", pos_id)
        positions.append(entry)
    return {
        "equity": float(state.get("equity", 0.0)),
        "balance": float(state.get("balance", 0.0)),
        "initial_equity": float(state.get("initial_equity", 0.0)),
        "positions": positions,
        "trade_history": state.get("trade_history", []),
        "updated_at": _path_updated_at(path),
    }


def load_evaluation_report(results_dir: Path) -> Dict[str, Any]:
    path = Path(results_dir) / "evaluation_report.json"
    report = _load_json(path)
    report["updated_at"] = _path_updated_at(path)
    return report


def load_training_scores(results_dir: Path) -> Dict[str, Any]:
    path = Path(results_dir) / "training_scores.json"
    scores = _load_json(path)
    scores["updated_at"] = _path_updated_at(path)
    return scores


def build_positions_frame(positions: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(positions)
    if df.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    for col in POSITION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["entry_time"] = pd.to_datetime(df["entry_time_ms"], unit="ms", utc=True, errors="coerce")
    return df


def build_trade_history_frame(trade_history: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(trade_history)
    if df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["entry_time"] = pd.to_datetime(df["entry_time_ms"], unit="ms", utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time_ms"], unit="ms", utc=True, errors="coerce")
    return df


def build_equity_curve(trade_history: List[Dict[str, Any]], initial_equity: float) -> pd.DataFrame:
    df = pd.DataFrame(trade_history)
    if df.empty:
        return pd.DataFrame(columns=["exit_time", "equity", "pnl"])
    if "exit_time_ms" in df.columns:
        df = df.sort_values("exit_time_ms")
    pnl_series = pd.to_numeric(df.get("pnl", pd.Series([0.0] * len(df))), errors="coerce").fillna(0)
    equity = initial_equity + pnl_series.cumsum()
    exit_time = pd.to_datetime(df.get("exit_time_ms"), unit="ms", utc=True, errors="coerce")
    return pd.DataFrame({"exit_time": exit_time, "equity": equity, "pnl": pnl_series})


def build_model_scores_frame(scores: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, model_scores in scores.items():
        if symbol == "updated_at":
            continue
        row = {"symbol": symbol}
        if isinstance(model_scores, dict):
            row.update(model_scores)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=MODEL_SCORE_COLUMNS)
    for col in MODEL_SCORE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def derive_metrics(
    trade_history: List[Dict[str, Any]],
    initial_equity: float,
    final_equity: float,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = report.get("metrics") if report else None
    if isinstance(metrics, dict) and metrics:
        return metrics
    computed: PerformanceMetrics = compute_metrics(trade_history, initial_equity, final_equity)
    return asdict(computed)


def load_dashboard_state(cfg: AppConfig) -> Dict[str, Any]:
    state_dir = Path(cfg.system.state_dir)
    results_dir = Path(cfg.system.results_dir)

    broker_state = load_paper_broker_state(state_dir)
    report = load_evaluation_report(results_dir)
    scores = load_training_scores(results_dir)

    initial_equity = broker_state.get("initial_equity") or cfg.trading.initial_equity
    final_equity = broker_state.get("equity") or initial_equity
    trade_history = broker_state.get("trade_history", [])

    metrics = derive_metrics(trade_history, initial_equity, final_equity, report)

    return {
        "broker_state": broker_state,
        "evaluation_report": report,
        "training_scores": scores,
        "metrics": metrics,
        "adjustments": report.get("adjustments", []) if report else [],
        "metrics_pass": report.get("pass") if report else None,
        "positions_df": build_positions_frame(broker_state.get("positions", [])),
        "trades_df": build_trade_history_frame(trade_history),
        "equity_curve_df": build_equity_curve(trade_history, initial_equity),
        "model_scores_df": build_model_scores_frame(scores),
    }

