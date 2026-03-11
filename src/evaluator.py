"""
Quantum Trading System – Evaluator
Computes comprehensive performance metrics from trade history and triggers
automatic strategy adjustments when thresholds are breached.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig, EvaluationConfig
from src.utils import fmt_pct, fmt_summary, get_logger, safe_divide

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 365


@dataclass
class PerformanceMetrics:
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl_usd: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    avg_trade_duration_hours: float = 0.0
    trades_per_day: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_leverage_used: float = 0.0
    avg_confidence: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    final_equity: float = 0.0
    initial_equity: float = 0.0
    # Extended metrics for quantum trading
    accuracy: float = 0.0
    equity_growth_pct: float = 0.0
    num_positions: int = 0
    gemini_answer_time_avg_s: float = 0.0
    action_time_avg_s: float = 0.0
    # Stabs / pierces: short-window early-warning flags
    stab_alert: bool = False          # True when short-window metrics breach stab thresholds
    pierce_alert: bool = False        # True when Sharpe pierces the pierce threshold


def compute_metrics(
    trade_history: List[Dict[str, Any]],
    initial_equity: float,
    final_equity: float,
    *,
    num_positions: int = 0,
    gemini_answer_time_avg_s: float = 0.0,
    action_time_avg_s: float = 0.0,
) -> PerformanceMetrics:
    """Compute full performance metrics from a list of closed trade dicts.

    Optional keyword arguments ``num_positions``, ``gemini_answer_time_avg_s``
    and ``action_time_avg_s`` are forwarded directly into the returned
    :class:`PerformanceMetrics` since they are measured externally by the
    signal cycle rather than derivable from trade history alone.
    """
    m = PerformanceMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        num_positions=num_positions,
        gemini_answer_time_avg_s=gemini_answer_time_avg_s,
        action_time_avg_s=action_time_avg_s,
    )

    if not trade_history:
        return m

    trades_df = pd.DataFrame(trade_history)

    pnls = trades_df["pnl"].values.astype(float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    m.total_trades = len(pnls)
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = safe_divide(m.winning_trades, m.total_trades)
    m.gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    m.gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    m.profit_factor = safe_divide(m.gross_profit, m.gross_loss, default=0.0)
    m.avg_trade_pnl_usd = float(np.mean(pnls))
    m.avg_win_usd = float(np.mean(wins)) if len(wins) > 0 else 0.0
    m.avg_loss_usd = float(np.mean(losses)) if len(losses) > 0 else 0.0

    if "fee_usd" in trades_df.columns:
        m.total_fees = float(trades_df["fee_usd"].sum())

    if "leverage" in trades_df.columns:
        m.avg_leverage_used = float(trades_df["leverage"].mean())

    if "duration_ms" in trades_df.columns:
        m.avg_trade_duration_hours = float(
            trades_df["duration_ms"].mean() / 3_600_000
        )

    # ── Return & equity curve ─────────────────────────────────────────────────
    m.total_return_pct = safe_divide(final_equity - initial_equity, initial_equity)
    m.equity_growth_pct = m.total_return_pct
    m.accuracy = m.win_rate  # accuracy = win rate in trading context

    if "entry_time_ms" in trades_df.columns and "exit_time_ms" in trades_df.columns:
        total_ms = (
            trades_df["exit_time_ms"].max() - trades_df["entry_time_ms"].min()
        )
        total_days = total_ms / 86_400_000 if total_ms > 0 else 1
        m.trades_per_day = m.total_trades / total_days if total_days > 0 else 0
        years = total_days / TRADING_DAYS_PER_YEAR
        with np.errstate(over="ignore"):
            ann_ret = (
                (1 + m.total_return_pct) ** (1 / years) - 1
                if years > 0
                else m.total_return_pct
            )
        m.annualized_return_pct = float(ann_ret) if np.isfinite(ann_ret) else m.total_return_pct

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cumulative_pnl = np.cumsum(pnls)
    equity_curve = initial_equity + cumulative_pnl
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    m.max_drawdown_pct = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0
    m.calmar_ratio = safe_divide(
        m.annualized_return_pct, m.max_drawdown_pct, default=0.0
    )

    # ── Risk-adjusted ratios ──────────────────────────────────────────────────
    if len(pnls) >= 2:
        returns_pct = pnls / initial_equity
        mean_ret = np.mean(returns_pct)
        std_ret = np.std(returns_pct)
        downside_ret = returns_pct[returns_pct < 0]
        downside_std = np.std(downside_ret) if len(downside_ret) >= 2 else std_ret

        rf_per_trade = 0.04 / (TRADING_DAYS_PER_YEAR * 24)  # ~4% annual risk-free
        m.sharpe_ratio = safe_divide(mean_ret - rf_per_trade, std_ret) * np.sqrt(
            len(returns_pct)
        )
        m.sortino_ratio = safe_divide(mean_ret - rf_per_trade, downside_std) * np.sqrt(
            len(returns_pct)
        )

    return m


# ── Evaluator ──────────────────────────────────────────────────────────────────

class Evaluator:
    """Evaluates trading performance and emits concrete adjustment recommendations."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.eval_cfg: EvaluationConfig = config.evaluation

    def evaluate(
        self,
        trade_history: List[Dict[str, Any]],
        initial_equity: float,
        final_equity: float,
        *,
        num_positions: int = 0,
        gemini_answer_time_avg_s: float = 0.0,
        action_time_avg_s: float = 0.0,
    ) -> Tuple[PerformanceMetrics, List[Dict[str, Any]]]:
        """
        Compute metrics and return (metrics, adjustment_recommendations).
        Each adjustment is a dict: {parameter, old_value, new_value, reason}.

        Stabs/pierces are evaluated on the most recent short window of trades
        in addition to the full-window standard evaluation.
        """
        metrics = compute_metrics(
            trade_history,
            initial_equity,
            final_equity,
            num_positions=num_positions,
            gemini_answer_time_avg_s=gemini_answer_time_avg_s,
            action_time_avg_s=action_time_avg_s,
        )

        # ── Stabs / pierces: short-window early-warning check ─────────────
        if self.eval_cfg.stabs_enabled and trade_history:
            window = self.eval_cfg.stabs_window_trades
            recent_trades = trade_history[-window:]
            if len(recent_trades) >= 2:
                # Compute self-consistent equity values for just this slice so
                # that compute_metrics sees the correct equity curve for the
                # short window rather than the full-run bookends.
                pre_window_trades = trade_history[: len(trade_history) - len(recent_trades)]
                pre_window_pnl = sum(t.get("pnl", 0.0) for t in pre_window_trades)
                recent_window_pnl = sum(t.get("pnl", 0.0) for t in recent_trades)
                short_initial_equity = initial_equity + pre_window_pnl
                short_final_equity = short_initial_equity + recent_window_pnl

                short_metrics = compute_metrics(
                    recent_trades, short_initial_equity, short_final_equity
                )
                if (
                    short_metrics.win_rate < self.eval_cfg.stabs_min_win_rate
                    or short_metrics.max_drawdown_pct > self.eval_cfg.stabs_max_drawdown_pct
                ):
                    metrics.stab_alert = True
                    log.warning(
                        "STAB alert: short-window WR=%.2f%% DD=%.2f%%",
                        short_metrics.win_rate * 100,
                        short_metrics.max_drawdown_pct * 100,
                    )
                if short_metrics.sharpe_ratio < self.eval_cfg.stabs_pierce_sharpe_threshold:
                    metrics.pierce_alert = True
                    log.warning(
                        "PIERCE alert: short-window Sharpe=%.3f below threshold %.3f",
                        short_metrics.sharpe_ratio,
                        self.eval_cfg.stabs_pierce_sharpe_threshold,
                    )

        adjustments = []

        if not self.eval_cfg.auto_adjust_enabled:
            return metrics, adjustments

        if len(trade_history) < self.eval_cfg.evaluation_window_trades:
            log.info(
                "Not enough trades for evaluation (%d < %d)",
                len(trade_history),
                self.eval_cfg.evaluation_window_trades,
            )
            return metrics, adjustments

        adjustments = self._generate_adjustments(metrics)
        return metrics, adjustments

    def print_report(
        self,
        metrics: PerformanceMetrics,
        adjustments: List[Dict[str, Any]],
    ) -> str:
        """Format a human-readable performance report."""
        report_lines = [fmt_summary(asdict(metrics))]

        if metrics.stab_alert:
            report_lines.append("\n  ⚠️  STAB ALERT: short-window performance deterioration detected.")
        if metrics.pierce_alert:
            report_lines.append(
                "\n  ⚠️  PIERCE ALERT: short-window Sharpe pierced the warning threshold."
            )

        if adjustments:
            report_lines.append("\n  AUTO-ADJUSTMENTS RECOMMENDED:")
            for adj in adjustments:
                report_lines.append(
                    f"  • {adj['parameter']}: {adj['old_value']} → {adj['new_value']}"
                    f" ({adj['reason']})"
                )
        else:
            report_lines.append("\n  No adjustments required.")

        report = "\n".join(report_lines)
        log.info("\n%s", report)
        return report

    def save_report(
        self,
        metrics: PerformanceMetrics,
        adjustments: List[Dict[str, Any]],
        path: Path,
        label: str = "",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "label": label,
            "metrics": asdict(metrics),
            "adjustments": adjustments,
            "pass": self._passes_thresholds(metrics),
        }
        with open(path, "w") as fh:
            json.dump(output, fh, indent=2)
        log.info("Evaluation report saved to %s", path)

    def passes_thresholds(self, metrics: PerformanceMetrics) -> bool:
        return self._passes_thresholds(metrics)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _passes_thresholds(self, m: PerformanceMetrics) -> bool:
        return (
            m.sharpe_ratio >= self.eval_cfg.min_sharpe
            and m.win_rate >= self.eval_cfg.min_win_rate
            and m.max_drawdown_pct <= self.eval_cfg.max_drawdown_pct
            and m.profit_factor >= self.eval_cfg.min_profit_factor
        )

    def _generate_adjustments(
        self, m: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        adjustments = []
        lev_cfg = self.cfg.trading.leverage

        # Sharpe too low → reduce leverage
        if m.sharpe_ratio < self.eval_cfg.min_sharpe:
            old_lev = lev_cfg.default
            new_lev = max(lev_cfg.min, old_lev - 5)
            adjustments.append(
                {
                    "parameter": "leverage.default",
                    "old_value": old_lev,
                    "new_value": new_lev,
                    "reason": f"Sharpe {m.sharpe_ratio:.2f} below threshold {self.eval_cfg.min_sharpe}",
                }
            )

        # Drawdown too high → reduce leverage and max position size
        if m.max_drawdown_pct > self.eval_cfg.max_drawdown_pct:
            old_lev = lev_cfg.default
            new_lev = max(lev_cfg.min, old_lev - 5)
            adjustments.append(
                {
                    "parameter": "leverage.default",
                    "old_value": old_lev,
                    "new_value": new_lev,
                    "reason": f"Max drawdown {fmt_pct(m.max_drawdown_pct)} exceeds {fmt_pct(self.eval_cfg.max_drawdown_pct)}",
                }
            )

        # Win rate low → tighten signal thresholds
        if m.win_rate < self.eval_cfg.min_win_rate:
            old_thresh = self.cfg.ml.long_threshold
            new_thresh = min(0.80, old_thresh + 0.05)
            adjustments.append(
                {
                    "parameter": "ml.signals.long_threshold",
                    "old_value": old_thresh,
                    "new_value": new_thresh,
                    "reason": f"Win rate {fmt_pct(m.win_rate)} below threshold {fmt_pct(self.eval_cfg.min_win_rate)}",
                }
            )

        # Profit factor low → widen take profit
        if m.profit_factor < self.eval_cfg.min_profit_factor:
            old_mult = self.cfg.trading.risk.take_profit_atr_multiplier
            new_mult = min(5.0, old_mult + 0.5)
            adjustments.append(
                {
                    "parameter": "risk.take_profit_atr_multiplier",
                    "old_value": old_mult,
                    "new_value": new_mult,
                    "reason": f"Profit factor {m.profit_factor:.2f} below threshold {self.eval_cfg.min_profit_factor}",
                }
            )

        # Good performance → consider increasing leverage
        if (
            m.sharpe_ratio > self.eval_cfg.min_sharpe * 1.5
            and m.max_drawdown_pct < self.eval_cfg.max_drawdown_pct * 0.5
            and m.win_rate > self.eval_cfg.min_win_rate * 1.2
        ):
            old_lev = lev_cfg.default
            new_lev = min(lev_cfg.max, old_lev + 5)
            if new_lev != old_lev:
                adjustments.append(
                    {
                        "parameter": "leverage.default",
                        "old_value": old_lev,
                        "new_value": new_lev,
                        "reason": "Strong performance – elevating leverage",
                    }
                )

        return adjustments
