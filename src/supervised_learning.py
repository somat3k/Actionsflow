"""
Quantum Trading System – Live-Supervised Learning Module

Evaluates recent trading performance and dynamically adjusts model parameters
and trading strategy based on live feedback.  Acts as a bridge between the
Evaluator and the ML ensemble, applying data-driven elevations to improve
signal quality over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.utils import get_logger, safe_divide

log = get_logger(__name__)


@dataclass
class LearningAdjustment:
    """Single parameter adjustment derived from live evaluation."""
    parameter: str
    old_value: Any
    new_value: Any
    reason: str


class SupervisedLearningModule:
    """Live-supervised learning: evaluates recent trading outcomes and
    dynamically adjusts strategy parameters.

    The module:
    1. Tracks prediction accuracy per symbol and timeframe.
    2. Adjusts signal thresholds based on rolling win-rate accuracy.
    3. Re-weights model ensemble based on individual model accuracy.
    4. Elevates training data by surfacing recent market regimes.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._accuracy_log: Dict[str, List[float]] = {}
        self._adjustment_history: List[LearningAdjustment] = []

    def record_prediction(
        self, symbol: str, predicted_signal: int, actual_outcome: int
    ) -> None:
        """Record a prediction result for accuracy tracking."""
        key = symbol
        if key not in self._accuracy_log:
            self._accuracy_log[key] = []
        correct = 1.0 if predicted_signal == actual_outcome else 0.0
        self._accuracy_log[key].append(correct)
        # Keep last 200 records
        if len(self._accuracy_log[key]) > 200:
            self._accuracy_log[key] = self._accuracy_log[key][-200:]

    def get_accuracy(self, symbol: str) -> float:
        """Rolling accuracy for a symbol."""
        records = self._accuracy_log.get(symbol, [])
        if not records:
            return 0.0
        return sum(records) / len(records)

    def evaluate_and_adjust(
        self,
        trade_history: List[Dict[str, Any]],
        current_metrics: Dict[str, Any],
    ) -> List[LearningAdjustment]:
        """Analyse recent performance and return recommended adjustments."""
        adjustments: List[LearningAdjustment] = []

        win_rate = current_metrics.get("win_rate", 0.0)
        sharpe = current_metrics.get("sharpe_ratio", 0.0)
        drawdown = current_metrics.get("max_drawdown_pct", 0.0)

        # Tighten thresholds if accuracy is low
        if win_rate < self.cfg.evaluation.min_win_rate and len(trade_history) >= 10:
            old_thresh = self.cfg.ml.long_threshold
            new_thresh = min(0.80, old_thresh + 0.02)
            if new_thresh != old_thresh:
                adj = LearningAdjustment(
                    parameter="ml.long_threshold",
                    old_value=old_thresh,
                    new_value=new_thresh,
                    reason=f"Win rate {win_rate:.2%} below {self.cfg.evaluation.min_win_rate:.2%}",
                )
                adjustments.append(adj)
                self.cfg.ml.long_threshold = new_thresh
                self.cfg.ml.short_threshold = new_thresh

        # Relax thresholds if accuracy is very good
        if win_rate > 0.65 and sharpe > 1.5:
            old_thresh = self.cfg.ml.long_threshold
            new_thresh = max(0.50, old_thresh - 0.02)
            if new_thresh != old_thresh:
                adj = LearningAdjustment(
                    parameter="ml.long_threshold",
                    old_value=old_thresh,
                    new_value=new_thresh,
                    reason=f"Strong performance (WR={win_rate:.2%}, Sharpe={sharpe:.2f})",
                )
                adjustments.append(adj)
                self.cfg.ml.long_threshold = new_thresh
                self.cfg.ml.short_threshold = new_thresh

        # Reduce ensemble agreement if signal quality is high
        if win_rate > 0.60 and drawdown < 0.10:
            old_agree = self.cfg.ml.min_ensemble_agreement
            new_agree = max(0.40, old_agree - 0.05)
            if new_agree != old_agree:
                adj = LearningAdjustment(
                    parameter="ml.min_ensemble_agreement",
                    old_value=old_agree,
                    new_value=new_agree,
                    reason="Good accuracy allows lower agreement threshold",
                )
                adjustments.append(adj)
                self.cfg.ml.min_ensemble_agreement = new_agree

        # Increase ensemble agreement if too many bad trades
        if win_rate < 0.40 and len(trade_history) >= 20:
            old_agree = self.cfg.ml.min_ensemble_agreement
            new_agree = min(0.80, old_agree + 0.05)
            if new_agree != old_agree:
                adj = LearningAdjustment(
                    parameter="ml.min_ensemble_agreement",
                    old_value=old_agree,
                    new_value=new_agree,
                    reason=f"Low win rate {win_rate:.2%} requires higher agreement",
                )
                adjustments.append(adj)
                self.cfg.ml.min_ensemble_agreement = new_agree

        self._adjustment_history.extend(adjustments)
        if adjustments:
            log.info(
                "Supervised learning adjustments: %s",
                [(a.parameter, a.new_value) for a in adjustments],
            )
        return adjustments

    def save_state(self, path: Path) -> None:
        """Persist learning state."""
        state = {
            "accuracy_log": self._accuracy_log,
            "adjustment_history": [
                {
                    "parameter": a.parameter,
                    "old_value": a.old_value,
                    "new_value": a.new_value,
                    "reason": a.reason,
                }
                for a in self._adjustment_history[-100:]
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2, default=str)

    def load_state(self, path: Path) -> None:
        """Restore learning state."""
        if not path.exists():
            return
        try:
            with open(path) as fh:
                state = json.load(fh)
            self._accuracy_log = state.get("accuracy_log", {})
            for entry in state.get("adjustment_history", []):
                self._adjustment_history.append(
                    LearningAdjustment(
                        parameter=entry["parameter"],
                        old_value=entry["old_value"],
                        new_value=entry["new_value"],
                        reason=entry["reason"],
                    )
                )
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Failed to load supervised learning state: %s", exc)
