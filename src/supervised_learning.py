"""
Quantum Trading System – Live-Supervised Learning Module

Evaluates recent trading performance and dynamically adjusts model parameters
and trading strategy based on live feedback.  Acts as a bridge between the
Evaluator and the ML ensemble, applying data-driven adjustments to improve
signal quality over time.

Infinity-Loop Supervision
--------------------------
The module implements an "infinity-loop" mode where the AI leader
(Gemini / OpenRouter / Groq orchestrator) acts as a supervised-learning
student: it receives evaluation feedback after each epoch batch, proposes
hyperparameter adjustments, and those adjustments are applied before the next
batch starts.

Zero-Trade Handling
--------------------
When the evaluation window contains zero (or very few) trades it means the
current hyperparameters are too restrictive – signals cannot pass the filters.
The module detects this condition and *relaxes* the relevant thresholds /
agreement levels automatically (or via AI guidance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
    1. Tracks prediction accuracy per symbol.
    2. Adjusts signal thresholds based on rolling win-rate accuracy.
    3. Re-weights model ensemble based on individual model accuracy.
    4. Surfaces recent market regimes for training data elevation.
    5. Detects zero-trade conditions and relaxes hyperparameters so data
       can flow through the system.
    6. Supports an infinity-loop training mode where adjustments are applied
       between epoch batches with optional AI-leader guidance.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._accuracy_log: Dict[str, List[float]] = {}
        self._adjustment_history: List[LearningAdjustment] = []
        self._epoch_counter: int = 0
        self._zero_trade_streak: int = 0  # consecutive eval windows with 0 trades

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
            old_long_thresh = self.cfg.ml.long_threshold
            old_short_thresh = self.cfg.ml.short_threshold
            new_thresh = min(0.80, old_long_thresh + 0.02)
            if new_thresh != old_long_thresh:
                reason = (
                    f"Win rate {win_rate:.2%} below {self.cfg.evaluation.min_win_rate:.2%}"
                )
                adjustments.append(LearningAdjustment(
                    parameter="ml.long_threshold",
                    old_value=old_long_thresh,
                    new_value=new_thresh,
                    reason=reason,
                ))
                adjustments.append(LearningAdjustment(
                    parameter="ml.short_threshold",
                    old_value=old_short_thresh,
                    new_value=new_thresh,
                    reason=reason,
                ))
                self.cfg.ml.long_threshold = new_thresh
                self.cfg.ml.short_threshold = new_thresh

        # Relax thresholds if accuracy is very good
        if win_rate > 0.65 and sharpe > 1.5:
            old_long_thresh = self.cfg.ml.long_threshold
            old_short_thresh = self.cfg.ml.short_threshold
            new_thresh = max(0.50, old_long_thresh - 0.02)
            if new_thresh != old_long_thresh:
                reason = (
                    f"Strong performance (WR={win_rate:.2%}, Sharpe={sharpe:.2f})"
                )
                adjustments.append(LearningAdjustment(
                    parameter="ml.long_threshold",
                    old_value=old_long_thresh,
                    new_value=new_thresh,
                    reason=reason,
                ))
                adjustments.append(LearningAdjustment(
                    parameter="ml.short_threshold",
                    old_value=old_short_thresh,
                    new_value=new_thresh,
                    reason=reason,
                ))
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

    # ── Zero-trade detection ──────────────────────────────────────────────────

    def handle_zero_trades(self) -> List[LearningAdjustment]:
        """Relax hyperparameters when no trades have been made.

        A zero-trade result means the current signal thresholds and/or
        ensemble-agreement requirements are *too restrictive* – data cannot
        pass the filters effectively.  This method loosens the relevant
        parameters by the configured step sizes and tracks consecutive
        zero-trade evaluation windows.

        Returns
        -------
        List[LearningAdjustment]
            Applied adjustments (may be empty if already at minimum).
        """
        self._zero_trade_streak += 1
        adjustments: List[LearningAdjustment] = []
        step_thresh = self.cfg.ml.infinity_hp_adjust_step_threshold
        step_agree = self.cfg.ml.infinity_hp_adjust_agreement_step

        log.warning(
            "Zero-trade condition detected (streak=%d). Relaxing hyperparameters.",
            self._zero_trade_streak,
        )

        # Relax long_threshold
        old_lt = self.cfg.ml.long_threshold
        new_lt = max(0.50, old_lt - step_thresh)
        if new_lt != old_lt:
            adjustments.append(LearningAdjustment(
                parameter="ml.long_threshold",
                old_value=old_lt,
                new_value=new_lt,
                reason="Zero trades: lowering long threshold so signals can pass filters",
            ))
            self.cfg.ml.long_threshold = new_lt

        # Relax short_threshold
        old_st = self.cfg.ml.short_threshold
        new_st = max(0.50, old_st - step_thresh)
        if new_st != old_st:
            adjustments.append(LearningAdjustment(
                parameter="ml.short_threshold",
                old_value=old_st,
                new_value=new_st,
                reason="Zero trades: lowering short threshold so signals can pass filters",
            ))
            self.cfg.ml.short_threshold = new_st

        # Relax ensemble agreement
        old_agree = self.cfg.ml.min_ensemble_agreement
        new_agree = max(0.40, old_agree - step_agree)
        if new_agree != old_agree:
            adjustments.append(LearningAdjustment(
                parameter="ml.min_ensemble_agreement",
                old_value=old_agree,
                new_value=new_agree,
                reason="Zero trades: reducing ensemble agreement requirement",
            ))
            self.cfg.ml.min_ensemble_agreement = new_agree

        if not adjustments:
            log.warning(
                "Hyperparameters already at minimum values. "
                "Consider checking data quality or model health."
            )

        self._adjustment_history.extend(adjustments)
        return adjustments

    def reset_zero_trade_streak(self) -> None:
        """Reset the zero-trade counter after a successful trade window."""
        if self._zero_trade_streak > 0:
            log.info("Trades detected – resetting zero-trade streak.")
        self._zero_trade_streak = 0

    # ── Infinity-loop epoch tracking ──────────────────────────────────────────

    def increment_epoch(self) -> int:
        """Increment the internal epoch counter and return the new value."""
        self._epoch_counter += 1
        return self._epoch_counter

    @property
    def epoch(self) -> int:
        """Current epoch index."""
        return self._epoch_counter

    def should_evaluate(self) -> bool:
        """Return True when an evaluation checkpoint is due."""
        interval = self.cfg.ml.infinity_evaluation_interval
        if interval <= 0:
            return False
        return self._epoch_counter > 0 and self._epoch_counter % interval == 0

    def evaluate_and_adjust_with_ai(
        self,
        trade_history: list,
        current_metrics: Dict[str, Any],
        ai_callback: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ) -> List[LearningAdjustment]:
        """Run ``evaluate_and_adjust`` and optionally consult the AI leader.

        Parameters
        ----------
        trade_history:
            List of closed trade dicts.
        current_metrics:
            Performance metrics dict (e.g. from ``asdict(PerformanceMetrics)``).
        ai_callback:
            Optional callable that accepts the metrics dict and returns a dict
            of suggested hyperparameter values from the AI leader.  When
            provided, any valid AI suggestions are applied on top of the
            rule-based adjustments.

        Returns
        -------
        List[LearningAdjustment]
            All applied adjustments (rule-based + AI-guided).
        """
        total_trades = len(trade_history)
        zero_trade_threshold = self.cfg.ml.infinity_zero_trade_threshold

        # ── Zero-trade branch ─────────────────────────────────────────────
        if total_trades <= zero_trade_threshold:
            adjustments = self.handle_zero_trades()
        else:
            self.reset_zero_trade_streak()
            adjustments = self.evaluate_and_adjust(trade_history, current_metrics)

        # ── AI-leader guidance ────────────────────────────────────────────
        if ai_callback is not None:
            try:
                ai_suggestions = ai_callback(current_metrics)
                if isinstance(ai_suggestions, dict):
                    ai_adjustments = self._apply_ai_suggestions(ai_suggestions)
                    adjustments.extend(ai_adjustments)
            except Exception as exc:
                log.warning("AI callback failed: %s", exc)

        return adjustments

    def _apply_ai_suggestions(
        self, suggestions: Dict[str, Any]
    ) -> List[LearningAdjustment]:
        """Apply AI-suggested hyperparameter values if they are within safe bounds."""
        applied: List[LearningAdjustment] = []
        safe_ranges = {
            "ml.long_threshold":           (0.50, 0.85),
            "ml.short_threshold":          (0.50, 0.85),
            "ml.min_ensemble_agreement":   (0.40, 0.85),
            "ml.close_threshold":          (0.30, 0.70),
        }
        attr_map = {
            "ml.long_threshold":         ("ml", "long_threshold"),
            "ml.short_threshold":        ("ml", "short_threshold"),
            "ml.min_ensemble_agreement": ("ml", "min_ensemble_agreement"),
            "ml.close_threshold":        ("ml", "close_threshold"),
        }

        for param, raw_val in suggestions.items():
            if param not in safe_ranges:
                continue
            try:
                new_val = float(raw_val)
            except (TypeError, ValueError):
                continue
            lo, hi = safe_ranges[param]
            new_val = max(lo, min(hi, new_val))
            section, attr = attr_map[param]
            cfg_section = getattr(self.cfg, section)
            old_val = getattr(cfg_section, attr)
            if abs(new_val - old_val) < 1e-6:
                continue
            setattr(cfg_section, attr, new_val)
            adj = LearningAdjustment(
                parameter=param,
                old_value=old_val,
                new_value=new_val,
                reason="AI leader suggestion",
            )
            applied.append(adj)
            log.info("AI suggestion applied: %s → %s", param, new_val)

        self._adjustment_history.extend(applied)
        return applied

    def save_state(self, path: Path) -> None:
        """Persist learning state."""
        state = {
            "accuracy_log": self._accuracy_log,
            "epoch_counter": self._epoch_counter,
            "zero_trade_streak": self._zero_trade_streak,
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
            self._epoch_counter = int(state.get("epoch_counter", 0))
            self._zero_trade_streak = int(state.get("zero_trade_streak", 0))
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
