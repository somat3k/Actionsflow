"""Helpers for dashboard control parsing and chat commands."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Optional, Sequence


@dataclass(frozen=True)
class AdjustmentSpec:
    key: str
    label: str
    patterns: tuple[str, ...]
    value_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


ADJUSTMENT_SPECS: tuple[AdjustmentSpec, ...] = (
    AdjustmentSpec(
        key="ml.long_threshold",
        label="Long threshold",
        patterns=("long threshold", "long signal threshold", "long probability"),
        value_type="ratio",
        min_value=0.0,
        max_value=1.0,
    ),
    AdjustmentSpec(
        key="ml.short_threshold",
        label="Short threshold",
        patterns=("short threshold", "short signal threshold", "short probability"),
        value_type="ratio",
        min_value=0.0,
        max_value=1.0,
    ),
    AdjustmentSpec(
        key="ml.close_threshold",
        label="Close threshold",
        patterns=("close threshold", "exit threshold", "close signal threshold"),
        value_type="ratio",
        min_value=0.0,
        max_value=1.0,
    ),
    AdjustmentSpec(
        key="ml.min_ensemble_agreement",
        label="Min ensemble agreement",
        patterns=("ensemble agreement", "min agreement", "agreement threshold"),
        value_type="ratio",
        min_value=0.0,
        max_value=1.0,
    ),
    AdjustmentSpec(
        key="ml.reinforcement_alpha",
        label="Reinforcement alpha",
        patterns=("reinforcement alpha", "reinforcement factor", "reinforcement weight"),
        value_type="ratio",
        min_value=0.0,
        max_value=1.0,
    ),
    AdjustmentSpec(
        key="ml.training_epochs",
        label="Training epochs",
        patterns=("training epochs", "epoch count", "epochs"),
        value_type="int",
        min_value=1,
        max_value=200,
    ),
    AdjustmentSpec(
        key="ml.retrain_interval_hours",
        label="Retrain interval (hrs)",
        patterns=("retrain interval", "retrain hours", "retrain interval hours"),
        value_type="int",
        min_value=1,
        max_value=168,
    ),
    AdjustmentSpec(
        key="trading.leverage.min",
        label="Min leverage",
        patterns=("min leverage", "minimum leverage", "leverage min"),
        value_type="int",
        min_value=1,
        max_value=100,
    ),
    AdjustmentSpec(
        key="trading.leverage.max",
        label="Max leverage",
        patterns=("max leverage", "maximum leverage", "leverage max"),
        value_type="int",
        min_value=1,
        max_value=100,
    ),
)

_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%?")


def parse_adjustment_request(text: str) -> Optional[Dict[str, Any]]:
    """Parse a natural-language adjustment request into a structured payload."""
    if not text:
        return None
    normalized = " ".join(text.lower().strip().split())
    match = _NUMBER_RE.search(normalized)
    if not match:
        return None
    raw_value = float(match.group(1))
    raw_segment = match.group(0)
    has_percent = "%" in raw_segment or "percent" in normalized or "pct" in normalized
    for spec in ADJUSTMENT_SPECS:
        if any(pattern in normalized for pattern in spec.patterns):
            value = _coerce_adjustment_value(raw_value, spec, has_percent)
            return {
                "key": spec.key,
                "label": spec.label,
                "value": value,
                "raw_value": raw_value,
            }
    return None


def parse_trade_request(
    text: str,
    symbols: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract a simple trade intent from chat text."""
    if not text:
        return None
    normalized = text.lower()
    side_match = re.search(r"\b(long|short)\b", normalized)
    if not side_match:
        return None
    symbol = _find_symbol(text, symbols or [])
    if symbol is None:
        return None
    leverage_match = re.search(r"(\d+(?:\.\d+)?)\s*x", normalized)
    leverage = None
    if leverage_match:
        leverage = int(round(float(leverage_match.group(1))))
    return {
        "symbol": symbol,
        "side": side_match.group(1).upper(),
        "leverage": leverage,
    }


def _find_symbol(text: str, symbols: Iterable[str]) -> Optional[str]:
    upper_text = text.upper()
    for symbol in symbols:
        upper_symbol = symbol.upper()
        if upper_symbol in upper_text:
            return upper_symbol
    match = re.search(r"\b([A-Z]{2,6})\b", upper_text)
    if match:
        return match.group(1)
    return None


def _coerce_adjustment_value(
    raw_value: float,
    spec: AdjustmentSpec,
    has_percent: bool,
) -> float | int:
    value = raw_value
    if spec.value_type == "ratio" and has_percent:
        value = value / 100
    if spec.value_type == "int":
        value = int(round(value))
    else:
        value = float(value)
    if spec.min_value is not None:
        value = max(spec.min_value, value)
    if spec.max_value is not None:
        value = min(spec.max_value, value)
    return value
