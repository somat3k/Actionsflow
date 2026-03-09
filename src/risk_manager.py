"""
Quantum Trading System – Risk & Leverage Manager
Computes position sizes, validates risk limits, and manages dynamic leverage
in the 10–35x range with Kelly criterion sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.utils import clamp, get_logger, safe_divide

log = get_logger(__name__)


@dataclass
class PositionRequest:
    symbol: str
    signal: int                    # 1=long, 2=short
    confidence: float
    current_price: float
    atr: float
    equity: float
    leverage: int
    open_positions: int


@dataclass
class PositionSpec:
    symbol: str
    side: str                      # "long" | "short"
    entry_price: float
    size_usd: float
    size_contracts: float
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    risk_usd: float
    allowed: bool = True
    reject_reason: str = ""


class RiskManager:
    """
    Validates and sizes positions using Kelly criterion and ATR-based stops.
    Applies portfolio-level risk limits.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.risk_cfg = config.trading.risk
        self.ps_cfg = config.trading.position_sizing
        self.lev_cfg = config.trading.leverage

        # Daily P&L tracking
        self._daily_pnl: float = 0.0
        self._initial_daily_equity: float = config.trading.initial_equity

    # ── Public interface ───────────────────────────────────────────────────────

    def compute_position(
        self,
        req: PositionRequest,
        trade_history: Optional[List[Dict]] = None,
    ) -> PositionSpec:
        """
        Compute the full position specification for a trade request.
        Returns a PositionSpec; if the trade should not be placed,
        allowed=False with a reject_reason.
        """
        side = "long" if req.signal == 1 else "short"

        # ── Risk limit checks ──────────────────────────────────────────────────
        reject = self._check_risk_limits(req)
        if reject:
            return PositionSpec(
                symbol=req.symbol,
                side=side,
                entry_price=req.current_price,
                size_usd=0.0,
                size_contracts=0.0,
                leverage=req.leverage,
                stop_loss=0.0,
                take_profit=0.0,
                trailing_stop_pct=self.risk_cfg.trailing_stop_pct,
                risk_usd=0.0,
                allowed=False,
                reject_reason=reject,
            )

        # ── Stop / take-profit ────────────────────────────────────────────────
        atr_stop = req.atr * self.risk_cfg.stop_loss_atr_multiplier
        atr_tp = req.atr * self.risk_cfg.take_profit_atr_multiplier

        if side == "long":
            stop_loss = req.current_price - atr_stop
            take_profit = req.current_price + atr_tp
        else:
            stop_loss = req.current_price + atr_stop
            take_profit = req.current_price - atr_tp

        # ── Position sizing ────────────────────────────────────────────────────
        size_usd = self._compute_size(req, atr_stop, trade_history)
        size_contracts = size_usd / req.current_price

        risk_per_contract = abs(req.current_price - stop_loss)
        risk_usd = size_contracts * risk_per_contract

        log.debug(
            "%s %s: size=%.2f USD, leverage=%dx, sl=%.4f, tp=%.4f, risk=%.2f USD",
            side.upper(), req.symbol, size_usd, req.leverage,
            stop_loss, take_profit, risk_usd,
        )

        return PositionSpec(
            symbol=req.symbol,
            side=side,
            entry_price=req.current_price,
            size_usd=size_usd,
            size_contracts=size_contracts,
            leverage=req.leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=self.risk_cfg.trailing_stop_pct,
            risk_usd=risk_usd,
            allowed=True,
        )

    def adjust_leverage(
        self,
        current_leverage: int,
        ml_confidence: float,
        gemini_recommendation: Optional[int] = None,
    ) -> int:
        """
        Determine final leverage (10–35x) blending ML confidence and Gemini.
        """
        lev = current_leverage

        # ML-driven adjustment
        if ml_confidence >= self.lev_cfg.high_confidence_threshold:
            lev = min(lev + self.lev_cfg.step, self.lev_cfg.max)
        elif ml_confidence < self.lev_cfg.low_confidence_threshold:
            lev = max(lev - self.lev_cfg.step, self.lev_cfg.min)

        # Blend with Gemini recommendation (60/40 ML/Gemini)
        if gemini_recommendation is not None:
            g_rec = clamp(gemini_recommendation, self.lev_cfg.min, self.lev_cfg.max)
            lev = round(0.6 * lev + 0.4 * g_rec)

        return int(clamp(lev, self.lev_cfg.min, self.lev_cfg.max))

    def record_daily_pnl(self, pnl: float, equity: float) -> None:
        self._daily_pnl += pnl
        self._initial_daily_equity = equity - pnl  # approximate

    def reset_daily_tracking(self, equity: float) -> None:
        self._daily_pnl = 0.0
        self._initial_daily_equity = equity

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_risk_limits(self, req: PositionRequest) -> str:
        """Return rejection reason string if trade should not proceed, else ''."""
        if req.open_positions >= self.risk_cfg.max_open_positions:
            return f"Max open positions reached ({self.risk_cfg.max_open_positions})"

        daily_loss_limit = self._initial_daily_equity * self.risk_cfg.daily_loss_limit_pct
        if self._daily_pnl < -daily_loss_limit:
            return f"Daily loss limit breached ({self._daily_pnl:.2f} USD)"

        if req.equity <= 0:
            return "Zero or negative equity"

        return ""

    def _compute_size(
        self,
        req: PositionRequest,
        atr_stop_distance: float,
        trade_history: Optional[List[Dict]],
    ) -> float:
        """Compute notional USD size using selected sizing method."""
        if self.ps_cfg.method == "kelly_fractional":
            size = self._kelly_size(req, atr_stop_distance, trade_history)
        else:
            size = req.equity * 0.10  # fallback: 10% equity

        # Apply maximum position cap
        max_size = req.equity * self.ps_cfg.max_position_pct * req.leverage
        size = min(size, max_size)
        size = max(size, self.ps_cfg.min_position_usd)
        return round(size, 2)

    def _kelly_size(
        self,
        req: PositionRequest,
        atr_stop_distance: float,
        trade_history: Optional[List[Dict]],
    ) -> float:
        """
        Kelly criterion position size in USD.
        f* = (p*b - q) / b  where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if trade_history and len(trade_history) >= 20:
            wins = [t for t in trade_history if t.get("pnl", 0) > 0]
            losses = [t for t in trade_history if t.get("pnl", 0) < 0]
            win_rate = len(wins) / len(trade_history)
            avg_win = np.mean([t["pnl"] for t in wins]) if wins else req.current_price * 0.01
            avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else req.current_price * 0.005
        else:
            win_rate = 0.55
            avg_win = atr_stop_distance * self.risk_cfg.take_profit_atr_multiplier
            avg_loss = atr_stop_distance

        b = safe_divide(avg_win, avg_loss, default=1.5)
        q = 1 - win_rate
        kelly_f = safe_divide(win_rate * b - q, b, default=0.05)
        kelly_f = clamp(kelly_f, 0.01, 0.30)

        # Fractional Kelly
        frac_kelly = kelly_f * self.ps_cfg.kelly_fraction

        # Scale by confidence
        confidence_scale = 0.5 + req.confidence * 0.5
        size_usd = req.equity * frac_kelly * confidence_scale * req.leverage
        return size_usd
