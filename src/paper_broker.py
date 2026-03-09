"""
Quantum Trading System – Paper Broker
Simulates isolated-margin perpetual futures trading with realistic fees,
funding rates, slippage, and liquidations for ML model training and validation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import AppConfig
from src.risk_manager import PositionSpec
from src.utils import get_logger, utc_now, utc_now_ms

log = get_logger(__name__)


@dataclass
class PaperPosition:
    position_id: str
    symbol: str
    side: str                      # "long" | "short"
    entry_price: float
    size_contracts: float
    size_usd: float
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    entry_time_ms: int
    margin_usd: float              # isolated margin
    funding_accrued: float = 0.0
    unrealised_pnl: float = 0.0
    max_favourable_excursion: float = 0.0  # for trailing stop


@dataclass
class ClosedTrade:
    position_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size_contracts: float
    size_usd: float
    leverage: int
    entry_time_ms: int
    exit_time_ms: int
    pnl: float
    pnl_pct: float
    fee_usd: float
    funding_usd: float
    exit_reason: str               # "take_profit" | "stop_loss" | "trailing_stop" | "signal" | "liquidation"
    duration_ms: int = 0

    def __post_init__(self) -> None:
        self.duration_ms = self.exit_time_ms - self.entry_time_ms


class PaperBroker:
    """
    Isolated-margin paper trading broker for strategy validation and ML training.
    Maintains a virtual portfolio with realistic cost simulation.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.bcfg = config.paper_broker

        self.equity: float = self.bcfg.initial_equity
        self.initial_equity: float = self.bcfg.initial_equity
        self.balance: float = self.bcfg.initial_equity   # free balance

        self.positions: Dict[str, PaperPosition] = {}    # key = position_id
        self.trade_history: List[ClosedTrade] = []

    # ── Position management ────────────────────────────────────────────────────

    def open_position(
        self, spec: PositionSpec, current_price: Optional[float] = None
    ) -> Optional[PaperPosition]:
        """
        Open an isolated-margin position.
        Returns the created position, or None if rejected.
        """
        price = current_price or spec.entry_price
        slipped_price = self._apply_slippage(price, spec.side)
        margin_required = spec.size_usd / spec.leverage
        fee = spec.size_usd * self.bcfg.taker_fee

        if margin_required + fee > self.balance:
            log.warning(
                "Insufficient balance: need %.2f, have %.2f", margin_required + fee, self.balance
            )
            return None

        self.balance -= margin_required + fee

        pos = PaperPosition(
            position_id=str(uuid.uuid4())[:8],
            symbol=spec.symbol,
            side=spec.side,
            entry_price=slipped_price,
            size_contracts=spec.size_contracts,
            size_usd=spec.size_usd,
            leverage=spec.leverage,
            stop_loss=spec.stop_loss,
            take_profit=spec.take_profit,
            trailing_stop_pct=spec.trailing_stop_pct,
            entry_time_ms=utc_now_ms(),
            margin_usd=margin_required,
        )
        self.positions[pos.position_id] = pos
        log.info(
            "OPEN %s %s @ %.4f | size=%.2f USD | lev=%dx | sl=%.4f | tp=%.4f",
            spec.side.upper(), spec.symbol, slipped_price,
            spec.size_usd, spec.leverage, spec.stop_loss, spec.take_profit,
        )
        return pos

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "signal",
    ) -> Optional[ClosedTrade]:
        """Close a position at exit_price."""
        pos = self.positions.pop(position_id, None)
        if pos is None:
            return None

        slipped_price = self._apply_slippage(exit_price, "sell" if pos.side == "long" else "buy")
        pnl = self._compute_pnl(pos, slipped_price)
        fee = pos.size_usd * self.bcfg.taker_fee

        self.balance += pos.margin_usd + pnl - fee
        self.equity = self.balance + sum(
            self._compute_pnl(p, exit_price) for p in self.positions.values()
        )

        trade = ClosedTrade(
            position_id=position_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=slipped_price,
            size_contracts=pos.size_contracts,
            size_usd=pos.size_usd,
            leverage=pos.leverage,
            entry_time_ms=pos.entry_time_ms,
            exit_time_ms=utc_now_ms(),
            pnl=pnl - fee + pos.funding_accrued,
            pnl_pct=(pnl - fee) / pos.margin_usd if pos.margin_usd > 0 else 0.0,
            fee_usd=fee,
            funding_usd=pos.funding_accrued,
            exit_reason=reason,
        )
        self.trade_history.append(trade)
        log.info(
            "CLOSE %s %s @ %.4f | PnL=%.2f USD (%.2f%%) | reason=%s",
            pos.side.upper(), pos.symbol, slipped_price,
            trade.pnl, trade.pnl_pct * 100, reason,
        )
        return trade

    def update_positions(
        self, symbol: str, current_price: float, funding_rate: float = 0.0
    ) -> List[ClosedTrade]:
        """
        Update all open positions for `symbol`:
        - Mark-to-market unrealised P&L
        - Apply funding payment
        - Check stop-loss / take-profit / trailing-stop
        - Check liquidation
        Returns list of positions that were auto-closed.
        """
        closed: List[ClosedTrade] = []
        to_close: List[tuple] = []

        for pid, pos in list(self.positions.items()):
            if pos.symbol != symbol:
                continue

            # Update unrealised PnL
            pos.unrealised_pnl = self._compute_pnl(pos, current_price)

            # Funding
            if self.bcfg.funding_enabled:
                funding_payment = pos.size_usd * abs(funding_rate)
                # Long pays when funding > 0, short receives; vice versa
                if funding_rate >= 0:
                    funding_sign = -1 if pos.side == "long" else 1
                else:
                    funding_sign = 1 if pos.side == "long" else -1
                pos.funding_accrued += funding_sign * funding_payment

            # Trailing stop: update stop as price moves in favour
            if pos.side == "long":
                if current_price > pos.max_favourable_excursion:
                    pos.max_favourable_excursion = current_price
                trail_stop = pos.max_favourable_excursion * (1 - pos.trailing_stop_pct)
                if trail_stop > pos.stop_loss:
                    pos.stop_loss = trail_stop
            else:
                if current_price < pos.max_favourable_excursion or pos.max_favourable_excursion == 0:
                    pos.max_favourable_excursion = current_price
                trail_stop = pos.max_favourable_excursion * (1 + pos.trailing_stop_pct)
                if trail_stop < pos.stop_loss or pos.stop_loss == 0:
                    pos.stop_loss = trail_stop

            # Liquidation check (isolated margin: loss > margin)
            if self.bcfg.liquidation_enabled:
                liq_loss = pos.margin_usd * 0.90  # 90% margin loss = liquidated
                if pos.unrealised_pnl < -liq_loss:
                    to_close.append((pid, current_price, "liquidation"))
                    continue

            # Stop-loss
            if pos.side == "long" and current_price <= pos.stop_loss:
                to_close.append((pid, current_price, "stop_loss"))
            elif pos.side == "short" and current_price >= pos.stop_loss:
                to_close.append((pid, current_price, "stop_loss"))
            # Take-profit
            elif pos.side == "long" and current_price >= pos.take_profit:
                to_close.append((pid, current_price, "take_profit"))
            elif pos.side == "short" and current_price <= pos.take_profit:
                to_close.append((pid, current_price, "take_profit"))

        for pid, price, reason in to_close:
            trade = self.close_position(pid, price, reason)
            if trade:
                closed.append(trade)

        return closed

    def get_open_position(self, symbol: str) -> Optional[PaperPosition]:
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return pos
        return None

    def get_equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total equity including unrealised P&L."""
        total_upnl = 0.0
        for pos in self.positions.values():
            price = (prices or {}).get(pos.symbol, pos.entry_price)
            total_upnl += self._compute_pnl(pos, price)
        return self.balance + total_upnl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": self.equity,
            "balance": self.balance,
            "initial_equity": self.initial_equity,
            "open_positions": [asdict(p) for p in self.positions.values()],
            "trade_count": len(self.trade_history),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "equity": self.equity,
            "balance": self.balance,
            "initial_equity": self.initial_equity,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trade_history": [asdict(t) for t in self.trade_history],
        }
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2, default=str)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path) as fh:
            state = json.load(fh)
        self.equity = state.get("equity", self.initial_equity)
        self.balance = state.get("balance", self.initial_equity)
        self.initial_equity = state.get("initial_equity", self.initial_equity)
        self.positions = {
            k: PaperPosition(**v)
            for k, v in state.get("positions", {}).items()
        }
        self.trade_history = [
            ClosedTrade(**t) for t in state.get("trade_history", [])
        ]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, side: str) -> float:
        if self.bcfg.slippage_bps == 0:
            return price
        slip = price * self.bcfg.slippage_bps / 10_000
        # Buy/long: price increases; sell/short: price decreases
        return price + slip if side in ("long", "buy") else price - slip

    @staticmethod
    def _compute_pnl(pos: PaperPosition, current_price: float) -> float:
        if pos.side == "long":
            return (current_price - pos.entry_price) * pos.size_contracts
        else:
            return (pos.entry_price - current_price) * pos.size_contracts
