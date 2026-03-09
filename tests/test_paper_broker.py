"""Tests for PaperBroker."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.paper_broker import PaperBroker
from src.risk_manager import PositionSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def broker(config):
    return PaperBroker(config)


def _make_spec(
    symbol="BTC",
    side="long",
    price=40_000.0,
    size_usd=1_000.0,
    leverage=15,
) -> PositionSpec:
    contracts = size_usd / price
    sl = price * 0.98 if side == "long" else price * 1.02
    tp = price * 1.03 if side == "long" else price * 0.97
    return PositionSpec(
        symbol=symbol,
        side=side,
        entry_price=price,
        size_usd=size_usd,
        size_contracts=contracts,
        leverage=leverage,
        stop_loss=sl,
        take_profit=tp,
        trailing_stop_pct=0.015,
        risk_usd=abs(price - sl) * contracts,
        allowed=True,
    )


# ── Open / close positions ─────────────────────────────────────────────────────

class TestPaperBrokerPositions:
    def test_open_long_position(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=1_000.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None
        assert pos.side == "long"
        assert pos.symbol == "BTC"
        assert broker.balance < broker.initial_equity

    def test_open_short_position(self, broker):
        spec = _make_spec(side="short", price=40_000.0, size_usd=1_000.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None
        assert pos.side == "short"

    def test_close_long_profit(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=1_000.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None
        initial_balance = broker.balance

        trade = broker.close_position(pos.position_id, exit_price=41_000.0, reason="signal")
        assert trade is not None
        assert trade.pnl > 0
        assert trade.exit_reason == "signal"

    def test_close_long_loss(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=1_000.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None

        trade = broker.close_position(pos.position_id, exit_price=39_000.0, reason="signal")
        assert trade is not None
        assert trade.pnl < 0

    def test_close_short_profit(self, broker):
        spec = _make_spec(side="short", price=40_000.0, size_usd=1_000.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None

        trade = broker.close_position(pos.position_id, exit_price=39_000.0, reason="signal")
        assert trade is not None
        assert trade.pnl > 0

    def test_insufficient_balance(self, broker):
        # Try to open a position requiring more than available balance
        spec = _make_spec(side="long", price=40_000.0, size_usd=500_000.0, leverage=1)
        pos = broker.open_position(spec)
        assert pos is None

    def test_close_nonexistent_position(self, broker):
        trade = broker.close_position("nonexistent-id", exit_price=40_000.0)
        assert trade is None

    def test_position_count_and_cleanup(self, broker):
        spec = _make_spec(side="long", size_usd=500.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None
        assert len(broker.positions) == 1

        broker.close_position(pos.position_id, exit_price=40_000.0)
        assert len(broker.positions) == 0
        assert len(broker.trade_history) == 1


# ── Auto-close via update_positions ───────────────────────────────────────────

class TestAutoClose:
    def test_stop_loss_triggers(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=500.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None

        # Price drops below stop loss
        closed = broker.update_positions("BTC", current_price=38_000.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"

    def test_take_profit_triggers(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=500.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None

        # Price rises above take profit
        closed = broker.update_positions("BTC", current_price=42_000.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "take_profit"

    def test_no_auto_close_within_range(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=500.0, leverage=10)
        broker.open_position(spec)

        closed = broker.update_positions("BTC", current_price=40_100.0)
        assert len(closed) == 0
        assert len(broker.positions) == 1

    def test_funding_accrual(self, broker):
        spec = _make_spec(side="long", price=40_000.0, size_usd=500.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None

        # Positive funding rate – long pays
        broker.update_positions("BTC", current_price=40_100.0, funding_rate=0.0001)
        updated_pos = broker.positions[pos.position_id]
        assert updated_pos.funding_accrued < 0  # long pays

    def test_only_updates_correct_symbol(self, broker):
        spec_btc = _make_spec(symbol="BTC", side="long", price=40_000.0, size_usd=500.0)
        pos = broker.open_position(spec_btc)
        assert pos is not None

        # Update ETH – BTC position should not change
        closed = broker.update_positions("ETH", current_price=2_000.0)
        assert len(closed) == 0
        assert pos.position_id in broker.positions


# ── Get open position ─────────────────────────────────────────────────────────

class TestGetOpenPosition:
    def test_returns_position(self, broker):
        spec = _make_spec(symbol="BTC", side="long")
        pos = broker.open_position(spec)
        result = broker.get_open_position("BTC")
        assert result is not None
        assert result.symbol == "BTC"

    def test_returns_none_when_not_open(self, broker):
        assert broker.get_open_position("ETH") is None


# ── Equity calculation ────────────────────────────────────────────────────────

class TestEquity:
    def test_equity_increases_with_profitable_position(self, broker):
        initial = broker.get_equity()
        spec = _make_spec(side="long", price=40_000.0, size_usd=1_000.0, leverage=10)
        broker.open_position(spec)

        eq_up = broker.get_equity({"BTC": 42_000.0})
        assert eq_up > initial - 100  # fees reduce slightly

    def test_equity_decreases_with_losing_position(self, broker):
        initial = broker.get_equity()
        spec = _make_spec(side="long", price=40_000.0, size_usd=1_000.0, leverage=10)
        broker.open_position(spec)

        eq_down = broker.get_equity({"BTC": 38_000.0})
        assert eq_down < initial


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, broker, tmp_path):
        spec = _make_spec(side="long", price=40_000.0, size_usd=500.0, leverage=10)
        pos = broker.open_position(spec)
        assert pos is not None
        broker.close_position(pos.position_id, exit_price=41_000.0)

        path = tmp_path / "broker.json"
        broker.save(path)
        assert path.exists()

        # Load into a new instance
        broker2 = PaperBroker(load_config())
        broker2.load(path)
        assert len(broker2.trade_history) == 1
        assert broker2.trade_history[0].symbol == "BTC"

    def test_load_nonexistent_file(self, broker, tmp_path):
        # Should not raise
        broker.load(tmp_path / "nonexistent.json")
        assert broker.equity == broker.initial_equity
