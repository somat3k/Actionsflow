"""Tests for RiskManager."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.risk_manager import PositionRequest, RiskManager


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def risk_mgr(config):
    return RiskManager(config)


def _make_request(
    signal: int = 1,
    confidence: float = 0.70,
    price: float = 40_000.0,
    atr: float = 400.0,
    equity: float = 10_000.0,
    leverage: int = 15,
    open_positions: int = 0,
) -> PositionRequest:
    return PositionRequest(
        symbol="BTC",
        signal=signal,
        confidence=confidence,
        current_price=price,
        atr=atr,
        equity=equity,
        leverage=leverage,
        open_positions=open_positions,
    )


class TestRiskManagerPositionSizing:
    def test_long_position_allowed(self, risk_mgr):
        req = _make_request(signal=1)
        spec = risk_mgr.compute_position(req)
        assert spec.allowed is True
        assert spec.side == "long"

    def test_short_position_allowed(self, risk_mgr):
        req = _make_request(signal=2)
        spec = risk_mgr.compute_position(req)
        assert spec.allowed is True
        assert spec.side == "short"

    def test_stop_loss_below_entry_for_long(self, risk_mgr):
        req = _make_request(signal=1, price=40_000.0, atr=400.0)
        spec = risk_mgr.compute_position(req)
        assert spec.stop_loss < spec.entry_price

    def test_stop_loss_above_entry_for_short(self, risk_mgr):
        req = _make_request(signal=2, price=40_000.0, atr=400.0)
        spec = risk_mgr.compute_position(req)
        assert spec.stop_loss > spec.entry_price

    def test_take_profit_above_entry_for_long(self, risk_mgr):
        req = _make_request(signal=1, price=40_000.0, atr=400.0)
        spec = risk_mgr.compute_position(req)
        assert spec.take_profit > spec.entry_price

    def test_take_profit_below_entry_for_short(self, risk_mgr):
        req = _make_request(signal=2, price=40_000.0, atr=400.0)
        spec = risk_mgr.compute_position(req)
        assert spec.take_profit < spec.entry_price

    def test_size_above_minimum(self, risk_mgr, config):
        req = _make_request(signal=1)
        spec = risk_mgr.compute_position(req)
        assert spec.size_usd >= config.trading.position_sizing.min_position_usd

    def test_size_respects_max_pct(self, risk_mgr, config):
        req = _make_request(signal=1, equity=10_000.0, leverage=15)
        spec = risk_mgr.compute_position(req)
        max_size = req.equity * config.trading.position_sizing.max_position_pct * req.leverage
        assert spec.size_usd <= max_size + 0.01

    def test_rejected_when_max_positions_reached(self, risk_mgr, config):
        max_pos = config.trading.risk.max_open_positions
        req = _make_request(signal=1, open_positions=max_pos)
        spec = risk_mgr.compute_position(req)
        assert spec.allowed is False
        assert "Max open positions" in spec.reject_reason

    def test_rejected_when_zero_equity(self, risk_mgr):
        req = _make_request(signal=1, equity=0.0)
        spec = risk_mgr.compute_position(req)
        assert spec.allowed is False

    def test_daily_loss_limit_rejection(self, risk_mgr, config):
        # Simulate daily loss exceeding limit
        limit = config.trading.initial_equity * config.trading.risk.daily_loss_limit_pct
        risk_mgr._daily_pnl = -(limit + 100)
        req = _make_request(signal=1)
        spec = risk_mgr.compute_position(req)
        assert spec.allowed is False
        assert "daily loss" in spec.reject_reason.lower()


class TestLeverageAdjustment:
    def test_high_confidence_increases_leverage(self, risk_mgr, config):
        lev = config.trading.leverage.default
        new_lev = risk_mgr.adjust_leverage(lev, ml_confidence=0.90)
        assert new_lev > lev

    def test_low_confidence_decreases_leverage(self, risk_mgr, config):
        lev = config.trading.leverage.default
        new_lev = risk_mgr.adjust_leverage(lev, ml_confidence=0.40)
        assert new_lev < lev

    def test_leverage_stays_within_bounds(self, risk_mgr, config):
        lev_cfg = config.trading.leverage
        for conf in [0.0, 0.5, 1.0]:
            lev = risk_mgr.adjust_leverage(lev_cfg.default, ml_confidence=conf)
            assert lev_cfg.min <= lev <= lev_cfg.max

    def test_gemini_recommendation_blended(self, risk_mgr, config):
        default_lev = config.trading.leverage.default
        # High confidence would push up, but Gemini says lower
        result = risk_mgr.adjust_leverage(
            default_lev,
            ml_confidence=0.90,
            gemini_recommendation=config.trading.leverage.min,
        )
        # Blended result should be between min and what ML suggests
        assert config.trading.leverage.min <= result <= config.trading.leverage.max

    def test_leverage_capped_at_max(self, risk_mgr, config):
        lev = risk_mgr.adjust_leverage(
            config.trading.leverage.max,
            ml_confidence=1.0,
            gemini_recommendation=config.trading.leverage.max,
        )
        assert lev == config.trading.leverage.max

    def test_leverage_floored_at_min(self, risk_mgr, config):
        lev = risk_mgr.adjust_leverage(
            config.trading.leverage.min,
            ml_confidence=0.0,
            gemini_recommendation=config.trading.leverage.min,
        )
        assert lev == config.trading.leverage.min
