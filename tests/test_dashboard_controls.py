from __future__ import annotations

import pytest

from src.dashboard_controls import parse_adjustment_request, parse_trade_request


def test_parse_adjustment_request_percent_value():
    result = parse_adjustment_request("Set long threshold to 65% for next run.")
    assert result is not None
    assert result["key"] == "ml.long_threshold"
    assert result["value"] == pytest.approx(0.65)


def test_parse_trade_request_with_leverage():
    result = parse_trade_request("Open long BTC 15x and monitor risk", symbols=["BTC", "ETH"])
    assert result == {"symbol": "BTC", "side": "LONG", "leverage": 15}
