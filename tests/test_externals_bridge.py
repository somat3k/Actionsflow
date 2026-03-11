from __future__ import annotations

import pandas as pd

import externals.python.gemini_bridge_server as bridge


class DummyEnsemble:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, symbol: str) -> bool:
        return True

    def predict(self, features: pd.DataFrame) -> dict:
        return {"signal": 1, "confidence": 0.6}


class DummyEnsembleUnavailable(DummyEnsemble):
    def load(self, symbol: str) -> bool:
        return False


def _dummy_features(_: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2],
            "atr_14": [0.01, 0.01, 0.01],
        }
    )


def _make_payload() -> dict:
    return {
        "symbol": "EURUSD",
        "candles": [
            {
                "time": "2026-03-11T06:00:00Z",
                "open": 1.0834,
                "high": 1.0839,
                "low": 1.0832,
                "close": 1.0837,
                "volume": 1452,
            }
        ],
        "account": {"equity": 10000, "balance": 10000, "leverage": 20},
        "positions": {"long": 0, "short": 0},
        "recent_trades": [{"pnl": 10.0}, {"fee_usd": 0.2}],
    }


def _build_state(monkeypatch, ensemble_cls) -> bridge.BridgeState:
    monkeypatch.setattr(bridge, "QuantumEnsemble", ensemble_cls)
    monkeypatch.setattr(bridge, "GeminiOrchestrator", None)
    monkeypatch.setattr(bridge, "add_all_features", _dummy_features)
    return bridge.BridgeState()


def test_handle_payload_requires_symbol(monkeypatch):
    state = _build_state(monkeypatch, DummyEnsemble)
    payload = _make_payload()
    payload.pop("symbol", None)
    response = state.handle_payload(payload)
    assert response["status"] == "error"
    assert response["message"] == "symbol missing"


def test_handle_payload_filters_trades(monkeypatch):
    state = _build_state(monkeypatch, DummyEnsemble)
    response = state.handle_payload(_make_payload())
    assert response["status"] == "ok"
    assert response["metrics"]["total_trades"] == 1
    assert response["position"] is not None


def test_handle_payload_model_unavailable(monkeypatch):
    state = _build_state(monkeypatch, DummyEnsembleUnavailable)
    response = state.handle_payload(_make_payload())
    assert response["status"] == "model_unavailable"
