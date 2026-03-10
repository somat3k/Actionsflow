from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.config import load_config
from src.database_manager import DatabaseManager
from src.main import _get_hyperliquid_private_key, run_paper_signal
from src.utils import utc_now


class DummyEnsemble:
    default_loaded = True
    train_calls = []

    def __init__(self, config):
        self.cfg = config
        self.loaded = self.default_loaded

    def load(self, symbol):
        return self.loaded

    def train(self, df, symbol="BTC"):
        self.train_calls.append(symbol)
        self.loaded = True
        return {}

    def predict(self, df):
        return {
            "signal": 0,
            "confidence": 0.4,
            "long_prob": 0.0,
            "short_prob": 0.0,
            "flat_prob": 1.0,
            "agreement": 1.0,
            "model_signals": {},
        }


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()
    return tmp_path


def test_hyperliquid_secret_is_preferred_over_legacy_key(monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_SECRET", "secret-key")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "legacy-key")

    assert _get_hyperliquid_private_key() == "secret-key"


def test_hyperliquid_legacy_key_is_used_as_fallback(monkeypatch):
    monkeypatch.delenv("HYPERLIQUID_SECRET", raising=False)
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "legacy-key")

    assert _get_hyperliquid_private_key() == "legacy-key"


def test_signal_retrains_when_model_stale(test_env, monkeypatch):
    DummyEnsemble.default_loaded = True
    DummyEnsemble.train_calls = []
    monkeypatch.setattr("src.ml_models.QuantumEnsemble", DummyEnsemble)

    cfg = load_config()
    db_path = test_env / cfg.system.state_dir / cfg.system.database_file
    db = DatabaseManager(db_path)
    stale_time = (utc_now() - timedelta(hours=cfg.ml.retrain_interval_hours + 1)).isoformat()
    db.set_cache(
        "training:last_run",
        {market.symbol: stale_time for market in cfg.trading.markets},
    )

    assert run_paper_signal() == 0
    assert DummyEnsemble.train_calls, "Expected retraining when cached timestamp is stale"

    updated = db.get_cache("training:last_run")
    assert isinstance(updated, dict)
    first_symbol = cfg.trading.markets[0].symbol
    assert first_symbol in updated
    updated_time = datetime.fromisoformat(updated[first_symbol])
    assert updated_time > datetime.fromisoformat(stale_time)


def test_signal_skips_retrain_when_recent(test_env, monkeypatch):
    DummyEnsemble.default_loaded = True
    DummyEnsemble.train_calls = []
    monkeypatch.setattr("src.ml_models.QuantumEnsemble", DummyEnsemble)

    cfg = load_config()
    db_path = test_env / cfg.system.state_dir / cfg.system.database_file
    db = DatabaseManager(db_path)
    recent_time = utc_now().isoformat()
    db.set_cache(
        "training:last_run",
        {market.symbol: recent_time for market in cfg.trading.markets},
    )

    assert run_paper_signal() == 0
    assert not DummyEnsemble.train_calls, "Expected no retraining when cache is recent"
