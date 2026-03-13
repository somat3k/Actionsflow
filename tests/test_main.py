from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.config import load_config
from src.database_manager import DatabaseManager
from src.main import (
    _get_hyperliquid_private_key,
    _resolve_trading_eligibility,
    main,
    run_full_cycle,
    run_paper_signal,
)
from src.utils import utc_now

TEST_RETRAIN_BUFFER_HOURS = 1


class DummyEnsemble:
    default_loaded = True
    instances = []

    def __init__(self, config):
        self.cfg = config
        self.loaded = self.default_loaded
        self.train_calls = []
        self.__class__.instances.append(self)

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
    DummyEnsemble.instances = []
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
    monkeypatch.setattr("src.ml_models.QuantumEnsemble", DummyEnsemble)

    cfg = load_config()
    db_path = test_env / cfg.system.state_dir / cfg.system.database_file
    db = DatabaseManager(db_path)
    stale_time = (
        utc_now() - timedelta(hours=cfg.ml.retrain_interval_hours + TEST_RETRAIN_BUFFER_HOURS)
    ).isoformat()
    db.set_cache(
        "training:last_run",
        {market.symbol: stale_time for market in cfg.trading.markets},
    )

    assert run_paper_signal() == 0
    assert any(
        inst.train_calls for inst in DummyEnsemble.instances
    ), "Expected retraining when cached timestamp is stale"

    db.redis.delete("training:last_run")
    updated = db.get_cache("training:last_run")
    assert isinstance(updated, dict)
    first_symbol = cfg.trading.markets[0].symbol
    assert first_symbol in updated
    updated_time = datetime.fromisoformat(updated[first_symbol])
    assert updated_time > datetime.fromisoformat(stale_time)


def test_signal_skips_retrain_when_recent(test_env, monkeypatch):
    DummyEnsemble.default_loaded = True
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
    assert not any(
        inst.train_calls for inst in DummyEnsemble.instances
    ), "Expected no retraining when cache is recent"


def test_full_cycle_run_type_invokes_handler(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()

    with patch("src.main.run_full_cycle", return_value=0) as mock_cycle:
        assert main(["--run-type", "full-cycle", "--mode", "test"]) == 0
    mock_cycle.assert_called_once()


def test_resolve_trading_eligibility_uses_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADING_ELIGIBILITY_OVERRIDE", raising=False)
    cfg = load_config()
    db_path = tmp_path / cfg.system.state_dir / cfg.system.database_file
    db = DatabaseManager(db_path)
    db.set_cache(
        "evaluation:last_metrics",
        {"pass": True, "pause_trading": False, "pause_reason": ""},
    )

    allowed, reason = _resolve_trading_eligibility(db)
    assert allowed
    assert "passed" in reason.lower()


def test_resolve_trading_eligibility_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_ELIGIBILITY_OVERRIDE", "true")
    cfg = load_config()
    db_path = tmp_path / cfg.system.state_dir / cfg.system.database_file
    db = DatabaseManager(db_path)

    allowed, reason = _resolve_trading_eligibility(db)
    assert allowed
    assert "override" in reason.lower()


def _fail_live_signal(*_args, **_kwargs):
    pytest.fail("live trading ran")


def test_full_cycle_sequences_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()

    calls = []

    def _record(name):
        def _runner(*_args, **_kwargs):
            calls.append(name)
            return 0
        return _runner

    cfg = load_config()
    db_path = tmp_path / cfg.system.state_dir / cfg.system.database_file
    monkeypatch.setattr("src.main._build_db_manager", lambda _cfg: DatabaseManager(db_path))
    monkeypatch.setattr("src.main.run_training", _record("training"))
    monkeypatch.setattr("src.main.run_paper_signal", _record("signal"))
    monkeypatch.setattr("src.main.run_evaluation", _record("evaluate"))
    monkeypatch.setattr("src.main.run_model_export", _record("export"))
    monkeypatch.setattr("src.main._resolve_trading_eligibility", lambda _db: (True, "ok"))

    assert run_full_cycle() == 0
    assert calls == ["training", "signal", "evaluate", "export"]


def test_full_cycle_skips_live_trading_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()

    cfg = load_config()
    db_path = tmp_path / cfg.system.state_dir / cfg.system.database_file
    monkeypatch.setattr("src.main._build_db_manager", lambda _cfg: DatabaseManager(db_path))
    monkeypatch.setattr("src.main.run_training", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main.run_paper_signal", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main.run_evaluation", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main.run_model_export", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main._resolve_trading_eligibility", lambda _db: (True, "ok"))
    monkeypatch.setattr("src.main._is_live_trading_enabled", lambda: False)
    monkeypatch.setattr("src.main.run_live_signal", _fail_live_signal)

    assert run_full_cycle() == 0


def test_full_cycle_blocks_live_trading_when_ineligible(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()

    cfg = load_config()
    db_path = tmp_path / cfg.system.state_dir / cfg.system.database_file
    monkeypatch.setattr("src.main._build_db_manager", lambda _cfg: DatabaseManager(db_path))
    monkeypatch.setattr("src.main.run_training", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main.run_paper_signal", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main.run_evaluation", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main.run_model_export", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("src.main._resolve_trading_eligibility", lambda _db: (False, "no"))
    monkeypatch.setattr("src.main._is_live_trading_enabled", lambda: True)

    monkeypatch.setattr("src.main.run_live_signal", _fail_live_signal)

    assert run_full_cycle() == 0
