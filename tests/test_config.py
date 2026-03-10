from __future__ import annotations

from src.config import load_config


def test_data_intervals_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("PRIMARY_INTERVAL", "1m")
    monkeypatch.setenv("SECONDARY_INTERVAL", "5m")
    monkeypatch.setenv("MACRO_INTERVAL", "15m")

    cfg = load_config()

    assert cfg.data.primary_interval == "1m"
    assert cfg.data.secondary_interval == "5m"
    assert cfg.data.macro_interval == "15m"
