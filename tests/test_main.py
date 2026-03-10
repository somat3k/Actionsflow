from __future__ import annotations

from src.main import _get_hyperliquid_private_key


def test_hyperliquid_secret_is_preferred_over_legacy_key(monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_SECRET", "secret-key")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "legacy-key")

    assert _get_hyperliquid_private_key() == "secret-key"


def test_hyperliquid_legacy_key_is_used_as_fallback(monkeypatch):
    monkeypatch.delenv("HYPERLIQUID_SECRET", raising=False)
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "legacy-key")

    assert _get_hyperliquid_private_key() == "legacy-key"
