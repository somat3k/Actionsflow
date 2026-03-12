"""Tests for the embedded Redis controller."""

from __future__ import annotations

import pytest

from src.redis_controller import RedisController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctrl(**kwargs) -> RedisController:
    """Return a RedisController backed by embedded fakeredis."""
    return RedisController(namespace="test", **kwargs)


# ---------------------------------------------------------------------------
# Basic availability
# ---------------------------------------------------------------------------

def test_embedded_is_available():
    ctrl = make_ctrl()
    assert ctrl.is_available
    assert ctrl.is_embedded
    assert ctrl.ping()


# ---------------------------------------------------------------------------
# Set / Get round-trip
# ---------------------------------------------------------------------------

def test_set_and_get_string():
    ctrl = make_ctrl()
    ctrl.set("hello", "world")
    assert ctrl.get("hello") == "world"


def test_get_missing_key_returns_none():
    ctrl = make_ctrl()
    assert ctrl.get("no-such-key") is None


def test_set_overwrites_existing():
    ctrl = make_ctrl()
    ctrl.set("k", "v1")
    ctrl.set("k", "v2")
    assert ctrl.get("k") == "v2"


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

def test_set_with_explicit_ttl():
    ctrl = make_ctrl(default_ttl=None)
    ctrl.set("ttl-key", "value", ttl_seconds=60)
    remaining = ctrl.ttl("ttl-key")
    assert remaining is not None and remaining > 0


def test_set_with_no_ttl():
    ctrl = make_ctrl(default_ttl=None)
    ctrl.set("no-ttl", "value", ttl_seconds=None)
    assert ctrl.ttl("no-ttl") is None  # no expiry


def test_default_ttl_applied_when_minus_one():
    ctrl = make_ctrl(default_ttl=120)
    ctrl.set("default", "x", ttl_seconds=-1)
    remaining = ctrl.ttl("default")
    assert remaining is not None and 0 < remaining <= 120


def test_ttl_of_missing_key_returns_minus_two():
    ctrl = make_ctrl()
    assert ctrl.ttl("ghost") == -2


# ---------------------------------------------------------------------------
# Delete / exists
# ---------------------------------------------------------------------------

def test_delete_removes_key():
    ctrl = make_ctrl()
    ctrl.set("del-me", "bye")
    assert ctrl.exists("del-me")
    ctrl.delete("del-me")
    assert not ctrl.exists("del-me")


def test_delete_missing_key_is_idempotent():
    ctrl = make_ctrl()
    result = ctrl.delete("never-set")
    assert result is False


# ---------------------------------------------------------------------------
# Keys / flush
# ---------------------------------------------------------------------------

def test_keys_lists_all():
    ctrl = make_ctrl()
    ctrl.flush()
    ctrl.set("alpha", "1")
    ctrl.set("beta", "2")
    ctrl.set("gamma", "3")
    found = sorted(ctrl.keys())
    assert found == ["alpha", "beta", "gamma"]


def test_keys_with_glob_pattern():
    ctrl = make_ctrl()
    ctrl.flush()
    ctrl.set("training:btc", "a")
    ctrl.set("training:eth", "b")
    ctrl.set("evaluation:last", "c")
    training_keys = sorted(ctrl.keys("training:*"))
    assert training_keys == ["training:btc", "training:eth"]


def test_flush_removes_all_namespace_keys():
    ctrl = make_ctrl()
    ctrl.set("x", "1")
    ctrl.set("y", "2")
    deleted = ctrl.flush()
    assert deleted >= 2
    assert ctrl.keys() == []


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------

def test_two_controllers_with_different_namespaces_are_isolated():
    c1 = RedisController(namespace="ns1")
    c2 = RedisController(namespace="ns2")
    c1.set("shared-key", "from-ns1")
    assert c2.get("shared-key") is None


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

def test_close_disables_operations():
    ctrl = make_ctrl()
    ctrl.set("before-close", "ok")
    ctrl.close()
    # After close the client is None – operations should return safe defaults
    assert ctrl.get("before-close") is None
    assert ctrl.set("after-close", "x") is False
    assert ctrl.ping() is False
    assert ctrl.keys() == []
