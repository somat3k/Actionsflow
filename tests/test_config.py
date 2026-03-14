from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from src.config import load_config, CacheConfig


def test_data_intervals_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("PRIMARY_INTERVAL", "1m")
    monkeypatch.setenv("SECONDARY_INTERVAL", "5m")
    monkeypatch.setenv("MACRO_INTERVAL", "15m")

    cfg = load_config()

    assert cfg.data.primary_interval == "1m"
    assert cfg.data.secondary_interval == "5m"
    assert cfg.data.macro_interval == "15m"


def test_cache_config_zero_ttl_preserved(tmp_path):
    """TTL=0 (no expiry) must not be replaced by 3600."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump({"cache": {"default_ttl_seconds": 0}}))
    cfg = load_config(cfg_file)
    assert cfg.cache.default_ttl_seconds == 0


def test_cache_config_defaults():
    """CacheConfig defaults match expected values."""
    cc = CacheConfig()
    assert cc.enabled is True
    assert cc.redis_url == ""
    assert cc.default_ttl_seconds == 3600
    assert cc.namespace == "qt"


def test_cache_config_enabled_false_loaded(tmp_path):
    """enabled=false must be respected in loaded config."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump({"cache": {"enabled": False, "namespace": "myns"}}))
    cfg = load_config(cfg_file)
    assert cfg.cache.enabled is False
    assert cfg.cache.namespace == "myns"


def test_redis_url_env_var_overrides_config(tmp_path, monkeypatch):
    """REDIS_URL env var must override the redis_url in YAML."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump({"cache": {"redis_url": "redis://cfg-host:6379"}}))
    monkeypatch.setenv("REDIS_URL", "redis://env-host:6379")
    cfg = load_config(cfg_file)
    assert cfg.cache.redis_url == "redis://env-host:6379"


def test_ml_nn_override_threshold_env_var_takes_precedence(tmp_path, monkeypatch):
    """ML_NN_OVERRIDE_THRESHOLD env var must override YAML/default and be parsed as float."""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump({"ml": {"signals": {"nn_override_threshold": 0.75}}}))

    # Without env var, the YAML value is used.
    cfg_yaml = load_config(cfg_file)
    assert cfg_yaml.ml.nn_override_threshold == pytest.approx(0.75)

    # With env var set, it takes precedence over the YAML value.
    monkeypatch.setenv("ML_NN_OVERRIDE_THRESHOLD", "0.42")
    cfg_env = load_config(cfg_file)
    assert cfg_env.ml.nn_override_threshold == pytest.approx(0.42)
    assert isinstance(cfg_env.ml.nn_override_threshold, float)
