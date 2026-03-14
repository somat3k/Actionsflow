"""Tests for AI provider and ML model health checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ai_orchestrator as ao
from src.config import load_config


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeAgentOrchestratorOK:
    """Fake AgentOrchestrator that always returns a valid analysis response."""

    def __init__(self, config):
        self.available = True

    def analyse_market_context(self, symbol, ml_signal, market_snapshot):
        return {
            "validated_signal": 0,
            "confidence_adjustment": 0.0,
            "regime": "ranging",
            "reasoning": "health probe",
            "risk_flags": [],
        }

    def recommend_leverage(self, *args, **kwargs):
        return {"recommended_leverage": 20, "reasoning": "probe"}

    def review_performance(self, *args, **kwargs):
        return {
            "adjustments": [],
            "overall_assessment": "ok",
            "pause_trading": False,
            "pause_reason": "",
        }


def _fake_call_ok(self, prompt):
    return (
        '{"validated_signal": 0, "confidence_adjustment": 0.0,'
        ' "regime": "ranging", "reasoning": "probe", "risk_flags": []}'
    )


def _fake_call_error(self, prompt):
    raise RuntimeError("connection refused")


# ── AI Provider Health Check Tests ────────────────────────────────────────────


def test_health_check_all_ok(monkeypatch):
    """All four providers return ok when API keys are set and calls succeed."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)
    monkeypatch.setattr(ao.OpenAICompatibleOrchestrator, "_call_model", _fake_call_ok)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = "openai-key"
    cfg.groq.api_key = "groq-key"

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.health_check()

    by_name = {r["provider"]: r for r in results}
    assert set(by_name) == {"Agent", "OpenRouter", "OpenAI", "Groq"}
    assert all(r["status"] == "ok" for r in results)
    assert all(r["latency_ms"] is not None for r in results)


def test_health_check_unconfigured_providers_skipped(monkeypatch):
    """Providers without API keys are reported as skipped, not error."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)

    cfg = load_config()
    cfg.openrouter.api_key = ""
    cfg.openai.api_key = ""
    cfg.groq.api_key = ""

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.health_check()

    by_name = {r["provider"]: r for r in results}
    assert by_name["Agent"]["status"] == "ok"
    assert by_name["OpenRouter"]["status"] == "skipped"
    assert by_name["OpenAI"]["status"] == "skipped"
    assert by_name["Groq"]["status"] == "skipped"
    # Skipped entries carry no latency
    assert by_name["OpenRouter"]["latency_ms"] is None


def test_health_check_provider_error_reported(monkeypatch):
    """A provider that raises an exception is reported with status='error'."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)
    monkeypatch.setattr(ao.OpenAICompatibleOrchestrator, "_call_model", _fake_call_error)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = ""
    cfg.groq.api_key = ""

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.health_check()

    by_name = {r["provider"]: r for r in results}
    assert by_name["OpenRouter"]["status"] == "error"
    assert "connection refused" in by_name["OpenRouter"]["error"]
    # Agent (fake) still passes
    assert by_name["Agent"]["status"] == "ok"


def test_health_check_empty_response_reported_as_error(monkeypatch):
    """A provider returning None/empty dict is reported as error."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)

    def _fake_call_none(self, prompt):
        return None  # triggers empty response path

    monkeypatch.setattr(ao.OpenAICompatibleOrchestrator, "_call_model", _fake_call_none)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = ""
    cfg.groq.api_key = ""

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.health_check()

    by_name = {r["provider"]: r for r in results}
    assert by_name["OpenRouter"]["status"] == "error"
    assert "empty" in by_name["OpenRouter"]["error"]


# ── ML Model Health Check Tests ───────────────────────────────────────────────


def test_ml_health_check_not_loaded():
    """Returns a single 'not_loaded' entry when no models have been trained."""
    from src.ml_models import QuantumEnsemble

    cfg = load_config()
    ensemble = QuantumEnsemble(cfg)
    results = ensemble.health_check()

    assert len(results) == 1
    assert results[0]["status"] == "not_loaded"
    assert results[0]["signal"] is None


def _make_enriched_df(n: int = 150) -> "pd.DataFrame":
    """Minimal OHLCV + technical-indicator DataFrame for ensemble tests."""
    rng = np.random.default_rng(42)
    prices = 40_000 + np.cumsum(rng.standard_normal(n) * 100)
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices,
        "volume": rng.uniform(1_000, 5_000, n),
    })
    for col in [
        "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
        "ema_9", "ema_21", "ema_50", "bb_upper", "bb_middle", "bb_lower",
        "bb_bandwidth", "atr_14", "adx_14", "stochastic_k", "stochastic_d",
        "cci_20", "williams_r", "vwap", "obv", "ichimoku_tenkan",
        "ichimoku_kijun", "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
        "vol_20", "vol_5",
    ]:
        df[col] = rng.standard_normal(n).astype(np.float32)
    return df


@pytest.mark.slow
def test_ml_health_check_after_training(tmp_path):
    """All loaded sub-models pass the inference probe after training."""
    from src.ml_models import QuantumEnsemble

    cfg = load_config()
    cfg.ml.model_save_dir = str(tmp_path)

    df = _make_enriched_df(150)

    ensemble = QuantumEnsemble(cfg)
    ensemble.train(df, symbol="TEST", save=True)

    results = ensemble.health_check()
    by_model = {r["model"]: r for r in results}

    # The three always-available models must be ok
    assert by_model["gb"]["status"] == "ok"
    assert by_model["rf"]["status"] == "ok"
    assert by_model["linear"]["status"] == "ok"

    # Full ensemble probe must be included and pass
    assert "ensemble (combined)" in by_model
    assert by_model["ensemble (combined)"]["status"] == "ok"

    # Every ok result has a valid signal and a non-negative latency
    for r in results:
        if r["status"] == "ok":
            assert r["signal"] in (0, 1, 2)
            assert r["latency_ms"] is not None and r["latency_ms"] >= 0


# ── run_health_check integration test ─────────────────────────────────────────


def test_run_health_check_exits_zero(monkeypatch, tmp_path):
    """run_health_check() completes and returns 0 even when no models exist."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)

    import os
    monkeypatch.setenv("TRADING_STATE_DIR", str(tmp_path))

    from src.main import run_health_check

    result = run_health_check()
    assert result == 0


# ── Orchestration Probe Tests ─────────────────────────────────────────────────


def test_orchestration_probe_all_ok(monkeypatch):
    """All three pipeline steps pass for each configured provider."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)
    monkeypatch.setattr(ao.OpenAICompatibleOrchestrator, "_call_model", _fake_call_ok)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = "openai-key"
    cfg.groq.api_key = "groq-key"

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.orchestration_probe()

    # Each provider should produce 3 step results
    by_provider_step = {(r["provider"], r["step"]): r for r in results}
    for provider in ("Agent", "OpenRouter", "OpenAI", "Groq"):
        for step in ("market_context", "leverage", "performance"):
            key = (provider, step)
            assert key in by_provider_step, f"Missing result for {key}"
            assert by_provider_step[key]["status"] == "ok", (
                f"{key} status={by_provider_step[key]['status']} "
                f"error={by_provider_step[key].get('error')}"
            )
            assert by_provider_step[key]["latency_ms"] is not None


def test_orchestration_probe_unconfigured_skipped(monkeypatch):
    """Providers without API keys produce skipped steps, not errors."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)

    cfg = load_config()
    cfg.openrouter.api_key = ""
    cfg.openai.api_key = ""
    cfg.groq.api_key = ""

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.orchestration_probe()

    by_provider_step = {(r["provider"], r["step"]): r for r in results}
    # Agent should be ok on all steps
    for step in ("market_context", "leverage", "performance"):
        assert by_provider_step[("Agent", step)]["status"] == "ok"


def test_orchestration_probe_error_reported(monkeypatch):
    """A provider that raises on every call records 'error' for that step."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)
    monkeypatch.setattr(ao.OpenAICompatibleOrchestrator, "_call_model", _fake_call_error)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = ""
    cfg.groq.api_key = ""

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.orchestration_probe()

    by_provider_step = {(r["provider"], r["step"]): r for r in results}
    for step in ("market_context", "leverage", "performance"):
        assert by_provider_step[("OpenRouter", step)]["status"] == "error"
        assert "connection refused" in by_provider_step[("OpenRouter", step)]["error"]
    # Agent (fake) still passes all steps
    for step in ("market_context", "leverage", "performance"):
        assert by_provider_step[("Agent", step)]["status"] == "ok"


def test_orchestration_probe_no_providers_uses_fallback(monkeypatch):
    """When no primary providers are configured, the Agent fallback is still probed."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestratorOK)

    cfg = load_config()
    cfg.openrouter.api_key = ""
    cfg.openai.api_key = ""
    cfg.groq.api_key = ""

    orchestrator = ao.MultiAIOrchestrator(cfg)
    results = orchestrator.orchestration_probe()

    # Should still have probe entries (fallback Agent via AgentProvider wrapper)
    assert len(results) > 0
