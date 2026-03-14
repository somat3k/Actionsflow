from __future__ import annotations

from src import ai_orchestrator as ao
from src.config import load_config


class _FakeAgentOrchestrator:
    def __init__(self, config):
        self.available = True


def test_provider_order_groq_first(monkeypatch):
    """Groq is the primary (first) provider; other providers follow in fallback order."""
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestrator)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = "openai-key"
    cfg.groq.api_key = "groq-key"

    orchestrator = ao.MultiAIOrchestrator(cfg)
    providers = orchestrator._providers

    # Groq must be first (primary fast-path provider).
    assert isinstance(providers[0], ao.OpenAICompatibleOrchestrator)
    assert providers[0].name == "Groq"
    # Remaining providers are Agent, OpenRouter, OpenAI in fallback order.
    assert [p.name for p in providers[1:]] == ["Agent", "OpenRouter", "OpenAI"]


def test_provider_order_excludes_openai_without_key(monkeypatch):
    monkeypatch.setattr(ao, "AgentOrchestrator", _FakeAgentOrchestrator)

    cfg = load_config()
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = ""
    cfg.groq.api_key = "groq-key"

    orchestrator = ao.MultiAIOrchestrator(cfg)
    providers = orchestrator._providers

    # Groq first, then Agent, then OpenRouter (no OpenAI – no key).
    assert providers[0].name == "Groq"
    assert [p.name for p in providers[1:]] == ["Agent", "OpenRouter"]
