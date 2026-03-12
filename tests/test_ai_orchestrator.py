from __future__ import annotations

from src import ai_orchestrator as ao
from src.config import load_config


class _FakeGeminiOrchestrator:
    def __init__(self, config):
        self._model = object()


def test_provider_order_includes_openai(monkeypatch):
    monkeypatch.setattr(ao, "GeminiOrchestrator", _FakeGeminiOrchestrator)

    cfg = load_config()
    cfg.gemini.api_key = "gemini-key"
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = "openai-key"
    cfg.groq.api_key = "groq-key"

    orchestrator = ao.MultiAIOrchestrator(cfg)
    providers = orchestrator._providers

    assert isinstance(providers[0], ao.GeminiProvider)
    assert [provider.name for provider in providers[1:]] == ["OpenRouter", "OpenAI", "Groq"]


def test_provider_order_excludes_openai_without_key(monkeypatch):
    monkeypatch.setattr(ao, "GeminiOrchestrator", _FakeGeminiOrchestrator)

    cfg = load_config()
    cfg.gemini.api_key = "gemini-key"
    cfg.openrouter.api_key = "openrouter-key"
    cfg.openai.api_key = ""
    cfg.groq.api_key = "groq-key"

    orchestrator = ao.MultiAIOrchestrator(cfg)
    providers = orchestrator._providers

    assert [provider.name for provider in providers[1:]] == ["OpenRouter", "Groq"]
