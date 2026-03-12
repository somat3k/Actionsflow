from __future__ import annotations

from src.config import load_config
from src.gemini_orchestrator import GeminiOrchestrator


class _FakeModelInfo:
    def __init__(self, name: str, methods: list[str]):
        self.name = name
        self.supported_generation_methods = methods


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


def test_model_not_found_auto_switches_to_supported_fallback(monkeypatch):
    from src import gemini_orchestrator as go

    go._MODEL_LIST_CACHE.clear()

    cfg = load_config()
    cfg.gemini.api_key = "test-key"
    cfg.gemini.model = "gemini-1.5-pro"

    calls: list[str] = []

    class FakeModel:
        def __init__(self, model_name: str, system_instruction: str):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config):
            calls.append(self.model_name)
            if self.model_name == "gemini-1.5-pro":
                raise Exception(
                    "404 models/gemini-1.5-pro is not found for API version v1beta, "
                    "or is not supported for generateContent."
                )
            return _FakeResponse(
                '{"validated_signal": 1, "confidence_adjustment": 0.0, '
                '"regime": "ranging", "reasoning": "ok", "risk_flags": []}'
            )

    class FakeGenAI:
        @staticmethod
        def configure(api_key: str):
            return None

        @staticmethod
        def list_models():
            return [
                _FakeModelInfo("models/gemini-1.5-pro", ["generateContent"]),
                _FakeModelInfo("models/gemini-1.5-flash", ["generateContent"]),
            ]

        GenerativeModel = FakeModel

    monkeypatch.setattr(go, "_GENAI_AVAILABLE", True)
    monkeypatch.setattr(go, "genai", FakeGenAI)

    orchestrator = GeminiOrchestrator(cfg)
    result = orchestrator.analyse_market_context("BTC", {"signal": 1}, {})

    assert result["regime"] == "ranging"
    assert calls == ["gemini-1.5-pro", "gemini-1.5-flash"]
    assert orchestrator.gcfg.model == "gemini-1.5-flash"


def test_model_not_found_without_supported_model_uses_heuristic_fallback(monkeypatch):
    from src import gemini_orchestrator as go

    go._MODEL_LIST_CACHE.clear()

    cfg = load_config()
    cfg.gemini.api_key = "test-key"
    cfg.gemini.model = "gemini-1.5-pro"

    class FakeModel:
        def __init__(self, model_name: str, system_instruction: str):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config):
            raise Exception(
                "404 models/gemini-1.5-pro is not found for API version v1beta, "
                "or is not supported for generateContent."
            )

    class FakeGenAI:
        @staticmethod
        def configure(api_key: str):
            return None

        @staticmethod
        def list_models():
            return []

        GenerativeModel = FakeModel

    monkeypatch.setattr(go, "_GENAI_AVAILABLE", True)
    monkeypatch.setattr(go, "genai", FakeGenAI)

    orchestrator = GeminiOrchestrator(cfg)
    result = orchestrator.analyse_market_context("BTC", {"signal": 2}, {})

    assert result["validated_signal"] == 2
    assert result["reasoning"] == "Gemini unavailable – using raw ML signal"


def test_model_not_found_with_only_same_model_uses_heuristic_fallback(monkeypatch):
    from src import gemini_orchestrator as go

    go._MODEL_LIST_CACHE.clear()

    cfg = load_config()
    cfg.gemini.api_key = "test-key"
    cfg.gemini.model = "gemini-1.5-pro"

    calls: list[str] = []

    class FakeModel:
        def __init__(self, model_name: str, system_instruction: str):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config):
            calls.append(self.model_name)
            raise Exception(
                "404 models/gemini-1.5-pro is not found for API version v1beta, "
                "or is not supported for generateContent."
            )

    class FakeGenAI:
        @staticmethod
        def configure(api_key: str):
            return None

        @staticmethod
        def list_models():
            return [
                _FakeModelInfo("models/gemini-1.5-pro", ["generateContent"]),
            ]

        GenerativeModel = FakeModel

    monkeypatch.setattr(go, "_GENAI_AVAILABLE", True)
    monkeypatch.setattr(go, "genai", FakeGenAI)

    orchestrator = GeminiOrchestrator(cfg)
    result = orchestrator.analyse_market_context("BTC", {"signal": 0}, {})

    assert calls == ["gemini-1.5-pro"]
    assert result["validated_signal"] == 0
    assert result["reasoning"] == "Gemini unavailable – using raw ML signal"
