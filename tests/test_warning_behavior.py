"""Focused tests for warning behavior in runtime entrypoints."""

from __future__ import annotations

import importlib
import sys
import warnings

from src.main import main


def test_importing_gemini_orchestrator_suppresses_deprecation_futurewarning():
    sys.modules.pop("src.gemini_orchestrator", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("src.gemini_orchestrator")

    assert not any(
        issubclass(w.category, FutureWarning) and "google.generativeai" in str(w.message)
        for w in caught
    )


def test_evaluate_path_does_not_import_ml_models(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()

    sys.modules.pop("src.ml_models", None)
    exit_code = main(["--run-type", "evaluate", "--mode", "test"])

    assert exit_code == 0
    assert "src.ml_models" not in sys.modules
