"""
Integration tests – complete trading pipeline in test mode.

These tests exercise the full application pipeline end-to-end:
    train-models → signal → evaluate

External API calls are replaced with synthetic data by setting
``TRADING_MODE=test``, so no credentials or network access are required.

Run individually:
    pytest tests/test_integration.py -v

Or as part of the full suite:
    pytest tests/ -v

All tests in this module are marked ``slow`` and are skipped by default.
Run with ``-m slow`` or ``-m ''`` to include them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.main import main, run_training, run_paper_signal, run_evaluation

pytestmark = pytest.mark.slow


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def test_pipeline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Isolated environment for integration tests.

    * Sets TRADING_MODE=test so the data fetcher returns synthetic data.
    * Sets TRAINING_EPOCHS=2 so epoch-based training completes quickly.
    * Changes the working directory to ``tmp_path`` so state/model/result
      files are written there instead of polluting the repository root.
    """
    monkeypatch.setenv("TRADING_MODE", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TRAINING_EPOCHS", "2")
    monkeypatch.chdir(tmp_path)

    # Pre-create directories expected by the pipeline.
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / ".trading_state").mkdir()

    yield tmp_path


# ── Individual stage tests ────────────────────────────────────────────────────

class TestTrainModels:
    def test_train_models_exits_successfully(self, test_pipeline_env):
        """Training stage runs to completion with synthetic market data."""
        exit_code = main(["--run-type", "train-models", "--mode", "test"])
        assert exit_code == 0

    def test_train_models_saves_artifacts(self, test_pipeline_env):
        """Training persists at least one model directory and a scores file."""
        main(["--run-type", "train-models", "--mode", "test"])

        models_dir = test_pipeline_env / "models"
        # At least one symbol directory should be created.
        symbol_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        assert len(symbol_dirs) > 0, "Expected model directories to be created"

        scores_file = test_pipeline_env / "results" / "training_scores.json"
        assert scores_file.exists(), "Expected training_scores.json to be written"

        with open(scores_file) as fh:
            scores = json.load(fh)
        assert isinstance(scores, dict)


class TestPaperSignal:
    def test_signal_after_training_exits_successfully(self, test_pipeline_env):
        """Signal cycle runs without errors after models have been trained."""
        assert main(["--run-type", "train-models", "--mode", "test"]) == 0
        assert main(["--run-type", "signal", "--mode", "test"]) == 0

    def test_signal_saves_broker_state(self, test_pipeline_env):
        """Signal cycle persists broker state to disk."""
        main(["--run-type", "train-models", "--mode", "test"])
        main(["--run-type", "signal", "--mode", "test"])

        broker_state = test_pipeline_env / ".trading_state" / "paper_broker.json"
        assert broker_state.exists(), "Expected paper_broker.json to be saved"

    def test_signal_without_models_exits_successfully(self, test_pipeline_env):
        """
        Signal stage should complete (skipping untrained symbols) even if no
        models are found – the pipeline must not crash on missing models.
        """
        exit_code = main(["--run-type", "signal", "--mode", "test"])
        assert exit_code == 0


class TestEvaluation:
    def test_evaluate_exits_successfully(self, test_pipeline_env):
        """Evaluation stage runs and produces a report."""
        assert main(["--run-type", "train-models", "--mode", "test"]) == 0
        assert main(["--run-type", "signal", "--mode", "test"]) == 0
        assert main(["--run-type", "evaluate", "--mode", "test"]) == 0

    def test_evaluate_writes_report(self, test_pipeline_env):
        """Evaluation saves an evaluation_report.json with expected keys."""
        main(["--run-type", "train-models", "--mode", "test"])
        main(["--run-type", "signal", "--mode", "test"])
        main(["--run-type", "evaluate", "--mode", "test"])

        report_path = test_pipeline_env / "results" / "evaluation_report.json"
        assert report_path.exists(), "Expected evaluation_report.json to be written"

        with open(report_path) as fh:
            report = json.load(fh)
        assert "metrics" in report, "Report must contain a 'metrics' key"
        assert "pass" in report, "Report must contain a 'pass' key"


# ── Full end-to-end pipeline ──────────────────────────────────────────────────

class TestFullPipeline:
    def test_complete_pipeline_succeeds(self, test_pipeline_env):
        """
        Full pipeline runs end-to-end without any errors:
        train-models → signal → evaluate.

        This is the primary integration smoke-test that mirrors the
        GitHub Actions ``integration-test`` job.
        """
        assert main(["--run-type", "train-models", "--mode", "test"]) == 0, (
            "train-models stage failed"
        )
        assert main(["--run-type", "signal", "--mode", "test"]) == 0, (
            "signal stage failed"
        )
        assert main(["--run-type", "evaluate", "--mode", "test"]) == 0, (
            "evaluate stage failed"
        )

        # Verify key artifacts were produced.
        assert (test_pipeline_env / "results" / "training_scores.json").exists()
        assert (test_pipeline_env / ".trading_state" / "paper_broker.json").exists()
        assert (test_pipeline_env / "results" / "evaluation_report.json").exists()
