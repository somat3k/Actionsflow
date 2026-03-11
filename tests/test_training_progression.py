from __future__ import annotations

import pytest

from src.config import load_config
from src.ml_models import QuantumEnsemble


def test_reinforcement_updates_model_weights(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.chdir(tmp_path)
    ensemble = QuantumEnsemble(cfg)
    before = dict(ensemble._model_weights)

    updated = ensemble.apply_reinforcement({"xgb": 0.9, "gb": 0.1}, alpha=0.5)

    alpha = 0.5
    updated_xgb = (1 - alpha) * before["xgb"] + alpha * 0.9
    updated_gb = (1 - alpha) * before["gb"] + alpha * 0.1
    expected_total = (
        sum(before.values()) - before["xgb"] - before["gb"] + updated_xgb + updated_gb
    )
    expected_xgb = updated_xgb / expected_total
    expected_gb = updated_gb / expected_total

    assert updated["xgb"] == pytest.approx(expected_xgb, rel=1e-6)
    assert updated["gb"] == pytest.approx(expected_gb, rel=1e-6)
    assert abs(sum(updated.values()) - 1.0) < 1e-6
