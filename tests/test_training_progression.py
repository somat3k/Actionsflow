from __future__ import annotations

import pytest

from src.config import load_config
from src.ml_models import QuantumEnsemble


def test_reinforcement_updates_model_weights():
    cfg = load_config()
    ensemble = QuantumEnsemble(cfg)
    before = dict(ensemble._model_weights)

    updated = ensemble.apply_reinforcement({"xgb": 0.9, "gb": 0.1}, alpha=0.5)

    expected_xgb = ((1 - 0.5) * before["xgb"] + 0.5 * 0.9) / 1.25
    expected_gb = ((1 - 0.5) * before["gb"] + 0.5 * 0.1) / 1.25

    assert updated["xgb"] == pytest.approx(expected_xgb, rel=1e-6)
    assert updated["gb"] == pytest.approx(expected_gb, rel=1e-6)
    assert abs(sum(updated.values()) - 1.0) < 1e-6
