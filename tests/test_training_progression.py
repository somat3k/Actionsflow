from __future__ import annotations

from src.config import load_config
from src.ml_models import QuantumEnsemble


def test_reinforcement_updates_model_weights():
    cfg = load_config()
    ensemble = QuantumEnsemble(cfg)
    before = dict(ensemble._model_weights)

    updated = ensemble.apply_reinforcement({"xgb": 0.9, "gb": 0.1}, alpha=0.5)

    assert updated["xgb"] > before["xgb"]
    assert abs(sum(updated.values()) - 1.0) < 1e-6
