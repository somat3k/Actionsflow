from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.ml_models import QuantumEnsemble, _build_label, _make_temporal_features


def test_reinforcement_updates_model_weights(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.chdir(tmp_path)
    ensemble = QuantumEnsemble(cfg)
    before = dict(ensemble._model_weights)

    updated = ensemble.apply_reinforcement({"xgb": 0.9, "gb": 0.1}, alpha=0.5)

    alpha = 0.5
    updated_xgb = (1 - alpha) * before["xgb"] + alpha * 0.9
    updated_gb = (1 - alpha) * before["gb"] + alpha * 0.1
    unchanged_total = sum(
        weight for name, weight in before.items() if name not in {"xgb", "gb"}
    )
    expected_total = updated_xgb + updated_gb + unchanged_total
    expected_xgb = updated_xgb / expected_total
    expected_gb = updated_gb / expected_total

    assert updated["xgb"] == pytest.approx(expected_xgb, rel=1e-6)
    assert updated["gb"] == pytest.approx(expected_gb, rel=1e-6)
    assert abs(sum(updated.values()) - 1.0) < 1e-6


# ── Wormhole-fix tests ────────────────────────────────────────────────────────

def _make_ohlcv_df(n: int, seed: int = 0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with a *close* column."""
    rng = np.random.default_rng(seed)
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })


class TestBuildLabelTrimsHorizon:
    """_build_label must exclude the last *horizon* rows (no valid future return)."""

    def test_output_length_is_n_minus_horizon(self):
        df = _make_ohlcv_df(50)
        horizon = 3
        labels = _build_label(df, horizon=horizon)
        assert len(labels) == len(df) - horizon

    def test_output_length_horizon_one(self):
        df = _make_ohlcv_df(20)
        labels = _build_label(df, horizon=1)
        assert len(labels) == len(df) - 1

    def test_label_values_are_0_1_or_2(self):
        df = _make_ohlcv_df(100)
        labels = _build_label(df)
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_no_nan_in_labels(self):
        df = _make_ohlcv_df(100)
        labels = _build_label(df)
        assert not labels.isna().any()

    def test_last_row_excluded_from_labels(self):
        """The close price of the last horizon rows has no future, so those
        rows must NOT appear in the returned label series."""
        df = _make_ohlcv_df(30)
        labels = _build_label(df, horizon=3)
        # Labels index should stop before the last 3 rows of df.
        last_label_loc = df.index.get_loc(labels.index[-1])
        assert last_label_loc == len(df) - 3 - 1


class TestTemporalFeatureNoBoundaryWormhole:
    """The NN temporal features must span the train/val boundary using
    training data as context – not re-start cold from the first val row."""

    def test_full_dataset_features_match_val_context(self):
        """Augmented features built on the full dataset must give the first
        validation sample context from the last training rows – not from
        other validation rows (the wormhole pattern).

        The test verifies that the rolling-context columns of the boundary
        validation samples differ between the fixed (full-dataset) and the old
        (isolated-val) approaches, because the fixed approach draws context
        from training rows while the old approach draws from other val rows.
        """
        n_total = 50
        window = 5
        n_features = 3
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_total, n_features)).astype(np.float32)
        y = rng.integers(0, 3, n_total)

        split = 40  # 80 % training, 20 % val

        # ── Full-dataset approach (correct, post-fix) ──────────────────────
        X_aug_full, y_aug_full = _make_temporal_features(X, y, window)
        aug_split = split - window
        X_aug_val_full = X_aug_full[aug_split:]  # shape (10, 4*n_features)

        # ── Isolated-val approach (old wormhole approach) ──────────────────
        X_aug_val_iso, _ = _make_temporal_features(X[split:], y[split:], window)
        # shape (5, 4*n_features); first val row here is original row split+window

        # The boundary rows (original rows split to split+window-1) exist only
        # in the full approach; the isolated approach skips them.
        assert len(X_aug_val_full) > len(X_aug_val_iso), (
            "Full-dataset val augmentation should produce more samples than "
            "the isolated-val approach (boundary rows were previously skipped)."
        )

        # For the overlap (rows that appear in both), the current-feature slice
        # must be the same (same original data) but rolling context differs.
        # X_aug_val_full[window] corresponds to original row split+window.
        # X_aug_val_iso[0]      corresponds to original row split+window.
        row_full = X_aug_val_full[window]     # original row split+window
        row_iso = X_aug_val_iso[0]            # original row split+window

        # Current features are identical (same original row).
        np.testing.assert_array_equal(row_full[:n_features], row_iso[:n_features])

        # But rolling context (mean/std/delta) must differ: the full-dataset
        # approach draws context from training rows split..split+window-1 for
        # this overlap row (making it identical to isolated for the overlap),
        # but the boundary rows only exist in the full approach and must draw
        # their context from rows BEFORE split (training data).
        # The REAL difference is for the very first boundary rows where
        # full-dataset context comes from training rows (before split).
        row_boundary = X_aug_val_full[0]      # original row split (boundary)
        # Its rolling mean (cols n_features to 2*n_features) must be computed
        # from training rows split-window..split-1, not from validation rows.
        expected_mean = X[split - window: split].mean(axis=0)
        np.testing.assert_allclose(
            row_boundary[n_features: 2 * n_features],
            expected_mean,
            rtol=1e-5,
            err_msg="Boundary val row's rolling mean should come from training rows.",
        )

    def test_aug_split_aligns_with_original_split(self):
        """Augmented row at index (split-window) must correspond to the first
        row of the original validation set (original index = split)."""
        n_total = 60
        window = 10
        split = 48  # 80 %
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n_total, 4)).astype(np.float32)
        y = rng.integers(0, 3, n_total)

        X_aug, y_aug = _make_temporal_features(X, y, window)
        aug_split = split - window

        # The current-features slice of X_aug[aug_split] must equal X[split].
        n_features = X.shape[1]
        np.testing.assert_array_equal(X_aug[aug_split, :n_features], X[split])

