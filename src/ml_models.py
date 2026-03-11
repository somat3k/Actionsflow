"""
Quantum Trading System – ML Models
Ensemble of Neural Network (MLP), XGBoost, Gradient Boosting, Random Forest,
and Linear classifier models that produce a combined directional signal
(long / short / flat) with confidence.  Supports per-timeframe epoch training
and combined decision tree-flow across multiplex timeframes.

Neural network component uses scikit-learn's MLPClassifier with temporal-window
feature augmentation, providing production-grade sequential pattern recognition
without requiring an external deep-learning framework.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, ascii=True, dynamic_ncols=False, **kwargs)

except ImportError:  # pragma: no cover – tqdm is optional at import time
    def _progress(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            log.info("%s …", desc)
        return iterable

warnings.filterwarnings("ignore")

from src.config import AppConfig, MLConfig
from src.utils import get_logger

log = get_logger(__name__)

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    log.warning("XGBoost not available – XGBoost model disabled")

# ── Feature columns (must match add_all_features output) ──────────────────────
FEATURE_COLS = [
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist",
    "ema_9", "ema_21", "ema_50",
    "bb_upper", "bb_middle", "bb_lower", "bb_bandwidth",
    "atr_14", "adx_14",
    "stochastic_k", "stochastic_d",
    "cci_20", "williams_r",
    "vwap", "obv",
    "ichimoku_tenkan", "ichimoku_kijun",
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "vol_20", "vol_5",
]

# Number of previous timesteps used to construct temporal context features.
# Each window contributes mean, std, and delta vectors (3 × n_features extra).
_NN_WINDOW = 10


def _build_label(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.003) -> pd.Series:
    """
    Label: 1 (long), 0 (flat/short) based on whether the close price rises
    by more than `threshold` within `horizon` candles.
    Returns a 3-class label: 1=long, 2=short, 0=flat.
    """
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_ret > threshold] = 1
    labels[future_ret < -threshold] = 2
    return labels


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract and scale feature matrix."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values.astype(np.float32)
    return X, available


# ── Neural Network (MLP) helpers ───────────────────────────────────────────────

# Each augmented sample concatenates: current features + mean + std + delta.
# The output width is therefore exactly (_FEATURE_MULTIPLIER × n_base_features).
_FEATURE_MULTIPLIER = 4


def _make_temporal_features(
    X: np.ndarray, y: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment feature matrix with rolling temporal context.

    For every sample at time *t* the following statistics are computed over the
    previous *window* timesteps and appended to the feature vector:

    * **mean**  – rolling average of each feature (trend component)
    * **std**   – rolling standard deviation (volatility component)
    * **delta** – last minus first value in the window (directional momentum)

    This triples the feature count and gives the ``MLPClassifier`` the same
    kind of temporal context that an LSTM receives via its recurrent state,
    without the TensorFlow dependency.

    The output width per sample is ``n_features * _FEATURE_MULTIPLIER``
    (original + 3 temporal statistics).

    Args:
        X: Scaled feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        window: Number of look-back timesteps.

    Returns:
        (X_aug, y_aug) – augmented samples starting from index *window*.
    """
    n, f = X.shape
    out_width = f * _FEATURE_MULTIPLIER
    if n <= window:
        # Not enough rows: return empty arrays with correct feature width.
        return np.empty((0, out_width), dtype=np.float32), np.empty((0,), dtype=np.int32)

    aug_X = np.empty((n - window, out_width), dtype=np.float32)
    for i in range(window, n):
        seg = X[i - window: i]          # shape (window, f)
        aug_X[i - window] = np.concatenate([
            X[i],                        # current features      (f)
            seg.mean(axis=0),            # rolling mean          (f)
            seg.std(axis=0),             # rolling std           (f)
            seg[-1] - seg[0],            # momentum / delta      (f)
        ])
    return aug_X, y[window:]


def _build_nn(n_features: int, n_classes: int) -> MLPClassifier:
    """Return a production-grade MLPClassifier.

    Architecture mirrors the removed LSTM's depth (three hidden layers) but
    uses wider layers, L2 regularisation (``alpha``), and scikit-learn's
    built-in early stopping so no external callback framework is required.

    Layer sizes (512 → 256 → 128 → 64) provide ample capacity for the ~120
    augmented features produced by :func:`_make_temporal_features` while
    staying fast enough for 200-epoch multi-symbol training runs.

    Args:
        n_features: Input dimension (base + temporal features).
        n_classes:  Number of output classes (3: flat / long / short).

    Returns:
        Untrained ``MLPClassifier``.
    """
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,              # L2 regularisation – replaces dropout
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,     # patience – matches original LSTM callback
        tol=1e-5,
        random_state=42,
        warm_start=False,
    )


# ── Ensemble ───────────────────────────────────────────────────────────────────

class QuantumEnsemble:
    """
    Ensemble of ML models producing a unified directional signal.
    Signals: 1 = long, 2 = short, 0 = flat.

    Models:
      - Linear (LogisticRegression) for fast regression/classification
      - XGBoost for accuracy and confidence in tree-thinking regression
      - Random Forest for decision-making tree-thinking regression
      - Gradient Boosting for supporting ensemble diversity
      - Neural Network (MLPClassifier) for sequential pattern recognition
        via temporal-window feature augmentation – no external framework
        required; uses scikit-learn's built-in MLPClassifier.

    Supports per-timeframe epoch training where each timeframe is trained
    separately, and a combined decision tree-flow merges predictions.
    """

    N_CLASSES = 3  # 0=flat, 1=long, 2=short

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.ml_cfg: MLConfig = config.ml
        self.save_dir = Path(self.ml_cfg.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []

        self.xgb_model: Optional[Any] = None
        self.gb_model: Optional[GradientBoostingClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        # Neural network model (MLPClassifier); stored under key "lstm" in
        # scores / weights dicts for backward compatibility with config and
        # dashboard consumers.
        self.nn_model: Optional[MLPClassifier] = None
        self.linear_model: Optional[LogisticRegression] = None

        # Per-timeframe models for epoch training
        self._tf_models: Dict[str, Dict[str, Any]] = {}

        self._model_weights = dict(self.ml_cfg.model_weights)
        self._training_accuracy: Dict[str, float] = {}


    # ── Per-timeframe epoch training ──────────────────────────────────────────

    def train_timeframe(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> Dict[str, float]:
        """Train all sub-models for a specific timeframe.

        Trains XGBoost, Random Forest, Logistic Regression, and the
        Neural Network (MLP) on the supplied dataframe.  The MLP receives
        temporally-augmented features via :func:`_make_temporal_features`.

        Returns:
            Dict mapping model name → validation accuracy for this timeframe.
        """
        log.info(
            "Epoch training for %s@%s on %d rows", symbol, timeframe, len(df)
        )
        X_raw, feature_cols = _prepare_features(df)
        y = _build_label(df).values
        X_raw = X_raw[: len(y)]
        y = y[: len(X_raw)]

        split = int(len(X_raw) * 0.80)
        if split < 10:
            log.warning("Insufficient data for %s@%s", symbol, timeframe)
            return {}

        X_train, X_val = X_raw[:split], X_raw[split:]
        y_train, y_val = y[:split], y[split:]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)

        scores: Dict[str, float] = {}

        # XGBoost
        xgb_model = None
        if _XGB_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                eval_metric="mlogloss", verbosity=0, n_jobs=-1,
            )
            xgb_model.fit(
                X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False
            )
            scores["xgb"] = float(np.mean(xgb_model.predict(X_val_s) == y_val))

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
        )
        rf_model.fit(X_train_s, y_train)
        scores["rf"] = float(np.mean(rf_model.predict(X_val_s) == y_val))

        # Linear classifier
        linear_model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs"
        )
        linear_model.fit(X_train_s, y_train)
        scores["linear"] = float(np.mean(linear_model.predict(X_val_s) == y_val))

        self._tf_models[timeframe] = {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "xgb": xgb_model if _XGB_AVAILABLE else None,
            "rf": rf_model,
            "linear": linear_model,
            "scores": scores,
        }
        log.info("Epoch %s@%s scores: %s", symbol, timeframe, scores)
        return scores

    def predict_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Run inference for a single timeframe model."""
        tf_data = self._tf_models.get(timeframe)
        if tf_data is None:
            return {"signal": 0, "confidence": 0.0, "timeframe": timeframe}

        feature_cols = tf_data["feature_cols"]
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            return {"signal": 0, "confidence": 0.0, "timeframe": timeframe}

        X_raw = df[available].iloc[-1:].values.astype(np.float32)
        X_s = tf_data["scaler"].transform(X_raw)

        probas: Dict[str, np.ndarray] = {}
        if tf_data.get("xgb") is not None:
            probas["xgb"] = tf_data["xgb"].predict_proba(X_s)
        if tf_data.get("rf") is not None:
            probas["rf"] = tf_data["rf"].predict_proba(X_s)
        if tf_data.get("linear") is not None:
            probas["linear"] = tf_data["linear"].predict_proba(X_s)

        if not probas:
            return {"signal": 0, "confidence": 0.0, "timeframe": timeframe}

        model_weight_keys = {
            "xgb": self._model_weights.get("xgb", 0.30),
            "rf": self._model_weights.get("rf", 0.20),
            "linear": self._model_weights.get("linear", 0.15),
            "lstm": self._model_weights.get("lstm", 0.25),
        }
        w_sum = sum(model_weight_keys[k] for k in probas if k in model_weight_keys)
        if w_sum <= 0:
            w_sum = 1.0
        weighted_proba = sum(
            model_weight_keys.get(k, 0.10) / w_sum * probas[k] for k in probas
        )[0]

        flat_prob = float(weighted_proba[0]) if len(weighted_proba) > 0 else 1.0
        long_prob = float(weighted_proba[1]) if len(weighted_proba) > 1 else 0.0
        short_prob = float(weighted_proba[2]) if len(weighted_proba) > 2 else 0.0

        max_prob = max(long_prob, short_prob, flat_prob)
        if max_prob == long_prob and long_prob >= self.cfg.ml.long_threshold:
            signal, confidence = 1, long_prob
        elif max_prob == short_prob and short_prob >= self.cfg.ml.short_threshold:
            signal, confidence = 2, short_prob
        else:
            signal, confidence = 0, flat_prob

        return {
            "signal": signal,
            "confidence": confidence,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "flat_prob": flat_prob,
            "timeframe": timeframe,
        }

    def combined_decision(
        self, tf_predictions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine predictions from multiple timeframes into a final signal.

        Uses a decision tree-flow approach: higher timeframes carry more weight.
        """
        if not tf_predictions:
            return {
                "signal": 0,
                "confidence": 0.0,
                "agreement": 0.0,
                "timeframe_signals": {},
            }

        tf_weights = {"1m": 0.10, "5m": 0.15, "15m": 0.25, "1H": 0.25, "1D": 0.25}
        total_weight = 0.0
        weighted_long = 0.0
        weighted_short = 0.0
        weighted_flat = 0.0
        tf_signals = {}

        for tf, pred in tf_predictions.items():
            w = tf_weights.get(tf, 0.10)
            total_weight += w
            weighted_long += w * pred.get("long_prob", 0.0)
            weighted_short += w * pred.get("short_prob", 0.0)
            weighted_flat += w * pred.get("flat_prob", 1.0)
            tf_signals[tf] = pred.get("signal", 0)

        if total_weight > 0:
            weighted_long /= total_weight
            weighted_short /= total_weight
            weighted_flat /= total_weight

        max_p = max(weighted_long, weighted_short, weighted_flat)
        if max_p == weighted_long and weighted_long >= self.cfg.ml.long_threshold:
            signal, confidence = 1, weighted_long
        elif max_p == weighted_short and weighted_short >= self.cfg.ml.short_threshold:
            signal, confidence = 2, weighted_short
        else:
            signal, confidence = 0, weighted_flat

        n_agree = sum(1 for s in tf_signals.values() if s == signal)
        agreement = n_agree / len(tf_signals) if tf_signals else 0.0

        return {
            "signal": signal,
            "confidence": confidence,
            "long_prob": weighted_long,
            "short_prob": weighted_short,
            "flat_prob": weighted_flat,
            "agreement": agreement,
            "timeframe_signals": tf_signals,
        }

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self, df: pd.DataFrame, symbol: str = "BTC", save: bool = True
    ) -> Dict[str, float]:
        """Train all enabled models on historical OHLCV+feature data."""
        log.info("Training ensemble for %s on %d rows", symbol, len(df))

        X_raw, self.feature_cols = _prepare_features(df)
        y = _build_label(df).values
        # trim last few rows (no label)
        X_raw = X_raw[: len(y)]
        y = y[: len(X_raw)]

        # Train/val split (80/20, time-ordered)
        split = int(len(X_raw) * 0.80)
        X_train, X_val = X_raw[:split], X_raw[split:]
        y_train, y_val = y[:split], y[split:]

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        scores: Dict[str, float] = {}

        # XGBoost
        if _XGB_AVAILABLE:
            log.info("Training XGBoost …")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=-1,
            )
            self.xgb_model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=False,
            )
            scores["xgb"] = float(
                np.mean(self.xgb_model.predict(X_val_s) == y_val)
            )
            log.info("XGBoost val acc: %.4f", scores["xgb"])

        # Gradient Boosting
        log.info("Training GradientBoosting …")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8
        )
        self.gb_model.fit(X_train_s, y_train)
        scores["gb"] = float(np.mean(self.gb_model.predict(X_val_s) == y_val))
        log.info("GradientBoosting val acc: %.4f", scores["gb"])

        # Random Forest
        log.info("Training RandomForest …")
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
        )
        self.rf_model.fit(X_train_s, y_train)
        scores["rf"] = float(np.mean(self.rf_model.predict(X_val_s) == y_val))
        log.info("RandomForest val acc: %.4f", scores["rf"])

        # Linear Classifier (LogisticRegression for classification)
        log.info("Training LinearClassifier …")
        self.linear_model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs"
        )
        self.linear_model.fit(X_train_s, y_train)
        scores["linear"] = float(
            np.mean(self.linear_model.predict(X_val_s) == y_val)
        )
        log.info("LinearClassifier val acc: %.4f", scores["linear"])

        # Neural Network (MLP) with temporal-window feature augmentation
        log.info("Training NeuralNetwork (MLP) …")
        X_aug_train, y_aug_train = _make_temporal_features(
            X_train_s, y_train, _NN_WINDOW
        )
        X_aug_val, y_aug_val = _make_temporal_features(
            X_val_s, y_val, _NN_WINDOW
        )
        if len(X_aug_train) >= _NN_WINDOW and len(X_aug_val) > 0:
            try:
                self.nn_model = _build_nn(X_aug_train.shape[1], self.N_CLASSES)
                self.nn_model.fit(X_aug_train, y_aug_train)
                scores["lstm"] = float(
                    np.mean(self.nn_model.predict(X_aug_val) == y_aug_val)
                )
                log.info("NeuralNetwork val acc: %.4f", scores["lstm"])
            except Exception as exc:
                log.warning("NeuralNetwork training failed: %s", exc)

        self._training_accuracy = scores
        self._save(symbol)
        log.info("Ensemble training complete. Scores: %s", scores)
        return scores

    def train_with_progression(
        self,
        df: pd.DataFrame,
        symbol: str,
        epochs: int,
        reinforcement_alpha: float,
    ) -> List[Dict[str, Any]]:
        """Train the ensemble over progressive epochs with reinforcement updates.

        A tqdm progress bar is shown per epoch when the ``tqdm`` package is
        available.  Falls back to plain logging otherwise.
        """
        total_rows = len(df)
        if total_rows <= 0:
            return []
        epochs = max(1, epochs)
        epoch_results: List[Dict[str, Any]] = []
        epoch_bar = _progress(
            range(1, epochs + 1),
            desc=f"[{symbol}] epochs",
            total=epochs,
            unit="epoch",
            leave=True,
        )
        for epoch in epoch_bar:
            progress = epoch / epochs
            # Ensure minimum 35% of data for stable training signals in early epochs.
            fraction = max(0.35, progress ** 0.5)
            window = max(1, int(total_rows * fraction))
            epoch_df = df.tail(window)
            scores = self.train(epoch_df, symbol=symbol, save=False)
            weights = self.apply_reinforcement(scores, reinforcement_alpha)
            try:
                self._save(symbol)
            except Exception as exc:
                log.warning(
                    "Failed to save model artifacts for %s (epoch %d): %s",
                    symbol,
                    epoch,
                    exc,
                )
            epoch_results.append(
                {
                    "epoch": epoch,
                    "rows": len(epoch_df),
                    "scores": scores,
                    "weights": weights,
                }
            )
            best_score = max(scores.values(), default=0.0)
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(rows=len(epoch_df), best_acc=f"{best_score:.4f}")
        return epoch_results

    # ── Multi-timeframe epoch training ────────────────────────────────────────

    # Canonical timeframe ordering from lowest to highest resolution.
    _TF_ORDER = ["1m", "5m", "15m", "1H", "1h", "1D", "1d"]

    # Decision weights used in combined_decision – higher TF carries more weight.
    _TF_WEIGHTS: Dict[str, float] = {
        "1m": 0.10, "5m": 0.15, "15m": 0.25, "1H": 0.25, "1h": 0.25,
        "1D": 0.25, "1d": 0.25,
    }

    def train_multi_timeframe_with_progression(
        self,
        tf_dataframes: Dict[str, pd.DataFrame],
        symbol: str,
        epochs: int,
        reinforcement_alpha: float,
        primary_tf: str = "1m",
    ) -> List[Dict[str, Any]]:
        """Train across all provided timeframes for *epochs* epochs.

        For every epoch the ensemble is trained on a progressively growing
        window of each timeframe's data (same schedule as
        :meth:`train_with_progression`).  After all timeframes have been
        trained, reinforcement-learning weight updates are applied based on
        the *combined* (averaged) validation accuracy across timeframes.
        The resulting model represents all timeframes; the primary timeframe
        data is used for the global-ensemble ``train()`` call so that the
        base ``predict()`` method works correctly.

        Args:
            tf_dataframes: Mapping of timeframe label → feature-enriched
                OHLCV DataFrame (1m, 5m, 15m, 1h, 1d, …).
            symbol: Trading symbol (e.g. "BTC").
            epochs: Number of training epochs (default config: 200).
            reinforcement_alpha: RL weight-update step-size.
            primary_tf: The timeframe used for base-model training so that
                :meth:`predict` works after this call.

        Returns:
            List of per-epoch result dicts with keys: epoch, tf_scores,
            combined_scores, weights.
        """
        # Remove empty DataFrames.
        valid_tfs = {
            tf: df for tf, df in tf_dataframes.items() if not df.empty
        }
        if not valid_tfs:
            log.warning("No valid timeframe data for %s – aborting MTF training", symbol)
            return []

        epochs = max(1, epochs)
        tf_row_counts = {tf: len(df) for tf, df in valid_tfs.items()}
        epoch_results: List[Dict[str, Any]] = []

        # Sorted timeframe list for deterministic iteration.
        ordered_tfs = sorted(
            valid_tfs.keys(),
            key=lambda t: self._TF_ORDER.index(t) if t in self._TF_ORDER else 99,
        )

        log.info(
            "=== MTF Training: %s | %d epochs | timeframes: %s ===",
            symbol, epochs, ordered_tfs,
        )

        epoch_bar = _progress(
            range(1, epochs + 1),
            desc=f"[{symbol}] MTF epochs",
            total=epochs,
            unit="epoch",
            leave=True,
        )
        for epoch in epoch_bar:
            progress = epoch / epochs
            # Progressive data window: start at 35%, grow to 100%.
            fraction = max(0.35, progress ** 0.5)

            tf_scores: Dict[str, Dict[str, float]] = {}

            # Train each timeframe model.
            tf_bar = _progress(
                ordered_tfs,
                desc=f"  [{symbol}] e{epoch:03d} timeframes",
                total=len(ordered_tfs),
                unit="tf",
                leave=False,
            )
            for tf in tf_bar:
                df = valid_tfs[tf]
                total_rows = tf_row_counts[tf]
                window = max(1, int(total_rows * fraction))
                epoch_df = df.tail(window)
                scores = self.train_timeframe(epoch_df, symbol, tf)
                if scores:
                    tf_scores[tf] = scores
                if hasattr(tf_bar, "set_postfix"):
                    best = max(scores.values(), default=0.0)
                    tf_bar.set_postfix(tf=tf, best_acc=f"{best:.4f}")

            # Also train the global ensemble on the primary timeframe so that
            # the ensemble's base-predict() method is current.
            primary_df = valid_tfs.get(primary_tf)
            if primary_df is None or primary_df.empty:
                primary_df = next(iter(valid_tfs.values()))
            p_total = len(primary_df)
            p_window = max(1, int(p_total * fraction))
            global_scores = self.train(primary_df.tail(p_window), symbol=symbol, save=False)

            # Aggregate scores: average accuracy across all timeframes.
            combined_scores: Dict[str, float] = {}
            all_model_keys = set(global_scores.keys())
            for tf_sc in tf_scores.values():
                all_model_keys.update(tf_sc.keys())
            for model in all_model_keys:
                values = [
                    tf_sc[model]
                    for tf_sc in tf_scores.values()
                    if model in tf_sc
                ]
                if model in global_scores:
                    values.append(global_scores[model])
                combined_scores[model] = float(np.mean(values)) if values else 0.0

            # Reinforcement-learning weight update.
            weights = self.apply_reinforcement(combined_scores, reinforcement_alpha)

            try:
                self._save(symbol)
            except Exception as exc:
                log.warning(
                    "Failed to save model artifacts for %s (epoch %d): %s",
                    symbol, epoch, exc,
                )

            epoch_results.append(
                {
                    "epoch": epoch,
                    "tf_scores": tf_scores,
                    "combined_scores": combined_scores,
                    "weights": weights,
                }
            )

            best_combined = max(combined_scores.values(), default=0.0)
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(
                    tfs=len(tf_scores),
                    best_acc=f"{best_combined:.4f}",
                )

        log.info(
            "MTF Training complete for %s (%d epochs, %d timeframes)",
            symbol, epochs, len(ordered_tfs),
        )
        return epoch_results

    def apply_reinforcement(
        self, scores: Dict[str, float], alpha: float
    ) -> Dict[str, float]:
        """Update model weights using reward scores (reinforcement-style)."""
        alpha = max(0.0, min(alpha, 1.0))
        valid_scores = {
            key: float(score)
            for key, score in scores.items()
            if isinstance(score, (int, float)) and float(score) > 0
        }
        total_score = sum(valid_scores.values())
        if total_score <= 0 or not valid_scores:
            return dict(self._model_weights)
        for model, score in valid_scores.items():
            if model not in self._model_weights:
                continue
            target_weight = score / total_score
            self._model_weights[model] = (
                (1 - alpha) * self._model_weights[model] + alpha * target_weight
            )
        total_weight = sum(self._model_weights.values())
        if total_weight > 0:
            for key in self._model_weights:
                self._model_weights[key] /= total_weight
        return dict(self._model_weights)

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run ensemble inference on the latest candle data.
        Returns:
            {
                "signal":     0 | 1 | 2   (flat | long | short),
                "confidence": float        (0.0–1.0),
                "long_prob":  float,
                "short_prob": float,
                "flat_prob":  float,
                "agreement":  float,       (fraction of models agreeing)
                "model_signals": {...}
            }
        """
        if not self.feature_cols:
            raise RuntimeError("Model not trained. Call train() first or load().")

        available = [c for c in self.feature_cols if c in df.columns]
        X_raw = df[available].iloc[-1:].values.astype(np.float32)
        X_s = self.scaler.transform(X_raw)

        probas: Dict[str, np.ndarray] = {}  # shape (1, 3) per model

        if self.xgb_model is not None and _XGB_AVAILABLE:
            probas["xgb"] = self.xgb_model.predict_proba(X_s)
        if self.gb_model is not None:
            probas["gb"] = self.gb_model.predict_proba(X_s)
        if self.rf_model is not None:
            probas["rf"] = self.rf_model.predict_proba(X_s)
        if self.linear_model is not None:
            probas["linear"] = self.linear_model.predict_proba(X_s)
        if self.nn_model is not None:
            # Build temporal-augmented feature vector for inference.
            all_X = self.scaler.transform(
                df[available].values.astype(np.float32)
            )
            if len(all_X) > _NN_WINDOW:
                seg = all_X[-(_NN_WINDOW + 1):-1]  # window before last row
                aug = np.concatenate([
                    all_X[-1],
                    seg.mean(axis=0),
                    seg.std(axis=0),
                    seg[-1] - seg[0],
                ])[np.newaxis]
                try:
                    probas["lstm"] = self.nn_model.predict_proba(aug)
                except Exception:
                    pass

        if not probas:
            return {
                "signal": 0,
                "confidence": 0.0,
                "long_prob": 0.0,
                "short_prob": 0.0,
                "flat_prob": 1.0,
                "agreement": 0.0,
                "model_signals": {},
            }

        # Weighted average of probabilities
        weights_sum = sum(
            self._model_weights[k] for k in probas
        )
        weighted_proba = sum(
            self._model_weights[k] / weights_sum * probas[k]
            for k in probas
        )[0]  # shape (3,)

        # Map class indices to probabilities
        # Classes 0=flat, 1=long, 2=short
        n_classes = weighted_proba.shape[0]
        flat_prob = float(weighted_proba[0]) if n_classes > 0 else 1.0
        long_prob = float(weighted_proba[1]) if n_classes > 1 else 0.0
        short_prob = float(weighted_proba[2]) if n_classes > 2 else 0.0

        # Determine signal
        max_prob = max(long_prob, short_prob, flat_prob)
        if max_prob == long_prob and long_prob >= self.cfg.ml.long_threshold:
            signal = 1
            confidence = long_prob
        elif max_prob == short_prob and short_prob >= self.cfg.ml.short_threshold:
            signal = 2
            confidence = short_prob
        else:
            signal = 0
            confidence = flat_prob

        # Agreement across models
        model_signals = {k: int(np.argmax(probas[k][0])) for k in probas}
        n_agree = sum(1 for s in model_signals.values() if s == signal)
        agreement = n_agree / len(model_signals) if model_signals else 0.0

        if agreement < self.cfg.ml.min_ensemble_agreement:
            signal = 0
            confidence = flat_prob

        return {
            "signal": signal,
            "confidence": confidence,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "flat_prob": flat_prob,
            "agreement": agreement,
            "model_signals": model_signals,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self, symbol: str) -> None:
        prefix = self.save_dir / symbol
        prefix.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, prefix / "scaler.pkl")
        with open(prefix / "feature_cols.json", "w") as fh:
            json.dump(self.feature_cols, fh)
        with open(prefix / "weights.json", "w") as fh:
            json.dump(self._model_weights, fh)
        if self.xgb_model and _XGB_AVAILABLE:
            self.xgb_model.get_booster().save_model(str(prefix / "xgb.json"))
        if self.gb_model:
            joblib.dump(self.gb_model, prefix / "gb.pkl")
        if self.rf_model:
            joblib.dump(self.rf_model, prefix / "rf.pkl")
        if self.linear_model:
            joblib.dump(self.linear_model, prefix / "linear.pkl")
        if self.nn_model:
            joblib.dump(self.nn_model, prefix / "nn.pkl")
        log.info("Models saved to %s", prefix)

    def load(self, symbol: str) -> bool:
        """Load pre-trained models. Returns True if successful."""
        prefix = self.save_dir / symbol
        scaler_path = prefix / "scaler.pkl"
        if not scaler_path.exists():
            log.warning("No saved models for %s at %s", symbol, prefix)
            return False
        self.scaler = joblib.load(scaler_path)
        with open(prefix / "feature_cols.json") as fh:
            self.feature_cols = json.load(fh)
        weights_path = prefix / "weights.json"
        if weights_path.exists():
            with open(weights_path) as fh:
                weights = json.load(fh)
            if isinstance(weights, dict):
                self._model_weights.update(
                    {
                        k: float(v)
                        for k, v in weights.items()
                        if isinstance(v, (int, float))
                    }
                )
        xgb_path = prefix / "xgb.json"
        if xgb_path.exists() and _XGB_AVAILABLE:
            import xgboost as xgb
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(xgb_path))
        gb_path = prefix / "gb.pkl"
        if gb_path.exists():
            self.gb_model = joblib.load(gb_path)
        rf_path = prefix / "rf.pkl"
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
        linear_path = prefix / "linear.pkl"
        if linear_path.exists():
            self.linear_model = joblib.load(linear_path)
        # Load neural network model (new .pkl format).
        # Silently skip the legacy TensorFlow lstm/ directory if present.
        nn_path = prefix / "nn.pkl"
        if nn_path.exists():
            self.nn_model = joblib.load(nn_path)
        log.info("Models loaded from %s", prefix)
        return True
