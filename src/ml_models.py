"""
Quantum Trading System – ML Models
Ensemble of LSTM, XGBoost, Gradient Boosting, and Random Forest models that
produce a combined directional signal (long / short / flat) with confidence.
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from src.config import AppConfig, MLConfig
from src.utils import get_logger

log = get_logger(__name__)

# Try to import optional deep-learning dependency
try:
    import tensorflow as tf
    from tensorflow import keras

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    log.warning("TensorFlow not available – LSTM model disabled")

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


# ── LSTM model ─────────────────────────────────────────────────────────────────

def _build_lstm(
    input_shape: Tuple[int, int],
    n_classes: int,
    units: List[int],
    dropout: float,
) -> "keras.Model":
    model = keras.Sequential()
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        model.add(
            keras.layers.LSTM(
                u,
                return_sequences=return_seq,
                input_shape=input_shape if i == 0 else None,
            )
        )
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


# ── Ensemble ───────────────────────────────────────────────────────────────────

class QuantumEnsemble:
    """
    Ensemble of ML models producing a unified directional signal.
    Signals: 1 = long, 2 = short, 0 = flat.
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
        self.lstm_model: Optional[Any] = None

        self._model_weights = {
            "xgb": 0.35,
            "gb": 0.15,
            "rf": 0.15,
            "lstm": 0.35,
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

        # LSTM
        if _TF_AVAILABLE:
            seq_len = 60
            try:
                log.info("Training LSTM …")
                X_seq_train, y_seq_train = _make_sequences(X_train_s, y_train, seq_len)
                X_seq_val, y_seq_val = _make_sequences(X_val_s, y_val, seq_len)
                if len(X_seq_train) > 0:
                    self.lstm_model = _build_lstm(
                        input_shape=(seq_len, X_train_s.shape[1]),
                        n_classes=self.N_CLASSES,
                        units=[128, 64, 32],
                        dropout=0.2,
                    )
                    early_stop = keras.callbacks.EarlyStopping(
                        patience=15, restore_best_weights=True
                    )
                    self.lstm_model.fit(
                        X_seq_train, y_seq_train,
                        validation_data=(X_seq_val, y_seq_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=0,
                    )
                    lstm_pred = np.argmax(
                        self.lstm_model.predict(X_seq_val, verbose=0), axis=1
                    )
                    scores["lstm"] = float(np.mean(lstm_pred == y_seq_val))
                    log.info("LSTM val acc: %.4f", scores["lstm"])
            except Exception as exc:
                log.warning("LSTM training failed: %s", exc)

        if save:
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
        """Train the ensemble over progressive epochs with reinforcement updates."""
        total_rows = len(df)
        if total_rows <= 0:
            return []
        epochs = max(1, epochs)
        epoch_results: List[Dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
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
        if self.lstm_model is not None and _TF_AVAILABLE:
            # Need sequence; use last 60 rows
            seq_len = 60
            all_X = self.scaler.transform(
                df[available].values.astype(np.float32)
            )
            if len(all_X) >= seq_len:
                seq = all_X[-seq_len:][np.newaxis]
                lstm_proba = self.lstm_model.predict(seq, verbose=0)[0]
                # Ensure shape (1, 3)
                probas["lstm"] = lstm_proba[np.newaxis]

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
        if self.lstm_model and _TF_AVAILABLE:
            self.lstm_model.save(str(prefix / "lstm"))
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
        lstm_path = prefix / "lstm"
        if lstm_path.exists() and _TF_AVAILABLE:
            self.lstm_model = keras.models.load_model(str(lstm_path))
        log.info("Models loaded from %s", prefix)
        return True
