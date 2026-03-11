"""
Dataset management for Hyperliquid OHLCV data.
Stores datasets on disk (safetensors/npz) and tracks metadata in SQLite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.database_manager import DatabaseManager
from src.utils import get_logger

log = get_logger(__name__)

try:
    from safetensors.numpy import load_file as _load_safetensors
    from safetensors.numpy import save_file as _save_safetensors

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


class DatasetManager:
    """Manage cached OHLCV datasets stored on disk and indexed in SQLite."""

    def __init__(self, config: AppConfig, db: DatabaseManager) -> None:
        self.cfg = config
        self.db = db
        self.dataset_dir = Path(config.data.dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_format = config.data.dataset_format.lower()
        if self.dataset_format == "safetensors" and not _SAFETENSORS_AVAILABLE:
            log.warning("safetensors not installed – falling back to npz format")
            self.dataset_format = "npz"

    def get_or_fetch_dataset(
        self,
        fetcher: Any,
        symbol: str,
        interval: str,
        lookback_candles: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        lookback = lookback_candles or self.cfg.data.training_lookback_candles
        if not force_refresh:
            cached = self.db.get_latest_dataset(symbol, interval)
            if cached and cached.get("rows", 0) >= lookback:
                path = Path(cached["path"])
                if path.exists():
                    df = self.load_dataset(path)
                    if not df.empty:
                        log.info("Loaded cached dataset for %s (%s)", symbol, path)
                        return df

        df = fetcher.download_ohlcv_history(symbol, interval, lookback_candles=lookback)
        if df.empty:
            return df

        path = self._build_dataset_path(symbol, interval, df)
        self.save_dataset(df, path)
        self.db.record_dataset(
            symbol=symbol,
            interval=interval,
            start_ms=int(df["open_time"].iloc[0]) if "open_time" in df.columns else None,
            end_ms=int(df["open_time"].iloc[-1]) if "open_time" in df.columns else None,
            rows=len(df),
            path=str(path),
        )
        return df

    def save_dataset(self, df: pd.DataFrame, path: Path) -> None:
        tensors = self._dataframe_to_tensors(df)
        if self.dataset_format == "safetensors":
            _save_safetensors(tensors, str(path))
        else:
            np.savez_compressed(path, **tensors)
        log.info("Saved dataset to %s", path)

    def load_dataset(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".safetensors" and _SAFETENSORS_AVAILABLE:
            tensors = _load_safetensors(str(path))
        else:
            with np.load(path, allow_pickle=False) as loader:
                tensors = {k: loader[k] for k in loader.files}
        return self._tensors_to_dataframe(tensors)

    def _build_dataset_path(self, symbol: str, interval: str, df: pd.DataFrame) -> Path:
        start_ms = int(df["open_time"].iloc[0]) if "open_time" in df.columns else 0
        end_ms = int(df["open_time"].iloc[-1]) if "open_time" in df.columns else 0
        suffix = "safetensors" if self.dataset_format == "safetensors" else "npz"
        filename = f"{symbol.upper()}_{interval}_{start_ms}_{end_ms}.{suffix}"
        return self.dataset_dir / filename

    @staticmethod
    def _dataframe_to_tensors(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        working = df.copy()
        if "open_time" not in working.columns:
            if isinstance(working.index, pd.DatetimeIndex):
                working["open_time"] = working.index.view("int64") // 1_000_000
            else:
                working["open_time"] = np.arange(len(working))
        if "datetime" in working.columns:
            working = working.drop(columns=["datetime"])
        tensors: Dict[str, np.ndarray] = {}
        for col in working.columns:
            series = pd.to_numeric(working[col], errors="coerce")
            if series.isnull().any():
                series = series.fillna(0)
            arr = series.to_numpy()
            if arr.dtype.kind in {"i", "u"}:
                tensors[col] = arr.astype(np.int64)
            else:
                tensors[col] = arr.astype(np.float32)
        return tensors

    @staticmethod
    def _tensors_to_dataframe(tensors: Dict[str, np.ndarray]) -> pd.DataFrame:
        df = pd.DataFrame({k: v for k, v in tensors.items()})
        if "open_time" in df.columns:
            df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
            df = df.sort_values("open_time").reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
            df = df.set_index("datetime")
        return df
