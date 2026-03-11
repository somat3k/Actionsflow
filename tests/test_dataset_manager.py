from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.data_fetcher import HyperliquidDataFetcher
from src.database_manager import DatabaseManager
from src.dataset_manager import DatasetManager


@pytest.fixture
def dataset_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRADING_MODE", "test")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".trading_state").mkdir()
    return tmp_path


def test_dataset_manager_caches_ohlcv(dataset_env: Path):
    cfg = load_config()
    db_path = dataset_env / cfg.system.state_dir / cfg.system.database_file
    db = DatabaseManager(db_path)
    fetcher = HyperliquidDataFetcher(cfg)
    manager = DatasetManager(cfg, db)

    df = manager.get_or_fetch_dataset(
        fetcher,
        symbol="BTC",
        interval=cfg.data.primary_interval,
        lookback_candles=50,
        force_refresh=True,
    )
    assert not df.empty

    cached = db.get_latest_dataset("BTC", cfg.data.primary_interval)
    assert cached is not None
    path = Path(cached["path"])
    assert path.exists()

    df_cached = manager.get_or_fetch_dataset(
        fetcher,
        symbol="BTC",
        interval=cfg.data.primary_interval,
        lookback_candles=10,
        force_refresh=False,
    )
    assert not df_cached.empty
