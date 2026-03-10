"""
Quantum Trading System – Configuration Management
Loads and validates trading configuration with environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ── Root directory ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT_DIR / "config" / "trading_config.yaml"


# ── Dataclasses for typed config access ───────────────────────────────────────

@dataclass
class LeverageConfig:
    min: int = 10
    max: int = 35
    default: int = 15
    high_confidence_threshold: float = 0.80
    low_confidence_threshold: float = 0.55
    step: int = 5


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 0.20
    daily_loss_limit_pct: float = 0.05
    max_open_positions: int = 4
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    trailing_stop_pct: float = 0.015


@dataclass
class PositionSizingConfig:
    method: str = "kelly_fractional"
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.30
    min_position_usd: float = 100.0


@dataclass
class MarketConfig:
    symbol: str = "BTC"
    enabled: bool = True
    weight: float = 0.25


@dataclass
class TradingConfig:
    mode: str = "paper"
    initial_equity: float = 10_000.0
    base_currency: str = "USDC"
    markets: List[MarketConfig] = field(default_factory=list)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)


@dataclass
class DataConfig:
    hyperliquid_api_url: str = "https://api.hyperliquid.xyz/info"
    hyperliquid_ws_url: str = "wss://api.hyperliquid.xyz/ws"
    primary_interval: str = "15m"
    secondary_interval: str = "1h"
    macro_interval: str = "4h"
    lookback_candles: int = 500


@dataclass
class MLConfig:
    long_threshold: float = 0.60
    short_threshold: float = 0.60
    close_threshold: float = 0.45
    min_ensemble_agreement: float = 0.60
    model_save_dir: str = "models"
    retrain_interval_hours: int = 24


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-1.5-pro"
    temperature: float = 0.1
    max_output_tokens: int = 2048


@dataclass
class PaperBrokerConfig:
    initial_equity: float = 10_000.0
    taker_fee: float = 0.00045
    maker_fee: float = 0.00020
    funding_enabled: bool = True
    liquidation_enabled: bool = True
    slippage_bps: int = 5


@dataclass
class EvaluationConfig:
    min_sharpe: float = 1.0
    min_win_rate: float = 0.45
    max_drawdown_pct: float = 0.25
    min_profit_factor: float = 1.20
    auto_adjust_enabled: bool = True
    evaluation_window_trades: int = 50


@dataclass
class SystemConfig:
    name: str = "Quantum Trader"
    version: str = "1.0.0"
    log_level: str = "INFO"
    state_dir: str = ".trading_state"
    database_file: str = "workflow_state.db"
    results_dir: str = "results"


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    paper_broker: PaperBrokerConfig = field(default_factory=PaperBrokerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────

def _deep_get(d: dict, *keys, default=None):
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})
    return d if d != {} else default


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load YAML config and apply environment variable overrides."""
    path = config_path or CONFIG_FILE
    raw: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    system_raw = raw.get("system", {})
    trading_raw = raw.get("trading", {})
    data_raw = raw.get("data", {})
    ml_raw = raw.get("ml", {})
    gemini_raw = raw.get("gemini", {})
    paper_raw = raw.get("paper_broker", {})
    eval_raw = raw.get("evaluation", {})

    # ── Markets ───────────────────────────────────────────────
    markets = [
        MarketConfig(
            symbol=m["symbol"],
            enabled=m.get("enabled", True),
            weight=m.get("weight", 0.25),
        )
        for m in trading_raw.get("markets", [{"symbol": "BTC", "weight": 1.0}])
    ]

    # ── Leverage ──────────────────────────────────────────────
    lev_raw = trading_raw.get("leverage", {})
    leverage = LeverageConfig(
        min=int(lev_raw.get("min", 10)),
        max=int(lev_raw.get("max", 35)),
        default=int(lev_raw.get("default", 15)),
        high_confidence_threshold=float(lev_raw.get("high_confidence_threshold", 0.80)),
        low_confidence_threshold=float(lev_raw.get("low_confidence_threshold", 0.55)),
        step=int(lev_raw.get("step", 5)),
    )

    # ── Risk ──────────────────────────────────────────────────
    risk_raw = trading_raw.get("risk", {})
    risk = RiskConfig(
        max_drawdown_pct=float(risk_raw.get("max_drawdown_pct", 0.20)),
        daily_loss_limit_pct=float(risk_raw.get("daily_loss_limit_pct", 0.05)),
        max_open_positions=int(risk_raw.get("max_open_positions", 4)),
        stop_loss_atr_multiplier=float(risk_raw.get("stop_loss_atr_multiplier", 2.0)),
        take_profit_atr_multiplier=float(risk_raw.get("take_profit_atr_multiplier", 3.0)),
        trailing_stop_pct=float(risk_raw.get("trailing_stop_pct", 0.015)),
    )

    # ── Position sizing ───────────────────────────────────────
    ps_raw = trading_raw.get("position_sizing", {})
    position_sizing = PositionSizingConfig(
        method=ps_raw.get("method", "kelly_fractional"),
        kelly_fraction=float(ps_raw.get("kelly_fraction", 0.25)),
        max_position_pct=float(ps_raw.get("max_position_pct", 0.30)),
        min_position_usd=float(ps_raw.get("min_position_usd", 100.0)),
    )

    # ── Trading ───────────────────────────────────────────────
    trading_mode = os.environ.get("TRADING_MODE", trading_raw.get("mode", "paper"))
    trading = TradingConfig(
        mode=trading_mode,
        initial_equity=float(
            os.environ.get("INITIAL_EQUITY", trading_raw.get("initial_equity", 10_000.0))
        ),
        base_currency=trading_raw.get("base_currency", "USDC"),
        markets=markets,
        leverage=leverage,
        risk=risk,
        position_sizing=position_sizing,
    )

    # ── Data ──────────────────────────────────────────────────
    intervals = data_raw.get("intervals", {})
    lookback = data_raw.get("lookback", {})
    data = DataConfig(
        hyperliquid_api_url=os.environ.get(
            "HYPERLIQUID_API_URL",
            data_raw.get("hyperliquid_api_url", "https://api.hyperliquid.xyz/info"),
        ),
        hyperliquid_ws_url=data_raw.get("hyperliquid_ws_url", "wss://api.hyperliquid.xyz/ws"),
        primary_interval=intervals.get("primary", "15m"),
        secondary_interval=intervals.get("secondary", "1h"),
        macro_interval=intervals.get("macro", "4h"),
        lookback_candles=int(lookback.get("candles", 500)),
    )

    # ── ML ────────────────────────────────────────────────────
    signals = ml_raw.get("signals", {})
    training = ml_raw.get("training", {})
    ml = MLConfig(
        long_threshold=float(signals.get("long_threshold", 0.60)),
        short_threshold=float(signals.get("short_threshold", 0.60)),
        close_threshold=float(signals.get("close_threshold", 0.45)),
        min_ensemble_agreement=float(signals.get("min_ensemble_agreement", 0.60)),
        model_save_dir=training.get("model_save_dir", "models"),
        retrain_interval_hours=int(training.get("retrain_interval_hours", 24)),
    )

    # ── Gemini ────────────────────────────────────────────────
    gemini = GeminiConfig(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        model=gemini_raw.get("model", "gemini-1.5-pro"),
        temperature=float(gemini_raw.get("temperature", 0.1)),
        max_output_tokens=int(gemini_raw.get("max_output_tokens", 2048)),
    )

    # ── Paper broker ──────────────────────────────────────────
    paper_broker = PaperBrokerConfig(
        initial_equity=float(
            os.environ.get("INITIAL_EQUITY", paper_raw.get("initial_equity", 10_000.0))
        ),
        taker_fee=float(paper_raw.get("taker_fee", 0.00045)),
        maker_fee=float(paper_raw.get("maker_fee", 0.00020)),
        funding_enabled=bool(paper_raw.get("funding_enabled", True)),
        liquidation_enabled=bool(paper_raw.get("liquidation_enabled", True)),
        slippage_bps=int(paper_raw.get("slippage_bps", 5)),
    )

    # ── Evaluation ────────────────────────────────────────────
    thresholds = eval_raw.get("thresholds", {})
    auto_adj = eval_raw.get("auto_adjust", {})
    evaluation = EvaluationConfig(
        min_sharpe=float(thresholds.get("min_sharpe", 1.0)),
        min_win_rate=float(thresholds.get("min_win_rate", 0.45)),
        max_drawdown_pct=float(thresholds.get("max_drawdown_pct", 0.25)),
        min_profit_factor=float(thresholds.get("min_profit_factor", 1.20)),
        auto_adjust_enabled=bool(auto_adj.get("enabled", True)),
        evaluation_window_trades=int(auto_adj.get("evaluation_window_trades", 50)),
    )

    # ── System ────────────────────────────────────────────────
    system = SystemConfig(
        name=system_raw.get("name", "Quantum Trader"),
        version=system_raw.get("version", "1.0.0"),
        log_level=os.environ.get("LOG_LEVEL", system_raw.get("log_level", "INFO")),
        state_dir=system_raw.get("state_dir", ".trading_state"),
        database_file=system_raw.get("database_file", "workflow_state.db"),
        results_dir=system_raw.get("results_dir", "results"),
    )

    return AppConfig(
        system=system,
        trading=trading,
        data=data,
        ml=ml,
        gemini=gemini,
        paper_broker=paper_broker,
        evaluation=evaluation,
    )
