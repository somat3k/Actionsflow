"""
Quantum Trading System – Configuration Management
Loads and validates trading configuration with environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    market_type: str = "crypto"      # "crypto" | "index"
    training_only: bool = False       # True = used for training only, not live trading
    yf_ticker: str = ""              # Yahoo Finance ticker (for index markets)


@dataclass
class TradingConfig:
    mode: str = "paper"
    initial_equity: float = 10_000.0
    base_currency: str = "USDC"
    markets: List[MarketConfig] = field(default_factory=list)
    index_markets: List[MarketConfig] = field(default_factory=list)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)


@dataclass
class DataConfig:
    hyperliquid_api_url: str = "https://api.hyperliquid.xyz/info"
    hyperliquid_ws_url: str = "wss://api.hyperliquid.xyz/ws"
    primary_interval: str = "1m"
    secondary_interval: str = "5m"
    macro_interval: str = "15m"
    hourly_interval: str = "1h"
    daily_interval: str = "1d"
    lookback_candles: int = 500
    training_lookback_candles: int = 5000
    dataset_dir: str = "datasets"
    dataset_format: str = "safetensors"
    historical_csv_dir: str = "datasets/csv"
    historical_csv_max_years: int = 10
    rate_limit_delay_s: float = 1.0


_DEFAULT_MODEL_WEIGHTS: Dict[str, float] = {
    "xgb": 0.25, "gb": 0.10, "rf": 0.15, "lstm": 0.20, "linear": 0.10, "tree_clf": 0.20,
}


@dataclass
class MLConfig:
    long_threshold: float = 0.60
    short_threshold: float = 0.60
    close_threshold: float = 0.45
    min_ensemble_agreement: float = 0.60
    model_save_dir: str = "models"
    retrain_interval_hours: int = 24
    training_epochs: int = 200
    reinforcement_alpha: float = 0.1
    model_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MODEL_WEIGHTS)
    )
    # ExtraTrees (tree-classifier-decision-making-system) hyperparameters
    extra_trees_n_estimators: int = 200
    extra_trees_max_depth: int = 10
    # Infinity-loop supervised learning
    infinity_loop_enabled: bool = True
    infinity_loop_max_epochs: int = 0         # 0 = infinite
    infinity_zero_trade_threshold: int = 0
    infinity_hp_adjust_step_threshold: float = 0.02
    infinity_hp_adjust_agreement_step: float = 0.05
    infinity_evaluation_interval: int = 10


@dataclass
class GeminiConfig:
    api_key: str = ""
    api_key_2: str = ""
    model: str = "gemini-2.5-pro"
    model_2: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_output_tokens: int = 2048


@dataclass
class GroqConfig:
    api_key: str = ""
    model: str = "llama3-70b-8192"
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    timeout_seconds: int = 30


@dataclass
class OpenAIConfig:
    api_key: str = ""
    model: str = "gpt-4o-mini"
    api_url: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    timeout_seconds: int = 30


@dataclass
class OpenRouterConfig:
    api_key: str = ""
    model: str = "openai/gpt-4o-mini"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0.1
    max_output_tokens: int = 2048
    timeout_seconds: int = 30


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
    # Trade volume targets (per day); defaults are intentionally conservative to
    # avoid limiting strategy throughput. WARNING: typical strategies should tune
    # this much lower (e.g., 10-50 trades/day). Set <=0 to disable adjustments.
    min_trades_per_day: int = 50
    # Stabs/pierces: short-window early-warning checks
    stabs_enabled: bool = True
    stabs_window_trades: int = 10
    stabs_min_win_rate: float = 0.35
    stabs_max_drawdown_pct: float = 0.12
    stabs_pierce_sharpe_threshold: float = 0.5


@dataclass
class SystemConfig:
    name: str = "Quantum Trader"
    version: str = "1.0.0"
    log_level: str = "INFO"
    state_dir: str = ".trading_state"
    database_file: str = "workflow_state.db"
    results_dir: str = "results"


@dataclass
class CacheConfig:
    """Redis cache configuration.

    When ``redis_url`` is empty the embedded ``fakeredis`` backend is used
    automatically – no external Redis server is required.
    Set ``REDIS_URL`` in the environment to connect to an external instance.
    """

    enabled: bool = True
    redis_url: str = ""          # Empty = use embedded fakeredis
    default_ttl_seconds: int = 3600   # 1 hour default TTL; 0 = no expiry
    namespace: str = "qt"


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
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
    cache_raw = raw.get("cache", {})
    trading_raw = raw.get("trading", {})
    data_raw = raw.get("data", {})
    ml_raw = raw.get("ml", {})
    gemini_raw = raw.get("gemini", {})
    groq_raw = raw.get("groq", {})
    openai_raw = raw.get("openai", {})
    openrouter_raw = raw.get("openrouter", {})
    paper_raw = raw.get("paper_broker", {})
    eval_raw = raw.get("evaluation", {})

    # ── Markets ───────────────────────────────────────────────
    markets = [
        MarketConfig(
            symbol=m["symbol"],
            enabled=m.get("enabled", True),
            weight=m.get("weight", 0.25),
            market_type=m.get("market_type", "crypto"),
            training_only=m.get("training_only", False),
            yf_ticker=m.get("yf_ticker", ""),
        )
        for m in trading_raw.get("markets", [{"symbol": "BTC", "weight": 1.0}])
    ]

    # ── Index markets (training-only) ─────────────────────────
    index_markets = [
        MarketConfig(
            symbol=m["symbol"],
            enabled=m.get("enabled", True),
            weight=m.get("weight", 0.10),
            market_type=m.get("market_type", "index"),
            training_only=m.get("training_only", True),
            yf_ticker=m.get("yf_ticker", m["symbol"]),
        )
        for m in trading_raw.get("index_markets", [])
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
        index_markets=index_markets,
        leverage=leverage,
        risk=risk,
        position_sizing=position_sizing,
    )

    # ── Data ──────────────────────────────────────────────────
    intervals = data_raw.get("intervals", {})
    lookback = data_raw.get("lookback", {})
    dataset_raw = data_raw.get("dataset", {})
    training_lookback_env = os.environ.get("LOOKBACK_CANDLES")
    if training_lookback_env:
        try:
            training_lookback = int(training_lookback_env)
        except ValueError as exc:
            raise ValueError("LOOKBACK_CANDLES must be an integer value") from exc
    else:
        training_lookback = int(lookback.get("training_candles", lookback.get("candles", 500)))
    data = DataConfig(
        hyperliquid_api_url=os.environ.get(
            "HYPERLIQUID_API_URL",
            data_raw.get("hyperliquid_api_url", "https://api.hyperliquid.xyz/info"),
        ),
        hyperliquid_ws_url=data_raw.get("hyperliquid_ws_url", "wss://api.hyperliquid.xyz/ws"),
        primary_interval=os.environ.get("PRIMARY_INTERVAL", intervals.get("primary", "1m")),
        secondary_interval=os.environ.get("SECONDARY_INTERVAL", intervals.get("secondary", "5m")),
        macro_interval=os.environ.get("MACRO_INTERVAL", intervals.get("macro", "15m")),
        hourly_interval=os.environ.get("HOURLY_INTERVAL", intervals.get("hourly", "1h")),
        daily_interval=os.environ.get("DAILY_INTERVAL", intervals.get("daily", "1d")),
        lookback_candles=int(lookback.get("candles", 500)),
        training_lookback_candles=training_lookback,
        dataset_dir=os.environ.get("DATASET_DIR", dataset_raw.get("dir", "datasets")),
        dataset_format=os.environ.get("DATASET_FORMAT", dataset_raw.get("format", "safetensors")),
        historical_csv_dir=os.environ.get(
            "HISTORICAL_CSV_DIR",
            data_raw.get("historical_csv", {}).get("dir", "datasets/csv"),
        ),
        historical_csv_max_years=int(
            data_raw.get("historical_csv", {}).get("max_years", 10)
        ),
        rate_limit_delay_s=float(
            data_raw.get("historical_csv", {}).get("rate_limit_delay_s", 1.0)
        ),
    )

    # ── ML ────────────────────────────────────────────────────
    signals = ml_raw.get("signals", {})
    training = ml_raw.get("training", {})
    # Load per-model weights from YAML if present.
    models_raw = ml_raw.get("models", {})
    default_weights = {
        "xgb": 0.25, "gb": 0.10, "rf": 0.15, "lstm": 0.20, "linear": 0.10, "tree_clf": 0.20,
    }
    yaml_name_map = {
        "xgboost": "xgb", "gradient_boost": "gb", "random_forest": "rf",
        "lstm": "lstm", "linear": "linear", "extra_trees": "tree_clf",
    }
    model_weights = dict(default_weights)
    for yaml_name, internal_name in yaml_name_map.items():
        model_cfg = models_raw.get(yaml_name, {})
        if "weight" in model_cfg:
            model_weights[internal_name] = float(model_cfg["weight"])
    infinity_raw = training.get("infinity_loop", ml_raw.get("infinity_loop", {}))
    ml = MLConfig(
        long_threshold=float(signals.get("long_threshold", 0.60)),
        short_threshold=float(signals.get("short_threshold", 0.60)),
        close_threshold=float(signals.get("close_threshold", 0.45)),
        min_ensemble_agreement=float(signals.get("min_ensemble_agreement", 0.60)),
        model_save_dir=training.get("model_save_dir", "models"),
        retrain_interval_hours=int(training.get("retrain_interval_hours", 24)),
        training_epochs=int(os.environ.get("TRAINING_EPOCHS", training.get("epochs", 200))),
        reinforcement_alpha=float(
            os.environ.get("REINFORCEMENT_ALPHA", training.get("reinforcement_alpha", 0.1))
        ),
        model_weights=model_weights,
        extra_trees_n_estimators=int(
            models_raw.get("extra_trees", {}).get("n_estimators", 200)
        ),
        extra_trees_max_depth=int(
            models_raw.get("extra_trees", {}).get("max_depth", 10)
        ),
        infinity_loop_enabled=bool(infinity_raw.get("enabled", True)),
        infinity_loop_max_epochs=int(infinity_raw.get("max_epochs", 0)),
        infinity_zero_trade_threshold=int(infinity_raw.get("zero_trade_threshold", 0)),
        infinity_hp_adjust_step_threshold=float(
            infinity_raw.get("hp_adjust_step_threshold", 0.02)
        ),
        infinity_hp_adjust_agreement_step=float(
            infinity_raw.get("hp_adjust_agreement_step", 0.05)
        ),
        infinity_evaluation_interval=int(infinity_raw.get("evaluation_interval_epochs", 10)),
    )

    # ── Gemini ────────────────────────────────────────────────
    gemini = GeminiConfig(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        api_key_2=os.environ.get("GEMINI_API_KEY2", ""),
        model=gemini_raw.get("model", "gemini-2.5-pro"),
        model_2=gemini_raw.get("model_2", "gemini-2.5-pro"),
        temperature=float(gemini_raw.get("temperature", 0.1)),
        max_output_tokens=int(gemini_raw.get("max_output_tokens", 2048)),
    )

    groq = GroqConfig(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        model=os.environ.get("GROQ_MODEL", groq_raw.get("model", "llama3-70b-8192")),
        api_url=os.environ.get(
            "GROQ_API_URL",
            groq_raw.get("api_url", "https://api.groq.com/openai/v1/chat/completions"),
        ),
        temperature=float(os.environ.get("GROQ_TEMPERATURE", groq_raw.get("temperature", 0.1))),
        max_output_tokens=int(
            os.environ.get("GROQ_MAX_TOKENS", groq_raw.get("max_output_tokens", 2048))
        ),
        timeout_seconds=int(
            os.environ.get("GROQ_TIMEOUT_SECONDS", groq_raw.get("timeout_seconds", 30))
        ),
    )

    openai = OpenAIConfig(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("OPENAI_MODEL", openai_raw.get("model", "gpt-4o-mini")),
        api_url=os.environ.get(
            "OPENAI_API_URL",
            openai_raw.get("api_url", "https://api.openai.com/v1/chat/completions"),
        ),
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", openai_raw.get("temperature", 0.1))),
        max_output_tokens=int(
            os.environ.get("OPENAI_MAX_TOKENS", openai_raw.get("max_output_tokens", 2048))
        ),
        timeout_seconds=int(
            os.environ.get("OPENAI_TIMEOUT_SECONDS", openai_raw.get("timeout_seconds", 30))
        ),
    )

    openrouter = OpenRouterConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model=os.environ.get("OPENROUTER_MODEL", openrouter_raw.get("model", "openai/gpt-4o-mini")),
        api_url=os.environ.get(
            "OPENROUTER_API_URL",
            openrouter_raw.get("api_url", "https://openrouter.ai/api/v1/chat/completions"),
        ),
        temperature=float(
            os.environ.get("OPENROUTER_TEMPERATURE", openrouter_raw.get("temperature", 0.1))
        ),
        max_output_tokens=int(
            os.environ.get("OPENROUTER_MAX_TOKENS", openrouter_raw.get("max_output_tokens", 2048))
        ),
        timeout_seconds=int(
            os.environ.get("OPENROUTER_TIMEOUT_SECONDS", openrouter_raw.get("timeout_seconds", 30))
        ),
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
    stabs_raw = eval_raw.get("stabs", {})
    trade_volume_raw = eval_raw.get("trade_volume", {})
    evaluation = EvaluationConfig(
        min_sharpe=float(thresholds.get("min_sharpe", 1.0)),
        min_win_rate=float(thresholds.get("min_win_rate", 0.45)),
        max_drawdown_pct=float(thresholds.get("max_drawdown_pct", 0.25)),
        min_profit_factor=float(thresholds.get("min_profit_factor", 1.20)),
        auto_adjust_enabled=bool(auto_adj.get("enabled", True)),
        evaluation_window_trades=int(auto_adj.get("evaluation_window_trades", 50)),
        min_trades_per_day=int(trade_volume_raw.get("min_trades_per_day", 50)),
        stabs_enabled=bool(stabs_raw.get("enabled", True)),
        stabs_window_trades=int(stabs_raw.get("window_trades", 10)),
        stabs_min_win_rate=float(stabs_raw.get("min_win_rate", 0.35)),
        stabs_max_drawdown_pct=float(stabs_raw.get("max_drawdown_pct", 0.12)),
        stabs_pierce_sharpe_threshold=float(stabs_raw.get("pierce_sharpe_threshold", 0.5)),
    )

    # ── Cache (Redis) ─────────────────────────────────────────
    _cache_ttl_raw = cache_raw.get("default_ttl_seconds", 3600)
    cache = CacheConfig(
        enabled=bool(cache_raw.get("enabled", True)),
        redis_url=os.environ.get("REDIS_URL", cache_raw.get("redis_url", "")),
        default_ttl_seconds=int(_cache_ttl_raw) if _cache_ttl_raw is not None else 3600,
        namespace=cache_raw.get("namespace", "qt"),
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
        cache=cache,
        trading=trading,
        data=data,
        ml=ml,
        gemini=gemini,
        groq=groq,
        openai=openai,
        openrouter=openrouter,
        paper_broker=paper_broker,
        evaluation=evaluation,
    )
