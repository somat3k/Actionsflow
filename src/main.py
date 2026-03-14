"""
Quantum Trading System – Main Entry Point
Orchestrates: data fetching → ML inference → Gemini validation →
risk sizing → paper/live execution → evaluation → adjustments.

Usage:
    python -m src.main --mode paper --run-type training
    python -m src.main --mode paper --run-type signal
    python -m src.main --mode live  --run-type signal
    python -m src.main --run-type evaluate
    python -m src.main --run-type train-models
    python -m src.main --run-type infinity-train
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, ascii=True, dynamic_ncols=False, **kwargs)

except ImportError:  # pragma: no cover
    def _progress(iterable, **kwargs):
        return iterable

from src.config import AppConfig, MarketConfig, load_config
from src.data_fetcher import HyperliquidDataFetcher
from src.database_manager import DatabaseManager
from src.evaluator import Evaluator, compute_metrics
from src.ai_orchestrator import MultiAIOrchestrator
from src.dataset_manager import DatasetManager
from src.live_trader import LiveTrader
from src.paper_broker import PaperBroker
from src.risk_manager import PositionRequest, RiskManager
from src.supervised_learning import SupervisedLearningModule
from src.utils import fmt_pct, fmt_usd, get_logger, parse_snapshot_end_ms, utc_now, utc_now_ms

log = get_logger(__name__)


def _build_db_manager(cfg) -> DatabaseManager:
    db_path = Path(cfg.system.state_dir) / cfg.system.database_file
    cache_cfg = getattr(cfg, "cache", None)
    cache_enabled: bool = True
    redis_url: Optional[str] = None
    cache_ttl: Optional[int] = 3600
    namespace: str = "qt"
    if cache_cfg is not None:
        cache_enabled = bool(cache_cfg.enabled)
        namespace = cache_cfg.namespace or "qt"
        redis_url = cache_cfg.redis_url or None
        cache_ttl = cache_cfg.default_ttl_seconds if cache_cfg.default_ttl_seconds > 0 else None
    return DatabaseManager(
        db_path,
        redis_url=redis_url,
        cache_ttl=cache_ttl,
        cache_enabled=cache_enabled,
        namespace=namespace,
    )


def _print_github_summary(text: str) -> None:
    """Write to GitHub Actions step summary if available."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write(text + "\n")


def _get_hyperliquid_private_key() -> Optional[str]:
    """Read Hyperliquid private key from supported environment variable names."""
    return os.environ.get("HYPERLIQUID_SECRET") or os.environ.get("HYPERLIQUID_PRIVATE_KEY")


def _ensure_data_snapshot_end_ms() -> int:
    """Ensure DATA_SNAPSHOT_END_MS is set and return the snapshot time."""
    raw = os.environ.get("DATA_SNAPSHOT_END_MS")
    parsed = parse_snapshot_end_ms(raw, logger=log)
    if parsed is not None:
        return parsed
    snapshot = utc_now_ms()
    os.environ["DATA_SNAPSHOT_END_MS"] = str(snapshot)
    return snapshot


def _is_live_trading_enabled() -> bool:
    """Return True when LIVE_TRADING_ENABLED explicitly enables live orders."""
    return os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"


def _resolve_trading_eligibility(db: DatabaseManager) -> tuple[bool, str]:
    """Return trading eligibility, honoring overrides before cached evaluation."""
    override = os.environ.get("TRADING_ELIGIBILITY_OVERRIDE", "").lower() == "true"
    if override:
        return True, "TRADING_ELIGIBILITY_OVERRIDE enabled"
    cached = db.get_cache("evaluation:last_metrics")
    if isinstance(cached, dict):
        if cached.get("pause_trading"):
            return False, cached.get("pause_reason") or "Pause recommended by evaluation"
        passed = cached.get("pass")
        if passed is True:
            return True, "Evaluation thresholds passed"
        if passed is False:
            return False, "Evaluation thresholds failed"
    return False, "No evaluation metrics recorded"


def _parse_cached_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp, assuming naive values are UTC."""
    if not value:
        return None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _get_last_training_time(db: DatabaseManager, symbol: str) -> Optional[datetime]:
    cache_value = db.get_cache("training:last_run")
    if isinstance(cache_value, dict):
        return _parse_cached_timestamp(cache_value.get(symbol))
    # Legacy cache stored a single timestamp for all symbols.
    return _parse_cached_timestamp(cache_value)


def _should_retrain(cfg: AppConfig, db: DatabaseManager, symbol: str) -> bool:
    """Return True when retraining is due/missing; interval <= 0 disables retraining."""
    interval_hours = cfg.ml.retrain_interval_hours
    if interval_hours <= 0:
        return False
    last_training = _get_last_training_time(db, symbol)
    if last_training is None:
        return True
    return utc_now() - last_training >= timedelta(hours=interval_hours)


def _record_training_time(db: DatabaseManager, symbols: List[str]) -> None:
    cache_value = db.get_cache("training:last_run")
    last_runs: Dict[str, str] = cache_value if isinstance(cache_value, dict) else {}
    timestamp = utc_now().isoformat()
    for symbol in symbols:
        last_runs[symbol] = timestamp
    db.set_cache("training:last_run", last_runs)


def _resolve_training_markets(cfg: AppConfig) -> List[MarketConfig]:
    """Filter training markets based on TRAINING_SYMBOLS env var if provided."""
    raw = os.environ.get("TRAINING_SYMBOLS")
    if not raw:
        return cfg.trading.markets
    allowed = {sym.strip().upper() for sym in raw.split(",") if sym.strip()}
    if not allowed:
        return cfg.trading.markets
    return [market for market in cfg.trading.markets if market.symbol.upper() in allowed]


def _resolve_training_program() -> str:
    """Resolve which training program to use (multi_timeframe|progressive|single)."""
    raw = os.environ.get("TRAINING_PROGRAM", "").strip().lower()
    if raw in {"mtf", "multi", "multi_timeframe", "multi-timeframe"}:
        return "multi_timeframe"
    if raw in {"progressive", "progression", "single_progression", "single-progression"}:
        return "progressive"
    if raw in {"single", "primary", "single_timeframe", "single-timeframe"}:
        return "single"
    return "multi_timeframe"


def _parse_bool_env(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    value = raw.strip().lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}
    if value in truthy:
        return True
    if value in falsy:
        return False
    log.warning("Invalid boolean value for %s=%s; using default=%s", key, raw, default)
    return default


def _build_multiplex_signal(
    cfg: AppConfig,
    ensemble: Any,
    delegation_agent: Any,
    snapshot: Dict[str, Any],
    prior_regime: str = "unknown",
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a multi-timeframe ML signal from snapshot candles.

    For each configured trading timeframe (1m, 5m, 15m, 1h) runs ensemble
    inference on the available candle data.  Uses the per-timeframe model when
    available (epoch / MTF training), otherwise uses the global ``predict()``.
    The predictions from all available timeframes are combined via
    ``combined_decision()``, with higher timeframes carrying more weight.

    If fewer than two timeframes are available the function falls back to
    ``delegation_agent.predict()`` using the primary timeframe data.
    The ensemble-agreement gate (``cfg.ml.min_ensemble_agreement``) is applied
    to the combined result so that weak multi-timeframe consensus produces a
    flat signal instead of a noisy long/short.

    Args:
        cfg:               Application configuration.
        ensemble:          Trained ``QuantumEnsemble`` instance.
        delegation_agent:  ``ModelDelegationAgent`` fallback.
        snapshot:          Market data snapshot dict from ``fetch_all_market_data``.
        prior_regime:      Cached regime label from the previous cycle.
        symbol:            Optional symbol for NN-priority overrides.

    Returns:
        Signal dict with at minimum ``signal``, ``confidence``, ``agreement``,
        and ``delegated_to`` keys.
    """
    trading_tfs = [
        cfg.data.primary_interval,
        cfg.data.secondary_interval,
        cfg.data.macro_interval,
        cfg.data.hourly_interval,
    ]
    candles = snapshot.get("candles", {})
    tf_predictions: Dict[str, Any] = {}
    symbol_upper = symbol.upper() if symbol else ""
    nn_priority_symbols = {s.upper() for s in cfg.ml.nn_priority_symbols}
    nn_priority_signal: Optional[Dict[str, Any]] = None

    def _apply_nn_priority() -> Optional[Dict[str, Any]]:
        if nn_priority_signal and nn_priority_signal.get("nn_decision"):
            nn_payload = dict(nn_priority_signal)
            nn_payload["delegated_to"] = "nn_override"
            return nn_payload
        return None
    for tf in trading_tfs:
        df_tf = candles.get(tf)
        if df_tf is None or df_tf.empty:
            continue
        try:
            # Prefer per-timeframe model when available (MTF epoch training).
            has_tf_model = ensemble.has_timeframe_model(tf)
            if has_tf_model:
                pred = ensemble.predict_timeframe(df_tf, tf)
                pred.setdefault("timeframe", tf)
            else:
                pred = ensemble.predict(df_tf)
                pred["timeframe"] = tf
            tf_predictions[tf] = pred
            should_capture_nn_priority_signal = (
                symbol_upper in nn_priority_symbols
                and tf == cfg.data.primary_interval
                and not has_tf_model
            )
            if should_capture_nn_priority_signal:
                nn_priority_signal = pred
        except Exception as exc:
            log.debug("Multiplex prediction skipped for %s: %s", tf, exc)

    if symbol_upper in nn_priority_symbols and nn_priority_signal is None:
        nn_primary_df = candles.get(cfg.data.primary_interval)
        if nn_primary_df is not None and not nn_primary_df.empty:
            try:
                nn_priority_signal = ensemble.predict(nn_primary_df)
            except Exception as exc:
                log.debug("NN priority prediction skipped for %s: %s", symbol, exc)

    if len(tf_predictions) >= 2:
        ml_signal = ensemble.combined_decision(tf_predictions)
        # Apply ensemble-agreement gate (same semantics as single-TF predict).
        if ml_signal.get("agreement", 1.0) < cfg.ml.min_ensemble_agreement:
            ml_signal["signal"] = 0
            ml_signal["confidence"] = ml_signal.get("flat_prob", 1.0)
        nn_override = _apply_nn_priority()
        if nn_override:
            return nn_override
        ml_signal["delegated_to"] = "multiplex"
        return ml_signal

    # Fewer than two timeframes – fall back to delegation agent.
    nn_override = _apply_nn_priority()
    if nn_override:
        return nn_override
    primary_df = candles.get(cfg.data.primary_interval)
    if primary_df is not None and not primary_df.empty:
        return delegation_agent.predict(primary_df, regime=prior_regime)
    for df_tf in candles.values():
        if not df_tf.empty:
            return delegation_agent.predict(df_tf, regime=prior_regime)
    return {"signal": 0, "confidence": 0.0, "agreement": 0.0, "delegated_to": "none"}


def _update_model_weights_from_evaluation(
    cfg: AppConfig,
    db: DatabaseManager,
    metrics: Any,
) -> None:
    """Apply a reinforcement-learning weight update to all saved models.

    Loads per-symbol training accuracy scores from the database cache and
    applies ``QuantumEnsemble.apply_reinforcement`` with a small step size.
    The reward for each model is its validation accuracy.  When the overall
    performance is poor (win rate below threshold), the step size is doubled
    so the ensemble adapts more aggressively.

    This creates a closed feedback loop:
      train → signal → evaluate → adjust weights → train …

    The ``QuantumEnsemble`` import is deferred so that the evaluate path does
    not trigger ML-model loading unless training scores are actually cached.

    Args:
        cfg:     Application configuration.
        db:      Database manager for cache access.
        metrics: Latest ``PerformanceMetrics`` from ``Evaluator.evaluate``.
    """
    training_scores = db.get_cache("training:last_scores")
    if not isinstance(training_scores, dict) or not training_scores:
        log.debug("No cached training scores – skipping evaluation-driven weight update")
        return

    # Deferred import: keeps the evaluate path free of ML-model imports when
    # there are no cached training scores to work with.
    from src.ml_models import QuantumEnsemble  # noqa: PLC0415

    # Use a larger alpha when performance is weak so the ensemble adapts faster.
    base_alpha = cfg.ml.reinforcement_alpha
    win_rate = getattr(metrics, "win_rate", 0.0)
    alpha = base_alpha * 2.0 if win_rate < cfg.evaluation.min_win_rate else base_alpha

    ensemble = QuantumEnsemble(cfg)
    updated_symbols: List[str] = []
    for symbol, scores in training_scores.items():
        if not isinstance(scores, dict) or not scores:
            continue
        if not ensemble.load(symbol):
            log.debug("Cannot load model for %s – skipping weight update", symbol)
            continue
        updated_weights = ensemble.apply_reinforcement(scores, alpha)
        try:
            ensemble.save(symbol)
            updated_symbols.append(symbol)
            log.info(
                "Evaluation-driven weight update for %s (alpha=%.3f): %s",
                symbol, alpha, updated_weights,
            )
        except Exception as exc:
            log.warning("Failed to save updated weights for %s: %s", symbol, exc)

    if updated_symbols:
        db.set_cache(
            "evaluation:weight_update",
            {
                "symbols": updated_symbols,
                "alpha": alpha,
                "win_rate": win_rate,
            },
        )


def _ensure_model_ready(
    cfg: AppConfig,
    db: DatabaseManager,
    fetcher: HyperliquidDataFetcher,
    ensemble: Any,
    symbol: str,
) -> bool:
    """Return True when a model is ready; False if load/training cannot provide one.

    When training is needed (first run or scheduled retrain), all configured
    timeframes are fetched and multi-timeframe training is used when
    ``training_epochs`` > 1.  This matches the richer training used by
    ``run_training()`` so that models produced during signal cycles are of
    the same quality as those from dedicated training runs.

    Records training timestamps when retraining succeeds.
    """
    from src.dataset_manager import DatasetManager  # deferred to avoid circular issues

    loaded = ensemble.load(symbol)
    is_scheduled_retrain_due = _should_retrain(cfg, db, symbol)
    needs_initial_train = not loaded
    should_train = is_scheduled_retrain_due or needs_initial_train
    if should_train:
        training_program = _resolve_training_program()
        action = "Training" if needs_initial_train else "Retraining"
        log.info("%s model for %s (%s) …", action, symbol, training_program)

        # Fetch data for all configured timeframes so the model benefits from
        # the same multi-timeframe context used in dedicated training runs.
        dataset_mgr = DatasetManager(cfg, db)
        all_timeframes = [
            cfg.data.primary_interval,
            cfg.data.secondary_interval,
            cfg.data.macro_interval,
            cfg.data.hourly_interval,
            cfg.data.daily_interval,
        ]
        tf_dataframes: Dict[str, Any] = {}
        for tf in all_timeframes:
            try:
                df_tf = dataset_mgr.get_or_fetch_dataset(
                    fetcher, symbol, tf,
                    lookback_candles=cfg.data.training_lookback_candles,
                    force_refresh=should_train,  # always refresh on any train/retrain
                )
                if not df_tf.empty:
                    tf_dataframes[tf] = df_tf
            except Exception as exc:
                log.debug("Skipping %s@%s during ensure_model_ready: %s", symbol, tf, exc)

        if not tf_dataframes:
            log.warning("No data for any timeframe of %s – skipping retraining", symbol)
            return loaded

        primary_df = tf_dataframes.get(cfg.data.primary_interval)
        if primary_df is None:
            primary_df = next(iter(tf_dataframes.values()))

        try:
            training_epochs = max(1, cfg.ml.training_epochs)
            reinforcement_alpha = cfg.ml.reinforcement_alpha
            if (
                training_program == "multi_timeframe"
                and training_epochs > 1
                and len(tf_dataframes) > 1
            ):
                ensemble.train_multi_timeframe_with_progression(
                    tf_dataframes,
                    symbol=symbol,
                    epochs=training_epochs,
                    reinforcement_alpha=reinforcement_alpha,
                    primary_tf=cfg.data.primary_interval,
                )
            elif training_program == "progressive" and training_epochs > 1:
                ensemble.train_with_progression(
                    primary_df,
                    symbol=symbol,
                    epochs=training_epochs,
                    reinforcement_alpha=reinforcement_alpha,
                )
            else:
                ensemble.train(primary_df, symbol=symbol)
        except Exception as exc:
            log.warning(
                "%s failed for %s (%s): %s",
                action,
                symbol,
                type(exc).__name__,
                exc,
            )
            return loaded
        _record_training_time(db, [symbol])
        loaded = True
    return loaded


def run_training(config_path: Optional[Path] = None) -> int:
    """Train ML models on historical data (crypto + indices).

    Implements the 200-epoch harmonogram:
    - Fetches OHLCV data for all five timeframes (1m, 5m, 15m, 1h, 1d).
    - Also fetches index/equity data (GOOGL, AAPL, NVDA, US30, SPX, JPM,
      SPY, NASDAQ) via Yahoo Finance for cross-market training enrichment.
    - Runs ``training_epochs`` (default 200) progressive epochs per symbol.
    - Applies reinforcement-learning weight updates after each epoch.
    - Displays tqdm progress bars for the training session, each symbol,
      and each epoch's timeframe loop.
    """
    from src.ml_models import QuantumEnsemble
    from src.index_data_fetcher import IndexDataFetcher

    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)

    training_epochs = max(1, cfg.ml.training_epochs)
    reinforcement_alpha = cfg.ml.reinforcement_alpha
    training_program = _resolve_training_program()
    force_retrain = _parse_bool_env("FORCE_RETRAIN")

    enabled_markets = [
        m for m in _resolve_training_markets(cfg) if m.enabled
    ]

    # ── Pre-fetch index / equity training data ────────────────────────────
    index_fetcher = IndexDataFetcher(cfg)
    index_dfs: Dict[str, Any] = {}
    enabled_index_markets = [
        m for m in cfg.trading.index_markets if m.enabled
    ]
    if enabled_index_markets:
        log.info(
            "Fetching index/equity training data for %d symbols …",
            len(enabled_index_markets),
        )
        for idx_market in enabled_index_markets:
            try:
                df_idx = index_fetcher.fetch_ohlcv_history(
                    idx_market.symbol,
                    interval=cfg.data.daily_interval,
                    lookback_candles=cfg.data.training_lookback_candles,
                    yf_ticker=idx_market.yf_ticker or None,
                )
                if not df_idx.empty:
                    index_dfs[idx_market.symbol] = df_idx
                    log.info(
                        "Index data ready: %s (%d rows)", idx_market.symbol, len(df_idx)
                    )
            except Exception as exc:
                log.warning("Failed to fetch index data for %s: %s", idx_market.symbol, exc)

    log.info(
        "=== TRAINING SESSION | %d epochs | %d crypto + %d index symbols ===",
        training_epochs, len(enabled_markets), len(index_dfs),
    )
    log.info("Training program: %s", training_program)

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    dataset_mgr = DatasetManager(cfg, db)

    # All configured timeframes for multi-timeframe training.
    all_timeframes = [
        cfg.data.primary_interval,
        cfg.data.secondary_interval,
        cfg.data.macro_interval,
        cfg.data.hourly_interval,
        cfg.data.daily_interval,
    ]

    # ── Phase 1: Pre-fetch all symbol data before training begins ─────────
    # Fetch data for every symbol upfront so that training runs without
    # waiting on network I/O between symbols (sequential, low-latency).
    log.info(
        "── Phase 1: Pre-fetching data for %d crypto symbols × %d timeframes ──",
        len(enabled_markets), len(all_timeframes),
    )
    all_tf_dataframes: Dict[str, Dict[str, Any]] = {}
    prefetch_bar = _progress(
        enabled_markets,
        desc="Pre-fetch",
        total=len(enabled_markets),
        unit="symbol",
        leave=False,
    )
    for market in prefetch_bar:
        symbol = market.symbol
        if hasattr(prefetch_bar, "set_postfix"):
            prefetch_bar.set_postfix(symbol=symbol)
        tf_dfs: Dict[str, Any] = {}
        for tf in all_timeframes:
            log.debug("  Pre-fetching %s@%s …", symbol, tf)
            df_tf = dataset_mgr.get_or_fetch_dataset(
                fetcher,
                symbol,
                tf,
                lookback_candles=cfg.data.training_lookback_candles,
                force_refresh=force_retrain,
            )
            if not df_tf.empty:
                tf_dfs[tf] = df_tf
        if tf_dfs:
            all_tf_dataframes[symbol] = tf_dfs
        else:
            log.warning("No data pre-fetched for %s – will skip training", symbol)

    log.info(
        "── Phase 2: Training %d symbols sequentially (one after another) ──",
        len(all_tf_dataframes),
    )

    results: Dict[str, Any] = {}
    epoch_scores: Dict[str, Any] = {}

    # ── Train index models first (cross-market enrichment) ────────────────
    if index_dfs:
        for idx_sym, idx_df in index_dfs.items():
            log.info("── Index symbol: %s ──", idx_sym)
            try:
                scores = ensemble.train(idx_df, symbol=idx_sym)
                results[idx_sym] = scores
            except Exception as exc:
                log.warning("Index training failed for %s: %s", idx_sym, exc)

    # ── Phase 2: Train each symbol sequentially using pre-fetched data ────
    # Symbols are trained one after another (no parallel fits) so each
    # training run completes fully before the next symbol starts.  This
    # avoids data-access contention and produces stable model artefacts that
    # capture the most recent market intervals.
    ready_markets = [m for m in enabled_markets if m.symbol in all_tf_dataframes]
    session_bar = _progress(
        ready_markets,
        desc="Training session",
        total=len(ready_markets),
        unit="symbol",
        leave=True,
    )
    total_ready = len(ready_markets)
    for seq_idx, market in enumerate(session_bar, start=1):
        symbol = market.symbol
        if hasattr(session_bar, "set_postfix"):
            session_bar.set_postfix(symbol=symbol)
        log.info("── Training symbol %s (%d/%d) ──", symbol, seq_idx, total_ready)

        # Use the already-fetched dataframes (no additional network I/O).
        tf_dataframes = all_tf_dataframes[symbol]

        primary_df = tf_dataframes.get(cfg.data.primary_interval)
        if primary_df is None or primary_df.empty:
            primary_df = next(iter(tf_dataframes.values()))

        # ── Train ─────────────────────────────────────────────────────────
        if (
            training_program == "multi_timeframe"
            and training_epochs > 1
            and len(tf_dataframes) > 1
        ):
            mtf_results = ensemble.train_multi_timeframe_with_progression(
                tf_dataframes,
                symbol=symbol,
                epochs=training_epochs,
                reinforcement_alpha=reinforcement_alpha,
                primary_tf=cfg.data.primary_interval,
            )
            if not mtf_results:
                continue
            epoch_scores[symbol] = mtf_results
            results[symbol] = mtf_results[-1]["combined_scores"]
        elif training_program == "progressive" and training_epochs > 1:
            prog_results = ensemble.train_with_progression(
                primary_df,
                symbol=symbol,
                epochs=training_epochs,
                reinforcement_alpha=reinforcement_alpha,
            )
            if not prog_results:
                continue
            epoch_scores[symbol] = prog_results
            results[symbol] = prog_results[-1]["scores"]
        else:
            scores = ensemble.train(primary_df, symbol=symbol)
            results[symbol] = scores

    # ── Persist results ───────────────────────────────────────────────────
    output_path = Path(cfg.system.results_dir) / "training_scores.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)

    summary = (
        "## 🤖 Model Training Results\n\n"
        f"**Epochs:** {training_epochs} | "
        f"**Program:** {training_program} | "
        f"**Timeframes:** {', '.join(all_timeframes)}\n\n"
        "| Symbol | NN (primary) | XGB | GB | RF | Linear |\n"
        "|---|---|---|---|---|---|\n"
    )
    for sym, scores in results.items():
        def _fmt(key: str) -> str:
            v = scores.get(key)
            return f"{v:.4f}" if isinstance(v, float) else "N/A"
        row = (
            f"| {sym} | **{_fmt('lstm')}** | {_fmt('xgb')} | {_fmt('gb')} "
            f"| {_fmt('rf')} | {_fmt('linear')} |\n"
        )
        summary += row
    _print_github_summary(summary)
    db.set_cache("training:last_scores", results)
    if epoch_scores:
        db.set_cache("training:epoch_scores", epoch_scores)
    if index_dfs:
        db.set_cache("training:index_symbols", list(index_dfs.keys()))
    if results:
        _record_training_time(db, list(results.keys()))
    db.record_task_completion(
        task_name="model_training",
        run_type="train-models",
        mode=cfg.trading.mode,
        status="success",
        metadata={
            "symbols_trained": list(results.keys()),
            "index_symbols": list(index_dfs.keys()),
        },
    )
    log.info("Training complete. Results: %s", results)
    return 0


def run_infinity_training(config_path: Optional[Path] = None) -> int:
    """Infinity-loop supervised-learning training mode.

    The AI leader (Gemini / OpenAI / OpenRouter / Groq) acts as a supervised-learning
    student in an endless loop of:
      1. Train models for ``training_epochs`` epochs.
      2. Evaluate performance metrics.
      3. Detect zero-trade conditions → relax hyperparameters.
      4. Consult AI leader for additional hyperparameter adjustments.
      5. Repeat from step 1.

    The loop runs until:
    * ``INFINITY_MAX_EPOCHS`` env var is set to a positive integer and the
      epoch counter reaches that limit.
    * ``ml.infinity_loop.max_epochs`` config value > 0 and limit reached.
    * The process is externally terminated (SIGTERM / Ctrl-C).

    This function is intended to be used by a GitHub Actions workflow or a
    long-running runner process.  It writes status to the database and
    GitHub Actions step summary after every evaluation checkpoint.
    """
    from src.ml_models import QuantumEnsemble
    from src.index_data_fetcher import IndexDataFetcher
    from dataclasses import asdict

    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)

    # Allow override from environment.
    env_max = os.environ.get("INFINITY_MAX_EPOCHS")
    if env_max:
        try:
            cfg.ml.infinity_loop_max_epochs = int(env_max)
        except ValueError:
            pass

    eval_interval_env = os.environ.get("INFINITY_EVALUATION_INTERVAL")
    if eval_interval_env:
        try:
            cfg.ml.infinity_evaluation_interval = int(eval_interval_env)
        except ValueError:
            log.warning("Invalid INFINITY_EVALUATION_INTERVAL=%s; using config value", eval_interval_env)

    exit_on_pass = _parse_bool_env("INFINITY_EXIT_ON_PASS", default=True)

    max_epochs = cfg.ml.infinity_loop_max_epochs  # 0 = infinite
    training_epochs = max(1, cfg.ml.training_epochs)
    reinforcement_alpha = cfg.ml.reinforcement_alpha
    training_program = _resolve_training_program()
    force_retrain = _parse_bool_env("FORCE_RETRAIN")
    should_refresh_dataset = force_retrain or cfg.ml.infinity_force_refresh
    payload_probe = _parse_bool_env("INFINITY_PAYLOAD_PROBE")
    infinity_symbols_env = os.environ.get("INFINITY_TRAINING_SYMBOLS")
    if infinity_symbols_env and infinity_symbols_env.strip():
        allowed = {sym.upper() for sym in cfg.ml.infinity_training_symbols}
        if allowed:
            enabled_markets = [
                m for m in cfg.trading.markets if m.enabled and m.symbol.upper() in allowed
            ]
        else:
            log.warning("INFINITY_TRAINING_SYMBOLS set but empty after parsing; using defaults")
            enabled_markets = [m for m in _resolve_training_markets(cfg) if m.enabled]
    elif os.environ.get("TRAINING_SYMBOLS"):
        enabled_markets = [m for m in _resolve_training_markets(cfg) if m.enabled]
    elif cfg.ml.infinity_training_symbols:
        allowed = {sym.upper() for sym in cfg.ml.infinity_training_symbols}
        enabled_markets = [
            m for m in cfg.trading.markets if m.enabled and m.symbol.upper() in allowed
        ]
    else:
        enabled_markets = [m for m in _resolve_training_markets(cfg) if m.enabled]
    index_fetcher = IndexDataFetcher(cfg)
    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    dataset_mgr = DatasetManager(cfg, db)
    ai_orchestrator = MultiAIOrchestrator(cfg)
    supervised = SupervisedLearningModule(cfg)
    evaluator = Evaluator(cfg)

    # Restore supervised learning state if available.
    state_dir = Path(cfg.system.state_dir)
    sl_state_path = state_dir / "supervised_learning_state.json"
    supervised.load_state(sl_state_path)

    broker = PaperBroker(cfg)
    broker.load(state_dir / "paper_broker.json")

    all_timeframes = [
        cfg.data.primary_interval,
        cfg.data.secondary_interval,
        cfg.data.macro_interval,
        cfg.data.hourly_interval,
        cfg.data.daily_interval,
    ]

    log.info(
        "=== INFINITY TRAINING LOOP | max_epochs=%s | symbols=%d ===",
        max_epochs or "∞",
        len(enabled_markets),
    )
    log.info("Infinity training program: %s", training_program)

    if payload_probe:
        try:
            probe_results = ai_orchestrator.orchestration_probe()
            db.set_cache("infinity_loop:payload_probe", probe_results)
            groq_results = [r for r in probe_results if r.get("provider") == "Groq"]
            if groq_results:
                _print_github_summary(
                    "## 🧪 Groq Payload Probe\n\n"
                    f"- Results: {len(groq_results)} response(s) recorded\n"
                )
        except Exception as exc:
            log.warning("Groq payload probe failed: %s", exc)

    global_epoch = 0
    exit_reason: Optional[str] = None
    while True:
        if max_epochs > 0 and global_epoch >= max_epochs:
            log.info("Infinity loop reached max_epochs=%d. Stopping.", max_epochs)
            exit_reason = "max_epochs"
            break

        global_epoch += 1
        supervised.increment_epoch()

        log.info("── Infinity Loop Epoch %d ──", global_epoch)

        # ── Fetch index data (daily, cached) ──────────────────────────────
        index_dfs: Dict[str, Any] = {}
        for idx_market in cfg.trading.index_markets:
            if not idx_market.enabled:
                continue
            try:
                df_idx = index_fetcher.fetch_ohlcv_history(
                    idx_market.symbol,
                    interval=cfg.data.daily_interval,
                    lookback_candles=cfg.data.training_lookback_candles,
                    yf_ticker=idx_market.yf_ticker or None,
                )
                if not df_idx.empty:
                    index_dfs[idx_market.symbol] = df_idx
            except Exception as exc:
                log.warning("Index fetch failed for %s: %s", idx_market.symbol, exc)

        # ── Train index models ────────────────────────────────────────────
        for idx_sym, idx_df in index_dfs.items():
            try:
                ensemble.train(idx_df, symbol=idx_sym)
            except Exception as exc:
                log.warning("Index training failed for %s: %s", idx_sym, exc)

        # ── Train crypto models ───────────────────────────────────────────
        trained_symbols: List[str] = []
        for market in enabled_markets:
            symbol = market.symbol
            tf_dataframes: Dict[str, Any] = {}
            for tf in all_timeframes:
                df_tf = dataset_mgr.get_or_fetch_dataset(
                    fetcher, symbol, tf,
                    lookback_candles=cfg.data.training_lookback_candles,
                    force_refresh=should_refresh_dataset,
                )
                if not df_tf.empty:
                    tf_dataframes[tf] = df_tf
            if not tf_dataframes:
                continue
            try:
                if (
                    training_program == "multi_timeframe"
                    and training_epochs > 1
                    and len(tf_dataframes) > 1
                ):
                    ensemble.train_multi_timeframe_with_progression(
                        tf_dataframes,
                        symbol=symbol,
                        epochs=training_epochs,
                        reinforcement_alpha=reinforcement_alpha,
                        primary_tf=cfg.data.primary_interval,
                    )
                elif training_program == "progressive" and training_epochs > 1:
                    primary_df = tf_dataframes.get(cfg.data.primary_interval) or next(
                        iter(tf_dataframes.values())
                    )
                    ensemble.train_with_progression(
                        primary_df,
                        symbol=symbol,
                        epochs=training_epochs,
                        reinforcement_alpha=reinforcement_alpha,
                    )
                else:
                    primary_df = tf_dataframes.get(cfg.data.primary_interval) or next(
                        iter(tf_dataframes.values())
                    )
                    ensemble.train(primary_df, symbol=symbol)
                trained_symbols.append(symbol)
            except Exception as exc:
                log.warning("Training failed for %s: %s", symbol, exc)
        if trained_symbols:
            _record_training_time(db, trained_symbols)

        # ── Periodic evaluation & hyperparameter adjustment ───────────────
        # Force an evaluation after the first epoch when exit-on-pass is enabled
        # so training can exit immediately if initial results satisfy thresholds.
        should_eval = supervised.should_evaluate() or (exit_on_pass and supervised.epoch == 1)
        if should_eval:
            trade_history = [asdict(t) for t in broker.trade_history]
            last_cycle = db.get_cache("signal:paper:last_cycle")
            cached_gemini_time = 0.0
            cached_action_time = 0.0
            if isinstance(last_cycle, dict):
                cached_gemini_time = float(last_cycle.get("avg_gemini_time_s", 0.0))
                cached_action_time = float(last_cycle.get("avg_action_time_s", 0.0))

            metrics, _ = evaluator.evaluate(
                trade_history,
                initial_equity=cfg.paper_broker.initial_equity,
                final_equity=broker.equity,
                gemini_answer_time_avg_s=cached_gemini_time,
                action_time_avg_s=cached_action_time,
            )
            metrics_dict = asdict(metrics)

            # Build AI-leader callback
            def _ai_callback(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                try:
                    review = ai_orchestrator.review_performance(
                        trade_history[-20:], m
                    )
                    if isinstance(review, dict):
                        return review.get("hyperparameter_suggestions")
                except Exception as exc:
                    log.warning("AI review failed: %s", exc)
                return None

            adjustments = supervised.evaluate_and_adjust_with_ai(
                trade_history,
                metrics_dict,
                ai_callback=_ai_callback,
            )
            passes = evaluator.passes_thresholds(metrics)

            # Log stab/pierce alerts
            if metrics.stab_alert or metrics.pierce_alert:
                alert_msg = []
                if metrics.stab_alert:
                    alert_msg.append("STAB")
                if metrics.pierce_alert:
                    alert_msg.append("PIERCE")
                log.warning(
                    "Epoch %d: %s alert(s) detected. Adjustments applied: %d",
                    global_epoch, "/".join(alert_msg), len(adjustments),
                )

            # Persist state
            supervised.save_state(sl_state_path)
            db.set_cache(
                "infinity_loop:last_eval",
                {
                    "epoch": global_epoch,
                    "total_trades": metrics.total_trades,
                    "sharpe": metrics.sharpe_ratio,
                    "win_rate": metrics.win_rate,
                    "stab_alert": metrics.stab_alert,
                    "pierce_alert": metrics.pierce_alert,
                    "adjustments": len(adjustments),
                    "pass": passes,
                },
            )
            _print_github_summary(
                f"## 🔁 Infinity Loop – Epoch {global_epoch}\n\n"
                f"- Trades: {metrics.total_trades} | "
                f"Sharpe: {metrics.sharpe_ratio:.3f} | "
                f"WR: {fmt_pct(metrics.win_rate)}\n"
                f"- Adjustments: {len(adjustments)}\n"
                f"- Stab: {'⚠️' if metrics.stab_alert else '✅'} | "
                f"Pierce: {'⚠️' if metrics.pierce_alert else '✅'} | "
                f"Thresholds: {'✅' if passes else '❌'}\n"
            )
            if exit_on_pass and passes:
                log.info(
                    "Infinity loop thresholds satisfied (epoch=%d). Exiting loop.",
                    global_epoch,
                )
                exit_reason = "thresholds_passed"
                break

    if exit_reason is None:
        exit_reason = "interrupted"
    db.record_task_completion(
        task_name="infinity_training",
        run_type="infinity-train",
        mode=cfg.trading.mode,
        status="success",
        metadata={"final_epoch": global_epoch, "exit_reason": exit_reason},
    )
    return 0



def run_paper_signal(config_path: Optional[Path] = None) -> int:
    """Run one signal evaluation cycle in paper-trading mode."""
    from src.ml_models import QuantumEnsemble, ModelDelegationAgent

    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)
    log.info("=== PAPER SIGNAL CYCLE ===")

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    delegation_agent = ModelDelegationAgent(ensemble)
    ai_orchestrator = MultiAIOrchestrator(cfg)
    gemini = ai_orchestrator._fallback  # GeminiOrchestrator for short-message payloads
    risk_mgr = RiskManager(cfg)
    supervised = SupervisedLearningModule(cfg)
    state_dir = Path(cfg.system.state_dir)

    # Load broker state
    broker = PaperBroker(cfg)
    broker_state_path = state_dir / "paper_broker.json"
    broker.load(broker_state_path)
    supervised.load_state(state_dir / "supervised_learning.json")

    # Load cached regimes from the previous signal cycle so the delegation
    # agent can immediately route to the correct model (regime is updated
    # after each AI analysis below).
    cached_regimes: Dict[str, str] = {}
    regime_cache = db.get_cache("signal:paper:regimes")
    if isinstance(regime_cache, dict):
        cached_regimes = regime_cache

    actions_taken: List[Dict] = []
    action_times: List[float] = []
    signal_payloads: List[Dict] = []

    for market in cfg.trading.markets:
        if not market.enabled:
            continue
        symbol = market.symbol
        action_start = time.monotonic()

        if not _ensure_model_ready(cfg, db, fetcher, ensemble, symbol):
            log.warning("No model for %s – skipping", symbol)
            continue

        # Fetch data
        snapshot = fetcher.fetch_all_market_data(symbol)
        df = snapshot["candles"].get(cfg.data.primary_interval)
        if df is None or df.empty:
            log.warning("No candle data for %s", symbol)
            continue

        current_price = float(df["close"].iloc[-1])
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.01

        # Multi-timeframe ML signal: combines 1m / 5m / 15m / 1h predictions.
        # Falls back to delegation agent when fewer than two timeframes are
        # available (e.g. first run, stub data, or missing intervals).
        prior_regime = cached_regimes.get(symbol, "unknown")
        ml_signal = _build_multiplex_signal(
            cfg, ensemble, delegation_agent, snapshot, prior_regime, symbol=symbol
        )
        log.info(
            "%s ML signal: %s (conf=%.3f, agree=%.2f, nn_primary=%s, delegated_to=%s)",
            symbol,
            {0: "FLAT", 1: "LONG", 2: "SHORT"}[ml_signal["signal"]],
            ml_signal["confidence"],
            ml_signal.get("agreement", 0.0),
            ml_signal.get("nn_decision", False),
            ml_signal.get("delegated_to", "ensemble"),
        )

        # Gemini validation
        current_leverage = cfg.trading.leverage.default
        gemini_analysis = ai_orchestrator.analyse_market_context(symbol, ml_signal, snapshot)
        validated_signal = gemini_analysis.get("validated_signal", ml_signal["signal"])
        regime = gemini_analysis.get("regime", "unknown")
        # Update cached regime for next cycle
        cached_regimes[symbol] = regime

        # Leverage adjustment
        recent_history = [asdict(t) for t in broker.trade_history[-20:]]
        perf_metrics = compute_metrics(
            recent_history, cfg.paper_broker.initial_equity, broker.equity
        )
        lev_rec = ai_orchestrator.recommend_leverage(
            symbol,
            ml_signal["confidence"],
            regime,
            current_leverage,
            asdict(perf_metrics),
        )
        final_leverage = risk_mgr.adjust_leverage(
            current_leverage,
            ml_signal["confidence"],
            lev_rec.get("recommended_leverage"),
        )

        # Build short message payload
        payload = gemini.build_short_message_payload(
            symbol, validated_signal, ml_signal["confidence"],
            regime, final_leverage, current_price,
        )
        signal_payloads.append(payload)
        log.info("Signal payload: %s", payload["message"])

        # Update positions (apply stops)
        funding_rate = snapshot.get("funding", {}).get("funding_rate", 0.0)
        closed = broker.update_positions(symbol, current_price, funding_rate)
        for closed_trade in closed:
            log.info("Auto-closed %s: %.2f USD (%s)", symbol, closed_trade.pnl, closed_trade.exit_reason)

        # Check existing position
        existing = broker.get_open_position(symbol)

        # Proactive risk-based closure: when the AI orchestrator flags elevated
        # risk conditions AND the current position is losing money, close it
        # immediately to prevent further losses (capital preservation).
        risk_flags = gemini_analysis.get("risk_flags", [])
        if risk_flags and existing and existing.unrealised_pnl < 0:
            log.warning(
                "Closing losing %s position proactively – risk flags: %s "
                "(unrealised=%.2f USD)",
                symbol, risk_flags, existing.unrealised_pnl,
            )
            broker.close_position(existing.position_id, current_price, "risk_flag")
            actions_taken.append(
                {
                    "action": "risk_close",
                    "symbol": symbol,
                    "price": current_price,
                    "risk_flags": risk_flags,
                }
            )
            existing = None

        # Execute signal
        if validated_signal in (1, 2):
            if existing and existing.side != ("long" if validated_signal == 1 else "short"):
                # Opposing signal – close existing
                broker.close_position(existing.position_id, current_price, "signal")
                existing = None

            if existing is None:
                pos_req = PositionRequest(
                    symbol=symbol,
                    signal=validated_signal,
                    confidence=ml_signal["confidence"],
                    current_price=current_price,
                    atr=atr,
                    equity=broker.equity,
                    leverage=final_leverage,
                    open_positions=len(broker.positions),
                )
                spec = risk_mgr.compute_position(
                    pos_req, [asdict(t) for t in broker.trade_history]
                )
                if spec.allowed:
                    pos = broker.open_position(spec, current_price)
                    if pos:
                        actions_taken.append(
                            {
                                "action": "open",
                                "symbol": symbol,
                                "side": spec.side,
                                "price": current_price,
                                "size_usd": spec.size_usd,
                                "leverage": final_leverage,
                                "regime": regime,
                            }
                        )
                else:
                    log.info("Trade rejected: %s", spec.reject_reason)

        elif validated_signal == 0 and existing:
            broker.close_position(existing.position_id, current_price, "signal")
            actions_taken.append(
                {"action": "close", "symbol": symbol, "price": current_price}
            )

        action_times.append(time.monotonic() - action_start)

    # Live-supervised learning evaluation
    all_history = [asdict(t) for t in broker.trade_history]
    if len(all_history) >= 10:
        sup_metrics = compute_metrics(
            all_history, cfg.paper_broker.initial_equity, broker.equity
        )
        supervised.evaluate_and_adjust(all_history, asdict(sup_metrics))
    supervised.save_state(state_dir / "supervised_learning.json")

    # Persist broker state
    broker.save(broker_state_path)

    equity = broker.get_equity()
    total_ret = (equity - cfg.paper_broker.initial_equity) / cfg.paper_broker.initial_equity
    avg_action_time = sum(action_times) / len(action_times) if action_times else 0.0
    avg_gemini_time = gemini.avg_answer_time
    summary = (
        f"## 📊 Paper Trading Cycle – {utc_now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"- **Equity:** {fmt_usd(equity)}\n"
        f"- **Total Return:** {fmt_pct(total_ret)}\n"
        f"- **Open Positions:** {len(broker.positions)}\n"
        f"- **Total Trades:** {len(broker.trade_history)}\n"
        f"- **Actions This Cycle:** {len(actions_taken)}\n"
        f"- **Avg Action Time:** {avg_action_time:.2f}s\n"
        f"- **Avg Gemini Time:** {avg_gemini_time:.2f}s\n"
    )
    _print_github_summary(summary)
    db.set_cache(
        "signal:paper:last_cycle",
        {
            "equity": equity,
            "total_return": total_ret,
            "actions": len(actions_taken),
            "trades": len(broker.trade_history),
            "num_positions": len(broker.positions),
            "avg_action_time_s": avg_action_time,
            "avg_gemini_time_s": avg_gemini_time,
            "signal_payloads": signal_payloads,
        },
    )
    # Persist updated regimes for the next signal cycle's delegation agent
    db.set_cache("signal:paper:regimes", cached_regimes)
    db.record_task_completion(
        task_name="paper_signal_cycle",
        run_type="signal",
        mode="paper",
        status="success",
        metadata={"actions_taken": len(actions_taken), "equity": equity},
    )
    log.info("Paper cycle complete. Equity: %s (return: %s)", fmt_usd(equity), fmt_pct(total_ret))
    return 0


def run_live_signal(config_path: Optional[Path] = None) -> int:
    """Run one signal evaluation cycle in live-trading mode."""
    from src.ml_models import QuantumEnsemble, ModelDelegationAgent

    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)
    log.info("=== LIVE SIGNAL CYCLE ===")

    private_key = _get_hyperliquid_private_key()
    if not private_key:
        log.error("HYPERLIQUID_SECRET/HYPERLIQUID_PRIVATE_KEY not set – aborting live trading")
        return 1

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    delegation_agent = ModelDelegationAgent(ensemble)
    ai_orchestrator = MultiAIOrchestrator(cfg)
    risk_mgr = RiskManager(cfg)
    live_trader = LiveTrader(cfg, private_key=private_key)

    # Load cached regimes for model delegation
    cached_regimes: Dict[str, str] = {}
    regime_cache = db.get_cache("signal:live:regimes")
    if isinstance(regime_cache, dict):
        cached_regimes = regime_cache

    state_dir = Path(cfg.system.state_dir)
    trade_log_path = state_dir / "live_trades.json"
    trade_log: List[Dict] = []
    if trade_log_path.exists():
        with open(trade_log_path) as fh:
            trade_log = json.load(fh)

    for market in cfg.trading.markets:
        if not market.enabled:
            continue
        symbol = market.symbol

        if not _ensure_model_ready(cfg, db, fetcher, ensemble, symbol):
            log.warning("No model for %s – skipping", symbol)
            continue

        snapshot = fetcher.fetch_all_market_data(symbol)
        df = snapshot["candles"].get(cfg.data.primary_interval)
        if df is None or df.empty:
            continue

        current_price = float(df["close"].iloc[-1])
        atr = float(df.get("atr_14", df["close"] * 0.01).iloc[-1])

        prior_regime = cached_regimes.get(symbol, "unknown")
        # Multi-timeframe signal: combines 1m / 5m / 15m / 1h predictions.
        ml_signal = _build_multiplex_signal(
            cfg, ensemble, delegation_agent, snapshot, prior_regime, symbol=symbol
        )
        gemini_analysis = ai_orchestrator.analyse_market_context(symbol, ml_signal, snapshot)
        validated_signal = gemini_analysis.get("validated_signal", ml_signal["signal"])
        regime = gemini_analysis.get("regime", "unknown")
        cached_regimes[symbol] = regime

        # Check risk flags from Gemini
        risk_flags = gemini_analysis.get("risk_flags", [])
        if risk_flags:
            log.warning("Risk flags for %s: %s", symbol, risk_flags)

        # Proactive risk-based closure for live positions: when the AI flags
        # elevated risk AND the position is losing money, close it immediately
        # to protect capital (mirrors the same logic in run_paper_signal).
        if risk_flags:
            live_pos_sym = [
                p for p in open_positions
                if p.get("position", {}).get("coin") == symbol
            ]
            for live_pos in live_pos_sym:
                pos_data = live_pos.get("position", {})
                unrealised_pnl = float(pos_data.get("unrealizedPnl", 0.0))
                if unrealised_pnl < 0:
                    side = pos_data.get("stype", "long")
                    size_contracts = abs(float(pos_data.get("szi", 0.0)))
                    if size_contracts > 0:
                        log.warning(
                            "Closing losing live %s %s position proactively "
                            "– risk flags: %s (unrealised=%.2f USD)",
                            symbol, side, risk_flags, unrealised_pnl,
                        )
                        live_trader.close_position(symbol, side, size_contracts, current_price)
                        # Refresh open positions count after closure.
                        open_positions = live_trader.get_open_positions()
                        n_open = len(open_positions)

        perf = compute_metrics(trade_log[-50:], cfg.trading.initial_equity, cfg.trading.initial_equity)
        lev_rec = ai_orchestrator.recommend_leverage(
            symbol, ml_signal["confidence"], regime,
            cfg.trading.leverage.default, asdict(perf)
        )
        final_leverage = risk_mgr.adjust_leverage(
            cfg.trading.leverage.default,
            ml_signal["confidence"],
            lev_rec.get("recommended_leverage"),
        )

        # Fetch live account state
        account = live_trader.get_account_state()
        equity = float(
            account.get("marginSummary", {}).get("accountValue", cfg.trading.initial_equity)
        )

        open_positions = live_trader.get_open_positions()
        n_open = len(open_positions)

        if validated_signal in (1, 2):
            pos_req = PositionRequest(
                symbol=symbol,
                signal=validated_signal,
                confidence=ml_signal["confidence"],
                current_price=current_price,
                atr=atr,
                equity=equity,
                leverage=final_leverage,
                open_positions=n_open,
            )
            spec = risk_mgr.compute_position(pos_req, trade_log)
            if spec.allowed:
                result = live_trader.place_market_order(spec)
                if result:
                    trade_log.append(
                        {
                            "symbol": symbol,
                            "side": spec.side,
                            "entry_price": result.filled_price,
                            "size_usd": spec.size_usd,
                            "leverage": final_leverage,
                            "stop_loss": spec.stop_loss,
                            "take_profit": spec.take_profit,
                            "timestamp_ms": result.timestamp_ms,
                            "regime": regime,
                            "pnl": 0.0,  # updated on close
                        }
                    )
            else:
                log.info("Live trade rejected: %s", spec.reject_reason)

    # Save trade log
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(trade_log_path, "w") as fh:
        json.dump(trade_log[-500:], fh, indent=2, default=str)

    summary = (
        f"## ⚡ Live Trading Cycle – {utc_now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"- **Mode:** LIVE\n"
        f"- **Open Positions:** {n_open}\n"
        f"- **Total Log Trades:** {len(trade_log)}\n"
    )
    _print_github_summary(summary)
    db.set_cache(
        "signal:live:last_cycle",
        {"open_positions": n_open, "trade_log_size": len(trade_log)},
    )
    # Persist updated regimes for the next signal cycle's delegation agent
    db.set_cache("signal:live:regimes", cached_regimes)
    db.record_task_completion(
        task_name="live_signal_cycle",
        run_type="signal",
        mode="live",
        status="success",
        metadata={"open_positions": n_open, "trade_log_size": len(trade_log)},
    )
    return 0


def run_evaluation(config_path: Optional[Path] = None) -> int:
    """Evaluate paper trading performance and emit adjustments."""
    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)
    log.info("=== EVALUATION MODE ===")

    state_dir = Path(cfg.system.state_dir)
    broker = PaperBroker(cfg)
    broker.load(state_dir / "paper_broker.json")

    evaluator = Evaluator(cfg)
    trade_history = [asdict(t) for t in broker.trade_history]

    # Retrieve cached timing metrics from the last signal cycle.
    last_cycle = db.get_cache("signal:paper:last_cycle")
    cached_gemini_time = 0.0
    cached_action_time = 0.0
    if isinstance(last_cycle, dict):
        cached_gemini_time = float(last_cycle.get("avg_gemini_time_s", 0.0))
        cached_action_time = float(last_cycle.get("avg_action_time_s", 0.0))

    metrics, adjustments = evaluator.evaluate(
        trade_history,
        initial_equity=cfg.paper_broker.initial_equity,
        final_equity=broker.equity,
        num_positions=len(broker.positions),
        gemini_answer_time_avg_s=cached_gemini_time,
        action_time_avg_s=cached_action_time,
    )

    evaluator.print_report(metrics, adjustments)

    results_dir = Path(cfg.system.results_dir)
    evaluator.save_report(
        metrics, adjustments,
        results_dir / "evaluation_report.json",
        label=f"eval_{utc_now().strftime('%Y%m%d_%H%M')}",
    )

    # Evaluation-driven model weight update.
    # Re-weight each symbol's ensemble using cached per-model training accuracy
    # so that better-performing models carry more influence in the next signal
    # cycle.  The reinforcement step size is config-driven via
    # cfg.ml.reinforcement_alpha (and may be adaptively increased, e.g. doubled
    # on poor win rate) to keep adjustments stable across evaluation windows.
    _update_model_weights_from_evaluation(cfg, db, metrics)

    # Gemini performance review
    ai_orchestrator = MultiAIOrchestrator(cfg)
    perf_review = ai_orchestrator.review_performance(trade_history[-20:], asdict(metrics))
    pause_trading = bool(perf_review.get("pause_trading"))
    pause_reason = perf_review.get("pause_reason", "")
    if pause_trading:
        log.warning("⚠️  Gemini recommends pausing trading: %s", pause_reason)
        _print_github_summary(
            f"## ⚠️ Trading Paused\n\n{pause_reason}"
        )

    passes = evaluator.passes_thresholds(metrics)
    summary = (
        f"## 📈 Performance Evaluation – {utc_now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total Return | {fmt_pct(metrics.total_return_pct)} |\n"
        f"| Sharpe Ratio | {metrics.sharpe_ratio:.3f} |\n"
        f"| Max Drawdown | {fmt_pct(metrics.max_drawdown_pct)} |\n"
        f"| Win Rate | {fmt_pct(metrics.win_rate)} |\n"
        f"| Accuracy | {fmt_pct(metrics.accuracy)} |\n"
        f"| Equity Growth | {fmt_pct(metrics.equity_growth_pct)} |\n"
        f"| Profit Factor | {metrics.profit_factor:.3f} |\n"
        f"| Total Trades | {metrics.total_trades} |\n"
        f"| Num Positions | {metrics.num_positions} |\n"
        f"| Avg Trade Length | {metrics.avg_trade_duration_hours:.2f}h |\n"
        f"| Avg Leverage | {metrics.avg_leverage_used:.1f}x |\n"
        f"| Gemini Avg Time | {metrics.gemini_answer_time_avg_s:.2f}s |\n"
        f"| Action Avg Time | {metrics.action_time_avg_s:.2f}s |\n\n"
        f"**Threshold Check:** {'✅ PASS' if passes else '❌ FAIL'}\n"
    )
    if adjustments:
        summary += "\n### Auto-Adjustments\n"
        for adj in adjustments:
            summary += f"- `{adj['parameter']}`: {adj['old_value']} → **{adj['new_value']}** _{adj['reason']}_\n"
    _print_github_summary(summary)
    db.set_cache(
        "evaluation:last_metrics",
        {
            "pass": passes,
            "total_trades": metrics.total_trades,
            "sharpe_ratio": metrics.sharpe_ratio,
            "pause_trading": pause_trading,
            "pause_reason": pause_reason,
        },
    )
    db.record_task_completion(
        task_name="model_evaluation",
        run_type="evaluate",
        mode=cfg.trading.mode,
        status="success",
        metadata={"pass": passes, "total_trades": metrics.total_trades},
    )
    return 0


def run_full_cycle(config_path: Optional[Path] = None) -> int:
    """Run the full pipeline end-to-end on a consistent data snapshot.

    Returns 0 on success; non-zero if any pipeline stage fails. Behavior can be
    influenced by DATA_SNAPSHOT_END_MS (freeze data), TRADING_ELIGIBILITY_OVERRIDE
    (force eligibility), and LIVE_TRADING_ENABLED (allow live execution).
    """
    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)
    mode = os.environ.get("TRADING_MODE", cfg.trading.mode)

    log.info("=== FULL PIPELINE CYCLE ===")
    snapshot_end_ms = _ensure_data_snapshot_end_ms()
    snapshot_dt = datetime.fromtimestamp(snapshot_end_ms / 1000, tz=timezone.utc)
    log.info(
        "Data snapshot end time locked to %s (%s)",
        snapshot_end_ms,
        snapshot_dt.isoformat(),
    )

    steps = [
        ("training", run_training),
        ("signal", run_paper_signal),
        ("evaluate", run_evaluation),
        ("export", run_model_export),
    ]
    for name, step in steps:
        step_result = step(config_path)
        if step_result != 0:
            log.error("Full cycle failed during %s step", name)
            db.record_task_completion(
                task_name="full_cycle",
                run_type="full-cycle",
                mode=mode,
                status="failed",
                metadata={"failed_step": name},
            )
            return step_result

    eligible, reason = _resolve_trading_eligibility(db)
    _print_github_summary(
        "## 🧭 Trading Eligibility\n\n"
        f"- **Allowed:** {'✅' if eligible else '❌'}\n"
        f"- **Reason:** {reason}\n"
    )
    if not eligible:
        log.warning("Trading eligibility check failed: %s", reason)
        db.record_task_completion(
            task_name="full_cycle",
            run_type="full-cycle",
            mode=mode,
            status="success",
            metadata={"trading_allowed": False, "reason": reason},
        )
        return 0

    if mode == "live":
        if not _is_live_trading_enabled():
            log.warning("LIVE_TRADING_ENABLED not set; skipping live trading step")
        else:
            if os.environ.pop("DATA_SNAPSHOT_END_MS", None) is not None:
                log.info(
                    "Cleared DATA_SNAPSHOT_END_MS for live trading; using current market data"
                )
            trade_rc = run_live_signal(config_path)
            if trade_rc != 0:
                db.record_task_completion(
                    task_name="full_cycle",
                    run_type="full-cycle",
                    mode=mode,
                    status="failed",
                    metadata={"failed_step": "live_signal"},
                )
                return trade_rc

    db.record_task_completion(
        task_name="full_cycle",
        run_type="full-cycle",
        mode=mode,
        status="success",
        metadata={"trading_allowed": True, "snapshot_end_ms": snapshot_end_ms},
    )
    return 0


def run_training_pipeline(config_path: Optional[Path] = None) -> int:
    """Run the training pipeline in separate real-time stages.

    Executes training → evaluation → export sequentially while recording
    per-stage progress in the cache for monitoring.
    """
    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)
    mode = os.environ.get("TRADING_MODE", cfg.trading.mode)

    log.info("=== TRAINING PIPELINE ===")
    snapshot_override = os.environ.pop("DATA_SNAPSHOT_END_MS", None)
    if snapshot_override is not None:
        log.info("Cleared DATA_SNAPSHOT_END_MS; using real-time data for pipeline")

    def _escape_markdown_error(error: str) -> str:
        """Remove newlines and escape markdown special characters for summary output."""
        cleaned = error.replace("\\", "\\\\")
        cleaned = cleaned.replace("\n", " ").strip()
        return re.sub(r"([-`*_\[\]()#+!|<>])", r"\\\1", cleaned)

    def _record_stage(
        stage: str,
        status: str,
        rc: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "stage": stage,
            "status": status,
            "timestamp": utc_now().isoformat(),
        }
        if rc is not None:
            payload["exit_code"] = rc
        safe_error = ""
        if error:
            safe_error = _escape_markdown_error(error)
            payload["error"] = safe_error
        db.set_cache("training_pipeline:progress", payload)
        summary = (
            f"### 🧪 Training Pipeline – {stage}\n\n"
            f"- Status: **{status.upper()}**\n"
        )
        if safe_error:
            summary += f"- Error: {safe_error}\n"
        _print_github_summary(summary)

    def _record_failure(stage: str, rc: int, error: Optional[str] = None) -> int:
        sanitized_error = _escape_markdown_error(error) if error else None
        _record_stage(stage, "failed", rc, error=error)
        metadata = {"failed_stage": stage}
        if sanitized_error:
            metadata["error"] = sanitized_error
        db.record_task_completion(
            task_name="training_pipeline",
            run_type="training-pipeline",
            mode=mode,
            status="failed",
            metadata=metadata,
        )
        return rc

    steps = [
        ("training", run_training),
        ("evaluate", run_evaluation),
        ("export", run_model_export),
    ]
    try:
        for name, step in steps:
            _record_stage(name, "running")
            try:
                step_result = step(config_path)
            except Exception as exc:
                log.exception("Training pipeline errored during %s stage", name)
                return _record_failure(name, 1, error=str(exc))
            if step_result != 0:
                log.error("Training pipeline failed during %s stage", name)
                return _record_failure(name, step_result)
            _record_stage(name, "completed", step_result)

        db.record_task_completion(
            task_name="training_pipeline",
            run_type="training-pipeline",
            mode=mode,
            status="success",
            metadata={"stages": [s for s, _ in steps]},
        )
        return 0
    finally:
        if snapshot_override is not None:
            os.environ["DATA_SNAPSHOT_END_MS"] = snapshot_override
        db.close()


def run_model_export(config_path: Optional[Path] = None) -> int:
    """Export trained sklearn models (GB, RF, Linear) to ONNX format and
    save per-symbol OHLCV training data as CSV files.

    Scheduled daily at 12:00 UTC – the midpoint of the 24-hour training
    window that starts at 00:00 UTC – so model redeployment never overlaps
    with active model training.
    """
    from src.ml_models import QuantumEnsemble

    cfg = load_config(config_path)
    db = _build_db_manager(cfg)
    log.setLevel(cfg.system.log_level)
    log.info("=== MODEL EXPORT ===")

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    enabled_markets = [m for m in cfg.trading.markets if m.enabled]

    csv_dir = Path(cfg.data.historical_csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    export_results: Dict[str, Any] = {}
    csv_results: Dict[str, str] = {}

    for market in enabled_markets:
        symbol = market.symbol
        loaded = ensemble.load(symbol)
        if not loaded:
            log.warning("No trained model for %s – skipping export", symbol)
            continue

        # Export GB, RF, Linear models to ONNX
        onnx_paths = ensemble.export_onnx(symbol)
        export_results[symbol] = onnx_paths

        # Save OHLCV data as CSV for evaluation / hyperparameter tuning
        df = fetcher.fetch_candles(
            symbol,
            cfg.data.primary_interval,
            lookback_candles=cfg.data.lookback_candles,
        )
        if not df.empty:
            csv_path = csv_dir / f"{symbol}_ohlcv.csv"
            df.to_csv(csv_path, index=False)
            csv_results[symbol] = str(csv_path)
            log.info("Saved OHLCV CSV for %s → %s", symbol, csv_path)

    total_exported = sum(len(v) for v in export_results.values())
    summary = (
        f"## 🚀 Model Export – {utc_now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"| Symbol | ONNX Models Exported | CSV Saved |\n|---|---|---|\n"
    )
    for sym in [m.symbol for m in enabled_markets if m.symbol in export_results]:
        onnx_names = ", ".join(export_results[sym].keys()) or "none"
        csv_saved = "✅" if sym in csv_results else "❌"
        summary += f"| {sym} | {onnx_names} | {csv_saved} |\n"
    summary += f"\n**Total ONNX models exported:** {total_exported}\n"
    _print_github_summary(summary)

    db.set_cache("model_export:last_results", {
        "timestamp": utc_now().isoformat(),
        "onnx_exported": {sym: list(v.keys()) for sym, v in export_results.items()},
        "csv_saved": list(csv_results.keys()),
    })
    db.record_task_completion(
        task_name="model_export",
        run_type="export-models",
        mode=cfg.trading.mode,
        status="success",
        metadata={"total_onnx": total_exported, "csv_symbols": list(csv_results.keys())},
    )
    return 0


def run_health_check(cfg_path: Optional[Path] = None) -> int:
    """Check connectivity and realtime inference for all AI providers and ML models.

    Sends a probe request to each configured AI provider and verifies
    that each loaded ML model can run inference on synthetic data.  Prints
    a structured Markdown report and writes a GitHub Actions step summary
    when running in CI.

    Returns 0 regardless of individual probe failures so the check can be
    used as a non-blocking diagnostic in any workflow stage.
    """
    cfg = load_config(cfg_path)
    log.info("=== AI / ML Health Check ===")

    # ── AI Provider Check ─────────────────────────────────────────────────────
    ai_orchestrator = MultiAIOrchestrator(cfg)
    ai_results = ai_orchestrator.health_check()

    # ── ML Model Check ────────────────────────────────────────────────────────
    # Lazy import avoids loading TensorFlow / heavy dependencies unnecessarily.
    from src.ml_models import QuantumEnsemble  # noqa: PLC0415

    enabled_markets = [m for m in cfg.trading.markets if m.enabled]
    ml_results: Dict[str, List[Dict[str, Any]]] = {}
    for market in enabled_markets:
        ensemble = QuantumEnsemble(cfg)
        if ensemble.load(market.symbol):
            ml_results[market.symbol] = ensemble.health_check()
        else:
            ml_results[market.symbol] = [{
                "model": "ensemble",
                "status": "not_loaded",
                "latency_ms": None,
                "signal": None,
                "error": "no saved models found",
            }]

    # ── Build Report ──────────────────────────────────────────────────────────
    ts = utc_now().strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [f"## 🏥 Health Check – {ts}\n"]

    lines.append("### AI Providers\n")
    lines.append("| Provider | Status | Latency | Notes |")
    lines.append("|---|---|---|---|")
    _status_icon = {"ok": "✅", "error": "❌", "skipped": "⏭️"}
    for r in ai_results:
        icon = _status_icon.get(r["status"], "❓")
        latency = f"{r['latency_ms']:.0f}ms" if r.get("latency_ms") is not None else "—"
        lines.append(
            f"| {r['provider']} | {icon} {r['status']} | {latency} | {r.get('error', '')} |"
        )
    lines.append("")

    # ── Orchestration Probe ────────────────────────────────────────────────────
    log.info("=== Orchestration Probe ===")
    probe_results = ai_orchestrator.orchestration_probe()

    lines.append("### Orchestration Pipeline Probe\n")
    lines.append(
        "_Simulates the full live-trading sequence: "
        "market context → leverage → performance review_\n"
    )
    lines.append("| Provider | Step | Status | Latency | Notes |")
    lines.append("|---|---|---|---|---|")
    for r in probe_results:
        icon = _status_icon.get(r["status"], "❓")
        latency = f"{r['latency_ms']:.0f}ms" if r.get("latency_ms") is not None else "—"
        step_label = {
            "market_context": "1. Market Context",
            "leverage": "2. Leverage",
            "performance": "3. Performance",
        }.get(r.get("step", ""), r.get("step", ""))
        lines.append(
            f"| {r['provider']} | {step_label} | {icon} {r['status']} "
            f"| {latency} | {r.get('error', '')} |"
        )
    lines.append("")

    lines.append("### ML Models\n")
    _ml_status_icon = {"ok": "✅", "error": "❌", "not_loaded": "⬜"}
    _signal_labels = {0: "FLAT", 1: "LONG", 2: "SHORT"}
    for symbol, results in ml_results.items():
        lines.append(f"**{symbol}**\n")
        lines.append("| Model | Status | Latency | Signal | Notes |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            icon = _ml_status_icon.get(r["status"], "❓")
            latency = f"{r['latency_ms']:.2f}ms" if r.get("latency_ms") is not None else "—"
            sig = _signal_labels.get(r.get("signal"), "—")  # type: ignore[arg-type]
            lines.append(
                f"| {r['model']} | {icon} {r['status']} | {latency} | {sig} | {r.get('error', '')} |"
            )
        lines.append("")

    summary = "\n".join(lines)
    print(summary)
    _print_github_summary(summary)
    log.info("Health check complete")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quantum Trading System")
    parser.add_argument(
        "--run-type",
        choices=[
            "training", "signal", "evaluate", "train-models", "infinity-train",
            "export-models", "full-cycle", "training-pipeline", "health-check",
        ],
        required=True,
        help="What to run",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "test"],
        default=None,
        help="Trading mode override (test uses synthetic data, no API calls required)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML",
    )
    args = parser.parse_args(argv)

    if args.mode:
        os.environ["TRADING_MODE"] = args.mode

    cfg_path = args.config

    if args.run_type in ("training", "train-models"):
        return run_training(cfg_path)
    elif args.run_type == "infinity-train":
        return run_infinity_training(cfg_path)
    elif args.run_type == "signal":
        mode = os.environ.get("TRADING_MODE", "paper")
        if mode == "live":
            return run_live_signal(cfg_path)
        return run_paper_signal(cfg_path)
    elif args.run_type == "evaluate":
        return run_evaluation(cfg_path)
    elif args.run_type == "export-models":
        return run_model_export(cfg_path)
    elif args.run_type == "full-cycle":
        return run_full_cycle(cfg_path)
    elif args.run_type == "training-pipeline":
        return run_training_pipeline(cfg_path)
    elif args.run_type == "health-check":
        return run_health_check(cfg_path)
    else:
        log.error("Unknown run type: %s", args.run_type)
        return 1


if __name__ == "__main__":
    sys.exit(main())
