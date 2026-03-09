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
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.data_fetcher import HyperliquidDataFetcher
from src.evaluator import Evaluator, compute_metrics
from src.gemini_orchestrator import GeminiOrchestrator
from src.ml_models import QuantumEnsemble
from src.paper_broker import PaperBroker
from src.live_trader import LiveTrader
from src.risk_manager import PositionRequest, RiskManager
from src.utils import get_logger, fmt_pct, fmt_usd, utc_now

log = get_logger(__name__)


def _print_github_summary(text: str) -> None:
    """Write to GitHub Actions step summary if available."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write(text + "\n")


def run_training(config_path: Optional[Path] = None) -> int:
    """Train ML models on historical Hyperliquid data."""
    cfg = load_config(config_path)
    log.setLevel(cfg.system.log_level)
    log.info("=== TRAINING MODE ===")

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)

    results: Dict[str, Any] = {}
    for market in cfg.trading.markets:
        if not market.enabled:
            continue
        symbol = market.symbol
        log.info("Fetching training data for %s …", symbol)
        df = fetcher.fetch_candles(
            symbol,
            cfg.data.primary_interval,
            lookback_candles=cfg.data.lookback_candles,
        )
        if df.empty:
            log.warning("No data for %s – skipping training", symbol)
            continue

        scores = ensemble.train(df, symbol=symbol)
        results[symbol] = scores

    output_path = Path(cfg.system.results_dir) / "training_scores.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)

    summary = "## 🤖 Model Training Results\n\n| Symbol | XGB | GB | RF | LSTM |\n|---|---|---|---|---|\n"
    for sym, scores in results.items():
        row = f"| {sym} | {scores.get('xgb', 'N/A'):.4f if isinstance(scores.get('xgb'), float) else 'N/A'} | {scores.get('gb', 'N/A'):.4f if isinstance(scores.get('gb'), float) else 'N/A'} | {scores.get('rf', 'N/A'):.4f if isinstance(scores.get('rf'), float) else 'N/A'} | {scores.get('lstm', 'N/A'):.4f if isinstance(scores.get('lstm'), float) else 'N/A'} |\n"
        summary += row
    _print_github_summary(summary)
    log.info("Training complete. Results: %s", results)
    return 0


def run_paper_signal(config_path: Optional[Path] = None) -> int:
    """Run one signal evaluation cycle in paper-trading mode."""
    cfg = load_config(config_path)
    log.setLevel(cfg.system.log_level)
    log.info("=== PAPER SIGNAL CYCLE ===")

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    gemini = GeminiOrchestrator(cfg)
    risk_mgr = RiskManager(cfg)
    state_dir = Path(cfg.system.state_dir)

    # Load broker state
    broker = PaperBroker(cfg)
    broker_state_path = state_dir / "paper_broker.json"
    broker.load(broker_state_path)

    actions_taken: List[Dict] = []

    for market in cfg.trading.markets:
        if not market.enabled:
            continue
        symbol = market.symbol

        # Load model
        if not ensemble.load(symbol):
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

        # ML signal
        ml_signal = ensemble.predict(df)
        log.info(
            "%s ML signal: %s (conf=%.3f, agree=%.2f)",
            symbol,
            {0: "FLAT", 1: "LONG", 2: "SHORT"}[ml_signal["signal"]],
            ml_signal["confidence"],
            ml_signal["agreement"],
        )

        # Gemini validation
        current_leverage = cfg.trading.leverage.default
        gemini_analysis = gemini.analyse_market_context(symbol, ml_signal, snapshot)
        validated_signal = gemini_analysis.get("validated_signal", ml_signal["signal"])
        regime = gemini_analysis.get("regime", "unknown")

        # Leverage adjustment
        recent_history = [asdict(t) for t in broker.trade_history[-20:]]
        perf_metrics = compute_metrics(
            recent_history, cfg.paper_broker.initial_equity, broker.equity
        )
        lev_rec = gemini.recommend_leverage(
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

        # Update positions (apply stops)
        funding_rate = snapshot.get("funding", {}).get("funding_rate", 0.0)
        closed = broker.update_positions(symbol, current_price, funding_rate)
        for closed_trade in closed:
            log.info("Auto-closed %s: %.2f USD (%s)", symbol, closed_trade.pnl, closed_trade.exit_reason)

        # Check existing position
        existing = broker.get_open_position(symbol)

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

    # Persist broker state
    broker.save(broker_state_path)

    equity = broker.get_equity()
    total_ret = (equity - cfg.paper_broker.initial_equity) / cfg.paper_broker.initial_equity
    summary = (
        f"## 📊 Paper Trading Cycle – {utc_now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"- **Equity:** {fmt_usd(equity)}\n"
        f"- **Total Return:** {fmt_pct(total_ret)}\n"
        f"- **Open Positions:** {len(broker.positions)}\n"
        f"- **Total Trades:** {len(broker.trade_history)}\n"
        f"- **Actions This Cycle:** {len(actions_taken)}\n"
    )
    _print_github_summary(summary)
    log.info("Paper cycle complete. Equity: %s (return: %s)", fmt_usd(equity), fmt_pct(total_ret))
    return 0


def run_live_signal(config_path: Optional[Path] = None) -> int:
    """Run one signal evaluation cycle in live-trading mode."""
    cfg = load_config(config_path)
    log.setLevel(cfg.system.log_level)
    log.info("=== LIVE SIGNAL CYCLE ===")

    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY")
    if not private_key:
        log.error("HYPERLIQUID_PRIVATE_KEY not set – aborting live trading")
        return 1

    fetcher = HyperliquidDataFetcher(cfg)
    ensemble = QuantumEnsemble(cfg)
    gemini = GeminiOrchestrator(cfg)
    risk_mgr = RiskManager(cfg)
    live_trader = LiveTrader(cfg, private_key=private_key)

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

        if not ensemble.load(symbol):
            log.warning("No model for %s – skipping", symbol)
            continue

        snapshot = fetcher.fetch_all_market_data(symbol)
        df = snapshot["candles"].get(cfg.data.primary_interval)
        if df is None or df.empty:
            continue

        current_price = float(df["close"].iloc[-1])
        atr = float(df.get("atr_14", df["close"] * 0.01).iloc[-1])

        ml_signal = ensemble.predict(df)
        gemini_analysis = gemini.analyse_market_context(symbol, ml_signal, snapshot)
        validated_signal = gemini_analysis.get("validated_signal", ml_signal["signal"])
        regime = gemini_analysis.get("regime", "unknown")

        # Check risk flags from Gemini
        risk_flags = gemini_analysis.get("risk_flags", [])
        if risk_flags:
            log.warning("Risk flags for %s: %s", symbol, risk_flags)

        perf = compute_metrics(trade_log[-50:], cfg.trading.initial_equity, cfg.trading.initial_equity)
        lev_rec = gemini.recommend_leverage(
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
    return 0


def run_evaluation(config_path: Optional[Path] = None) -> int:
    """Evaluate paper trading performance and emit adjustments."""
    cfg = load_config(config_path)
    log.setLevel(cfg.system.log_level)
    log.info("=== EVALUATION MODE ===")

    state_dir = Path(cfg.system.state_dir)
    broker = PaperBroker(cfg)
    broker.load(state_dir / "paper_broker.json")

    evaluator = Evaluator(cfg)
    trade_history = [asdict(t) for t in broker.trade_history]
    metrics, adjustments = evaluator.evaluate(
        trade_history,
        initial_equity=cfg.paper_broker.initial_equity,
        final_equity=broker.equity,
    )

    report = evaluator.print_report(metrics, adjustments)

    results_dir = Path(cfg.system.results_dir)
    evaluator.save_report(
        metrics, adjustments,
        results_dir / "evaluation_report.json",
        label=f"eval_{utc_now().strftime('%Y%m%d_%H%M')}",
    )

    # Gemini performance review
    gemini = GeminiOrchestrator(cfg)
    perf_review = gemini.review_performance(trade_history[-20:], asdict(metrics))
    if perf_review.get("pause_trading"):
        log.warning("⚠️  Gemini recommends pausing trading: %s", perf_review.get("pause_reason"))
        _print_github_summary(
            f"## ⚠️ Trading Paused\n\n{perf_review.get('pause_reason', '')}"
        )

    passes = evaluator.passes_thresholds(metrics)
    summary = (
        f"## 📈 Performance Evaluation – {utc_now().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total Return | {fmt_pct(metrics.total_return_pct)} |\n"
        f"| Sharpe Ratio | {metrics.sharpe_ratio:.3f} |\n"
        f"| Max Drawdown | {fmt_pct(metrics.max_drawdown_pct)} |\n"
        f"| Win Rate | {fmt_pct(metrics.win_rate)} |\n"
        f"| Profit Factor | {metrics.profit_factor:.3f} |\n"
        f"| Total Trades | {metrics.total_trades} |\n"
        f"| Avg Leverage | {metrics.avg_leverage_used:.1f}x |\n\n"
        f"**Threshold Check:** {'✅ PASS' if passes else '❌ FAIL'}\n"
    )
    if adjustments:
        summary += "\n### Auto-Adjustments\n"
        for adj in adjustments:
            summary += f"- `{adj['parameter']}`: {adj['old_value']} → **{adj['new_value']}** _{adj['reason']}_\n"
    _print_github_summary(summary)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quantum Trading System")
    parser.add_argument(
        "--run-type",
        choices=["training", "signal", "evaluate", "train-models"],
        required=True,
        help="What to run",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=None,
        help="Trading mode override",
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
    elif args.run_type == "signal":
        mode = os.environ.get("TRADING_MODE", "paper")
        if mode == "live":
            return run_live_signal(cfg_path)
        return run_paper_signal(cfg_path)
    elif args.run_type == "evaluate":
        return run_evaluation(cfg_path)
    else:
        log.error("Unknown run type: %s", args.run_type)
        return 1


if __name__ == "__main__":
    sys.exit(main())
