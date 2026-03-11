from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.config import load_config
from src.evaluator import PerformanceMetrics, compute_metrics
from src.ml_models import QuantumEnsemble
from src.risk_manager import PositionRequest, RiskManager
from src.utils import add_all_features

try:
    from src.gemini_orchestrator import GeminiOrchestrator
except (ImportError, NameError) as exc:
    GeminiOrchestrator = None
    GEMINI_IMPORT_ERROR = exc
else:
    GEMINI_IMPORT_ERROR = None

log = logging.getLogger("gemini-bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BridgeState:
    def __init__(self) -> None:
        config_path = os.getenv("TRADING_CONFIG_PATH")
        self.cfg = load_config(Path(config_path)) if config_path else load_config()
        self.ensemble = QuantumEnsemble(self.cfg)
        self.gemini = GeminiOrchestrator(self.cfg) if GeminiOrchestrator else None
        self.risk_mgr = RiskManager(self.cfg)
        self.loaded_symbols: set[str] = set()

    def ensure_models(self, symbol: str) -> bool:
        if symbol in self.loaded_symbols:
            return True
        if self.ensemble.load(symbol):
            self.loaded_symbols.add(symbol)
            return True
        return False

    def handle_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(payload.get("symbol", "BTC"))
        candles = payload.get("candles", [])
        if not isinstance(candles, list) or not candles:
            return {"status": "error", "message": "candles payload missing"}

        df = pd.DataFrame(candles)
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(df.columns):
            return {"status": "error", "message": "candles require OHLCV fields"}

        for column in required_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=list(required_columns))
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time")

        if df.empty:
            return {"status": "error", "message": "candles payload invalid"}

        if not self.ensure_models(symbol):
            log.warning("No saved models found for symbol '%s'", symbol)
            return {
                "status": "model_unavailable",
                "symbol": symbol,
                "signal": 0,
                "confidence": 0.0,
                "message": f"models not available for symbol '{symbol}'",
            }

        features = add_all_features(df)
        if features.empty:
            return {
                "status": "error",
                "message": "not enough candles to compute features",
            }

        ml_signal = self.ensemble.predict(features)
        last_close = float(features["close"].iloc[-1])
        market_snapshot = self._build_market_snapshot(payload, last_close)

        gemini_result = self._analyse_with_gemini(symbol, ml_signal, market_snapshot)
        confidence = float(ml_signal.get("confidence", 0.0)) + float(
            gemini_result.get("confidence_adjustment", 0.0)
        )
        confidence = max(0.0, min(1.0, confidence))
        final_signal = int(gemini_result.get("validated_signal", ml_signal.get("signal", 0)))

        account_payload = payload.get("account")
        positions_payload = payload.get("positions")
        account = account_payload if isinstance(account_payload, dict) else {}
        positions = positions_payload if isinstance(positions_payload, dict) else {}
        equity = self._coerce_float(account.get("equity"), self.cfg.trading.initial_equity)
        balance = self._coerce_float(account.get("balance"), self.cfg.trading.initial_equity)
        default_leverage = self.cfg.trading.leverage.default or self.cfg.trading.leverage.min
        current_leverage = int(self._coerce_float(account.get("leverage"), default_leverage))
        open_positions = int(positions.get("long", 0)) + int(positions.get("short", 0))
        if payload.get("reset_daily"):
            self.risk_mgr.reset_daily_tracking(equity)

        recent_trades = payload.get("recent_trades", [])
        if not isinstance(recent_trades, list):
            recent_trades = []
        recent_trades = [t for t in recent_trades if isinstance(t, dict)]
        perf = compute_metrics(recent_trades, balance, equity)
        lev_rec = self._recommend_leverage(
            symbol,
            ml_signal.get("confidence", 0.0),
            gemini_result.get("regime", "unknown"),
            current_leverage,
            perf,
        )
        final_leverage = self.risk_mgr.adjust_leverage(
            current_leverage,
            confidence,
            lev_rec.get("recommended_leverage", current_leverage),
        )

        position_payload = None
        if final_signal in (1, 2):
            atr_value = float(features["atr_14"].iloc[-1]) if "atr_14" in features.columns else 0.0
            position_spec = self.risk_mgr.compute_position(
                PositionRequest(
                    symbol=symbol,
                    signal=final_signal,
                    confidence=confidence,
                    current_price=last_close,
                    atr=atr_value,
                    equity=equity,
                    leverage=final_leverage,
                    open_positions=open_positions,
                ),
                trade_history=recent_trades,
            )
            position_payload = asdict(position_spec)

        return {
            "status": "ok",
            "symbol": symbol,
            "signal": final_signal,
            "confidence": confidence,
            "regime": gemini_result.get("regime", "unknown"),
            "model": {
                "signal": ml_signal.get("signal", 0),
                "confidence": ml_signal.get("confidence", 0.0),
            },
            "gemini": gemini_result,
            "leverage": {
                "current": current_leverage,
                "recommended": lev_rec.get("recommended_leverage", current_leverage),
                "final": final_leverage,
                "reasoning": lev_rec.get("reasoning", ""),
            },
            "position": position_payload,
            "metrics": asdict(perf),
        }

    def _analyse_with_gemini(
        self, symbol: str, ml_signal: Dict[str, Any], market_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.gemini:
            return self.gemini.analyse_market_context(symbol, ml_signal, market_snapshot)
        fallback_reason = "Gemini orchestrator unavailable"
        if GEMINI_IMPORT_ERROR:
            fallback_reason = f"{fallback_reason}: {GEMINI_IMPORT_ERROR}"
        return {
            "validated_signal": ml_signal.get("signal", 0),
            "confidence_adjustment": 0.0,
            "regime": "unknown",
            "reasoning": fallback_reason,
            "risk_flags": [],
        }

    def _build_market_snapshot(self, payload: Dict[str, Any], last_close: float) -> Dict[str, Any]:
        snapshot = payload.get("market_snapshot") if isinstance(payload.get("market_snapshot"), dict) else {}
        funding = snapshot.get("funding") if isinstance(snapshot.get("funding"), dict) else {}
        order_book = snapshot.get("order_book") if isinstance(snapshot.get("order_book"), dict) else {}
        return {
            "funding": {
                "mark_price": self._coerce_float(funding.get("mark_price"), last_close),
                "funding_rate": self._coerce_float(funding.get("funding_rate"), 0.0),
                "open_interest": self._coerce_float(funding.get("open_interest"), 0.0),
            },
            "order_book": {
                "order_book_imbalance": self._coerce_float(
                    order_book.get("order_book_imbalance"), 0.0
                ),
                "bid_ask_spread_bps": self._coerce_float(order_book.get("bid_ask_spread_bps"), 0.0),
            },
            "trade_flow_imbalance": self._coerce_float(
                snapshot.get("trade_flow_imbalance"), 0.0
            ),
        }

    def _recommend_leverage(
        self,
        symbol: str,
        ml_confidence: float,
        regime: str,
        current_leverage: int,
        perf: PerformanceMetrics,
    ) -> Dict[str, Any]:
        perf_payload = asdict(perf) if isinstance(perf, PerformanceMetrics) else {}
        if self.gemini:
            return self.gemini.recommend_leverage(
                symbol, ml_confidence, regime, current_leverage, perf_payload
            )
        return {
            "recommended_leverage": current_leverage,
            "reasoning": "Gemini unavailable – using current leverage",
        }

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


STATE = BridgeState()


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "GeminiBridge/1.0"

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self.send_error(404, "Not found")
            return
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/signal":
            self.send_error(404, "Not found")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"status": "error", "message": "invalid JSON"}, status=400)
            return

        response = STATE.handle_payload(payload)
        status = 200 if response.get("status") in {"ok", "model_unavailable"} else 400
        self._send_json(response, status=status)

    def log_message(self, format_string: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), format_string % args)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = os.getenv("BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("BRIDGE_PORT", "8001"))
    server = ThreadingHTTPServer((host, port), BridgeHandler)
    log.info("Gemini bridge listening on http://%s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
