"""
Quantum Trading System – Google Gemini AI Orchestrator
Uses Gemini to analyse market context, validate ML signals, recommend leverage,
review performance, and suggest trading strategy adjustments.

Supports dual-model orchestration:
  - gemini-2.5-pro (GEMINI_API_KEY): primary analysis for all symbols
  - gemini-2.5-pro (GEMINI_API_KEY2): deep reasoning for performance review
"""

from __future__ import annotations

import json
import textwrap
import time
import warnings
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.utils import fmt_pct, fmt_usd, get_logger

log = get_logger(__name__)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    log.warning("google-generativeai not installed – Gemini orchestrator disabled")


_SYSTEM_PROMPT = textwrap.dedent("""
    You are the Quantum Trading Orchestrator AI for a fully automated
    perpetuals trading system on Hyperliquid. Your role is to:

    1. Validate ML model signals given current market context.
    2. Recommend appropriate leverage (10–35x) based on regime and confidence.
    3. Identify regime (trending / ranging / volatile / consolidating).
    4. Review recent trade performance and suggest concrete adjustments.
    5. Flag elevated risk conditions that should pause trading.

    Always respond with valid JSON matching the schema requested in each prompt.
    Be concise, precise, and conservative – capital preservation is paramount.
""")


class GeminiOrchestrator:
    """Interfaces with Google Gemini AI to orchestrate trading decisions.

    Supports dual-model orchestration using two API keys:
      - Primary (gemini-2.5-pro): market analysis and signal validation
      - Secondary (gemini-2.5-pro): deep performance review and strategy tuning
    """

    _PRIMARY_MODEL_PREFERENCE = (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    )
    _SECONDARY_MODEL_PREFERENCE = (
        "gemini-2.5-pro",
        "gemini-1.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    )

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.gcfg = config.gemini
        self._model = None
        self._model_2 = None
        self._api_key_1 = self.gcfg.api_key
        self._api_key_2 = self.gcfg.api_key_2
        self._last_answer_times: List[float] = []
        self._max_answer_times = 500

        if _GENAI_AVAILABLE and self.gcfg.api_key:
            genai.configure(api_key=self.gcfg.api_key)
            available_primary = self._list_supported_models()
            selected_primary = self._select_preferred_model(
                available_primary,
                self._PRIMARY_MODEL_PREFERENCE,
                self.gcfg.model,
            )
            if available_primary and selected_primary != self.gcfg.model:
                log.warning(
                    "Gemini primary model '%s' unavailable; switching to '%s'",
                    self.gcfg.model,
                    selected_primary,
                )
                self.gcfg.model = selected_primary
            self._model = genai.GenerativeModel(
                model_name=self.gcfg.model,
                system_instruction=_SYSTEM_PROMPT,
            )
            log.info("Gemini primary model '%s' initialised", self.gcfg.model)

            if self.gcfg.api_key_2:
                try:
                    # Reconfigure with second API key for the secondary model.
                    genai.configure(api_key=self.gcfg.api_key_2)
                    available_secondary = self._list_supported_models()
                    selected_secondary = self._select_preferred_model(
                        available_secondary,
                        self._SECONDARY_MODEL_PREFERENCE,
                        self.gcfg.model_2,
                    )
                    if available_secondary and selected_secondary != self.gcfg.model_2:
                        log.warning(
                            "Gemini secondary model '%s' unavailable; switching to '%s'",
                            self.gcfg.model_2,
                            selected_secondary,
                        )
                        self.gcfg.model_2 = selected_secondary
                    self._model_2 = genai.GenerativeModel(
                        model_name=self.gcfg.model_2,
                        system_instruction=_SYSTEM_PROMPT,
                    )
                    log.info(
                        "Gemini secondary model '%s' initialised",
                        self.gcfg.model_2,
                    )
                    # Restore primary API key as the active configuration.
                    genai.configure(api_key=self.gcfg.api_key)
                except Exception as exc:
                    log.warning(
                        "Failed to initialise secondary Gemini model '%s': %s",
                        self.gcfg.model_2,
                        exc,
                    )
                    # Restore primary API key on failure.
                    genai.configure(api_key=self.gcfg.api_key)
        else:
            log.warning(
                "Gemini unavailable (api_key=%s, library=%s) – using fallback heuristics",
                bool(self.gcfg.api_key),
                _GENAI_AVAILABLE,
            )

    @property
    def avg_answer_time(self) -> float:
        """Average Gemini response time in seconds."""
        if not self._last_answer_times:
            return 0.0
        return sum(self._last_answer_times) / len(self._last_answer_times)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse_market_context(
        self,
        symbol: str,
        ml_signal: Dict[str, Any],
        market_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ask Gemini to validate the ML signal in context and return:
        {
            "validated_signal": 0|1|2,
            "confidence_adjustment": float,   # -0.2 to +0.2
            "regime": str,
            "reasoning": str,
            "risk_flags": [str],
        }
        """
        if self._model is None:
            return self._fallback_market_analysis(ml_signal)

        prompt = self._build_market_context_prompt(symbol, ml_signal, market_snapshot)
        response = self._call_gemini(prompt)
        if response:
            try:
                result = json.loads(self._extract_json(response))
                log.info("Gemini market analysis for %s: %s", symbol, result.get("regime"))
                return result
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Failed to parse Gemini market analysis: %s", exc)
        return self._fallback_market_analysis(ml_signal)

    def recommend_leverage(
        self,
        symbol: str,
        ml_confidence: float,
        regime: str,
        current_leverage: int,
        recent_performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return:
        {
            "recommended_leverage": int,
            "reasoning": str,
        }
        """
        if self._model is None:
            return self._fallback_leverage(ml_confidence, current_leverage)

        prompt = self._build_leverage_prompt(
            symbol, ml_confidence, regime, current_leverage, recent_performance
        )
        response = self._call_gemini(prompt)
        if response:
            try:
                result = json.loads(self._extract_json(response))
                lev = int(result.get("recommended_leverage", current_leverage))
                lev = max(self.cfg.trading.leverage.min, min(self.cfg.trading.leverage.max, lev))
                result["recommended_leverage"] = lev
                log.info("Gemini leverage recommendation for %s: %dx", symbol, lev)
                return result
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("Failed to parse Gemini leverage recommendation: %s", exc)
        return self._fallback_leverage(ml_confidence, current_leverage)

    def review_performance(
        self,
        recent_trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Uses the secondary model (gemini-2.5-pro) for deep performance review.
        Return:
        {
            "adjustments": [{"parameter": str, "old_value": ..., "new_value": ..., "reason": str}],
            "overall_assessment": str,
            "pause_trading": bool,
            "pause_reason": str,
        }
        """
        model = self._model_2 or self._model
        if model is None:
            return self._fallback_performance_review(metrics)

        prompt = self._build_performance_review_prompt(recent_trades, metrics)
        response = self._call_gemini(prompt, model=model)
        if response:
            try:
                result = json.loads(self._extract_json(response))
                log.info(
                    "Gemini performance review: %s (pause=%s)",
                    result.get("overall_assessment", ""),
                    result.get("pause_trading", False),
                )
                return result
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Failed to parse Gemini performance review: %s", exc)
        return self._fallback_performance_review(metrics)

    def build_short_message_payload(
        self,
        symbol: str,
        signal: int,
        confidence: float,
        regime: str,
        leverage: int,
        price: float,
    ) -> Dict[str, Any]:
        """Build a compact trade signal payload suitable for messaging."""
        signal_label = {0: "FLAT", 1: "LONG", 2: "SHORT"}.get(signal, "FLAT")
        return {
            "symbol": symbol,
            "signal": signal_label,
            "confidence": round(confidence, 4),
            "regime": regime,
            "leverage": leverage,
            "price": round(price, 4),
            "message": (
                f"{symbol} {signal_label} | conf={confidence:.2%} | "
                f"{regime} | {leverage}x @ {price:.2f}"
            ),
        }

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_market_context_prompt(
        self,
        symbol: str,
        ml_signal: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> str:
        funding = snapshot.get("funding", {})
        order_book = snapshot.get("order_book", {})
        signal_map = {0: "FLAT", 1: "LONG", 2: "SHORT"}
        return textwrap.dedent(f"""
            Analyse the following market context for {symbol} and validate the ML signal.

            ML Signal: {signal_map.get(ml_signal.get('signal', 0), 'FLAT')}
            ML Confidence: {ml_signal.get('confidence', 0):.4f}
            Ensemble Agreement: {ml_signal.get('agreement', 0):.4f}
            Long Probability: {ml_signal.get('long_prob', 0):.4f}
            Short Probability: {ml_signal.get('short_prob', 0):.4f}

            Market Data:
            - Funding Rate: {funding.get('funding_rate', 0):.6f}
            - Open Interest: {funding.get('open_interest', 0):.2f}
            - Mark Price: {funding.get('mark_price', 0):.4f}
            - Order Book Imbalance: {order_book.get('order_book_imbalance', 0):.4f}
            - Bid/Ask Spread (bps): {order_book.get('bid_ask_spread_bps', 0):.2f}
            - Trade Flow Imbalance: {snapshot.get('trade_flow_imbalance', 0):.4f}

            Respond with JSON only:
            {{
                "validated_signal": <0|1|2>,
                "confidence_adjustment": <float between -0.2 and 0.2>,
                "regime": "<trending_up|trending_down|ranging|volatile|consolidating>",
                "reasoning": "<brief explanation>",
                "risk_flags": ["<flag1>", ...]
            }}
        """).strip()

    def _build_leverage_prompt(
        self,
        symbol: str,
        ml_confidence: float,
        regime: str,
        current_leverage: int,
        perf: Dict[str, Any],
    ) -> str:
        lev_cfg = self.cfg.trading.leverage
        return textwrap.dedent(f"""
            Recommend leverage for {symbol} trade.

            Current leverage: {current_leverage}x
            Allowed range: {lev_cfg.min}x – {lev_cfg.max}x
            ML Confidence: {ml_confidence:.4f}
            Market Regime: {regime}

            Recent Performance (last 20 trades):
            - Win Rate: {perf.get('win_rate', 0):.2%}
            - Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}
            - Max Drawdown: {perf.get('max_drawdown_pct', 0):.2%}
            - Profit Factor: {perf.get('profit_factor', 0):.3f}

            Rules: Higher confidence and better performance → higher leverage.
            Volatile or uncertain regimes → lower leverage.
            Conservative bias – reduce leverage when in doubt.

            Respond with JSON only:
            {{
                "recommended_leverage": <int between {lev_cfg.min} and {lev_cfg.max}>,
                "reasoning": "<brief explanation>"
            }}
        """).strip()

    def _build_performance_review_prompt(
        self,
        recent_trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> str:
        trades_summary = json.dumps(recent_trades[-10:], default=str, indent=2)
        metrics_summary = json.dumps(metrics, default=str, indent=2)
        return textwrap.dedent(f"""
            Review trading performance and suggest adjustments.

            Metrics:
            {metrics_summary}

            Last 10 Trades:
            {trades_summary}

            Respond with JSON only:
            {{
                "adjustments": [
                    {{"parameter": "<name>", "old_value": <old>, "new_value": <new>, "reason": "<why>"}}
                ],
                "overall_assessment": "<brief assessment>",
                "pause_trading": <true|false>,
                "pause_reason": "<reason if pause_trading is true, else empty string>"
            }}
        """).strip()

    # ── Gemini call ───────────────────────────────────────────────────────────

    def _call_gemini(self, prompt: str, model: Any = None) -> Optional[str]:
        active_model = model or self._model
        # When using the secondary model, reconfigure with its API key.
        use_secondary = (
            model is not None
            and model is self._model_2
            and self._api_key_2
            and _GENAI_AVAILABLE
        )
        if use_secondary:
            genai.configure(api_key=self._api_key_2)

        def _generate() -> Optional[str]:
            gen_cfg = {
                "temperature": self.gcfg.temperature,
                "max_output_tokens": self.gcfg.max_output_tokens,
            }
            response = active_model.generate_content(
                prompt,
                generation_config=gen_cfg,
            )
            return response.text

        start = time.monotonic()
        try:
            result = _generate()
            self._record_answer_time(time.monotonic() - start)
            return result
        except Exception as exc:
            failed_model = getattr(active_model, "model_name", self.gcfg.model)
            if self._is_model_not_found_error(exc) and self._switch_to_supported_model(failed_model):
                try:
                    active_model = self._model
                    result = _generate()
                    self._record_answer_time(time.monotonic() - start)
                    return result
                except Exception as retry_exc:
                    log.error("Gemini API call failed after model switch: %s", retry_exc)
                    return None
            log.error("Gemini API call failed: %s", exc)
            return None
        finally:
            # Restore primary API key after secondary model calls.
            if use_secondary and _GENAI_AVAILABLE:
                genai.configure(api_key=self._api_key_1)

    def _record_answer_time(self, elapsed: float) -> None:
        """Append response time and keep only the last N samples."""
        self._last_answer_times.append(elapsed)
        if len(self._last_answer_times) > self._max_answer_times:
            self._last_answer_times = self._last_answer_times[-self._max_answer_times:]

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "404" in msg and "model" in msg and "not found" in msg and "generatecontent" in msg

    @staticmethod
    def _list_supported_models() -> List[str]:
        try:
            available = []
            for model in genai.list_models():
                name = getattr(model, "name", "")
                methods = {
                    method.lower()
                    for method in getattr(model, "supported_generation_methods", [])
                    if isinstance(method, str)
                }
                if name.startswith("models/gemini") and "generatecontent" in methods:
                    available.append(name.replace("models/", ""))
            return available
        except Exception as exc:
            log.warning("Failed to list Gemini models: %s", exc)
            return []

    @staticmethod
    def _select_preferred_model(
        available: List[str],
        preferred_order: tuple[str, ...],
        configured: str,
        exclude: Optional[str] = None,
    ) -> str:
        if configured and configured in available and configured != exclude:
            return configured
        for preferred in preferred_order:
            for candidate in available:
                if candidate.startswith(preferred) and candidate != exclude:
                    return candidate
        return configured

    def _switch_to_supported_model(self, failed_model: str) -> bool:
        available = self._list_supported_models()
        if not available:
            return False

        candidate = self._select_preferred_model(
            available,
            self._PRIMARY_MODEL_PREFERENCE,
            self.gcfg.model,
            exclude=failed_model,
        )
        if not candidate or candidate == failed_model or candidate not in available:
            return False

        self._model = genai.GenerativeModel(
            model_name=candidate,
            system_instruction=_SYSTEM_PROMPT,
        )
        self.gcfg.model = candidate
        log.warning("Switched Gemini model to supported fallback '%s'", candidate)
        return True

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON block from a text that may contain markdown fences."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text

    # ── Fallback heuristics ───────────────────────────────────────────────────

    def _fallback_market_analysis(self, ml_signal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "validated_signal": ml_signal.get("signal", 0),
            "confidence_adjustment": 0.0,
            "regime": "unknown",
            "reasoning": "Gemini unavailable – using raw ML signal",
            "risk_flags": [],
        }

    def _fallback_leverage(
        self, ml_confidence: float, current_leverage: int
    ) -> Dict[str, Any]:
        lev_cfg = self.cfg.trading.leverage
        if ml_confidence >= lev_cfg.high_confidence_threshold:
            new_lev = min(current_leverage + lev_cfg.step, lev_cfg.max)
        elif ml_confidence < lev_cfg.low_confidence_threshold:
            new_lev = max(current_leverage - lev_cfg.step, lev_cfg.min)
        else:
            new_lev = current_leverage
        return {
            "recommended_leverage": new_lev,
            "reasoning": "Fallback: heuristic based on ML confidence",
        }

    def _fallback_performance_review(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        pause = (
            metrics.get("max_drawdown_pct", 0) > self.cfg.evaluation.max_drawdown_pct
            or metrics.get("sharpe_ratio", 1) < 0
        )
        return {
            "adjustments": [],
            "overall_assessment": "Fallback review – Gemini unavailable",
            "pause_trading": pause,
            "pause_reason": "Max drawdown or negative Sharpe exceeded" if pause else "",
        }
