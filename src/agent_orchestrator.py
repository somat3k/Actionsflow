"""
Quantum Trading System – Agent Orchestrator
Uses Groq (oss-120b / llama-3.3-70b-versatile) to analyse market context,
validate ML signals, recommend leverage, and review performance.

Falls back to heuristic rules when the Groq API is unavailable.
"""

from __future__ import annotations

import json
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests

from src.config import AppConfig
from src.gemini_orchestrator import _SYSTEM_PROMPT, build_market_context_prompt
from src.utils import get_logger

log = get_logger(__name__)


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


class AgentOrchestrator:
    """Interfaces with Groq AI to orchestrate trading decisions.

    Uses Groq's OpenAI-compatible API (oss-120b / llama-3.3-70b-versatile) to:
      - Validate ML signals given current market context
      - Recommend appropriate leverage based on regime and confidence
      - Review recent trade performance and suggest adjustments

    Falls back to heuristic rules when the Groq API key is not configured or
    the API call fails.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.gcfg = config.groq
        self._last_answer_times: List[float] = []
        self._max_answer_times = 500
        if self.gcfg.api_key:
            log.info(
                "AgentOrchestrator initialised with Groq model '%s'", self.gcfg.model
            )
        else:
            log_fn = log.info if config.trading.mode == "test" else log.warning
            log_fn("AgentOrchestrator: no Groq API key configured – using fallback heuristics")

    @property
    def available(self) -> bool:
        """True when the Groq API key is configured."""
        return bool(self.gcfg.api_key)

    @property
    def avg_answer_time(self) -> float:
        """Average agent response time in seconds."""
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
        Ask the agent to validate the ML signal in context and return:
        {
            "validated_signal": 0|1|2,
            "confidence_adjustment": float,   # -0.2 to +0.2
            "regime": str,
            "reasoning": str,
            "risk_flags": [str],
        }
        """
        if not self.available:
            return self._fallback_market_analysis(ml_signal)

        prompt = self._build_market_context_prompt(symbol, ml_signal, market_snapshot)
        response = self._call_agent(prompt)
        if response:
            try:
                result = json.loads(_extract_json(response))
                log.info("Agent market analysis for %s: %s", symbol, result.get("regime"))
                return result
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Failed to parse Agent market analysis: %s", exc)
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
        if not self.available:
            return self._fallback_leverage(ml_confidence, current_leverage)

        prompt = self._build_leverage_prompt(
            symbol, ml_confidence, regime, current_leverage, recent_performance
        )
        response = self._call_agent(prompt)
        if response:
            try:
                result = json.loads(_extract_json(response))
                lev = int(result.get("recommended_leverage", current_leverage))
                lev = max(
                    self.cfg.trading.leverage.min,
                    min(self.cfg.trading.leverage.max, lev),
                )
                result["recommended_leverage"] = lev
                log.info("Agent leverage recommendation for %s: %dx", symbol, lev)
                return result
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("Failed to parse Agent leverage recommendation: %s", exc)
        return self._fallback_leverage(ml_confidence, current_leverage)

    def review_performance(
        self,
        recent_trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return:
        {
            "adjustments": [...],
            "overall_assessment": str,
            "pause_trading": bool,
            "pause_reason": str,
        }
        """
        if not self.available:
            return self._fallback_performance_review(metrics)

        prompt = self._build_performance_review_prompt(recent_trades, metrics)
        response = self._call_agent(prompt)
        if response:
            try:
                result = json.loads(_extract_json(response))
                log.info(
                    "Agent performance review: %s (pause=%s)",
                    result.get("overall_assessment", ""),
                    result.get("pause_trading", False),
                )
                return result
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("Failed to parse Agent performance review: %s", exc)
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
        return build_market_context_prompt(symbol, ml_signal, snapshot)

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
            Calibrate leverage to match market conditions and ML confidence.

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

    # ── Groq API call ─────────────────────────────────────────────────────────

    def _call_agent(self, prompt: str) -> Optional[str]:
        if not self.gcfg.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.gcfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.gcfg.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.gcfg.temperature,
            "max_tokens": self.gcfg.max_output_tokens,
        }
        timeout = getattr(self.gcfg, "timeout_seconds", 30)
        start = time.monotonic()
        try:
            response = requests.post(
                self.gcfg.api_url,
                json=payload,
                timeout=(5, timeout),
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not isinstance(choices, list) or not choices:
                log.warning("Agent API response missing choices")
                return None
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if not isinstance(message, dict):
                log.warning("Agent API response missing message content")
                return None
            content = message.get("content")
            self._record_answer_time(time.monotonic() - start)
            return content
        except requests.exceptions.Timeout as exc:
            log.warning("Agent API timeout after %ss: %s", timeout, exc)
        except requests.exceptions.ConnectionError as exc:
            log.warning("Agent API connection error: %s", exc)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            log.warning("Agent API HTTP error (%s): %s", status, exc)
        except requests.exceptions.RequestException as exc:
            log.warning("Agent API call failed: %s", exc)
        return None

    def _record_answer_time(self, elapsed: float) -> None:
        """Append response time and keep only the last N samples."""
        self._last_answer_times.append(elapsed)
        if len(self._last_answer_times) > self._max_answer_times:
            self._last_answer_times = self._last_answer_times[-self._max_answer_times :]

    # ── Fallback heuristics ───────────────────────────────────────────────────

    def _fallback_market_analysis(self, ml_signal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "validated_signal": ml_signal.get("signal", 0),
            "confidence_adjustment": 0.0,
            "regime": "unknown",
            "reasoning": "Agent unavailable – using raw ML signal",
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
            "overall_assessment": "Fallback review – Agent unavailable",
            "pause_trading": pause,
            "pause_reason": "Max drawdown or negative Sharpe exceeded" if pause else "",
        }
