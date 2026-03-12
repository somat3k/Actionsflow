"""
Multi-provider AI orchestrator for Gemini, OpenAI, OpenRouter, and Groq.
Uses Gemini as a fallback when other providers are unavailable.
"""

from __future__ import annotations

import json
import statistics
import textwrap
from collections import Counter
from typing import Any, Dict, List, Optional

import requests

from src.config import AppConfig
from src.gemini_orchestrator import GeminiOrchestrator, _SYSTEM_PROMPT
from src.utils import get_logger

log = get_logger(__name__)


class OpenAICompatibleOrchestrator:
    """OpenAI-compatible chat API wrapper for OpenAI/Groq/OpenRouter."""

    def __init__(self, name: str, config: AppConfig, provider_cfg: Any) -> None:
        self.name = name
        self.cfg = config
        self.api_key = provider_cfg.api_key
        self.model = provider_cfg.model
        self.api_url = provider_cfg.api_url
        self.temperature = provider_cfg.temperature
        self.max_output_tokens = provider_cfg.max_output_tokens
        self.timeout_seconds = getattr(provider_cfg, "timeout_seconds", 30)

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model and self.api_url)

    def analyse_market_context(
        self,
        symbol: str,
        ml_signal: Dict[str, Any],
        market_snapshot: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        prompt = _build_market_context_prompt(symbol, ml_signal, market_snapshot)
        response = self._call_model(prompt)
        return _parse_json_response(response)

    def recommend_leverage(
        self,
        symbol: str,
        ml_confidence: float,
        regime: str,
        current_leverage: int,
        recent_performance: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        prompt = _build_leverage_prompt(
            self.cfg,
            symbol,
            ml_confidence,
            regime,
            current_leverage,
            recent_performance,
        )
        response = self._call_model(prompt)
        result = _parse_json_response(response)
        if isinstance(result, dict) and "recommended_leverage" in result:
            try:
                result["recommended_leverage"] = int(result["recommended_leverage"])
            except (TypeError, ValueError):
                result["recommended_leverage"] = current_leverage
        return result

    def review_performance(
        self,
        recent_trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        prompt = _build_performance_review_prompt(recent_trades, metrics)
        response = self._call_model(prompt)
        return _parse_json_response(response)

    def _call_model(self, prompt: str) -> Optional[str]:
        if not self.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=(5, self.timeout_seconds),
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not isinstance(choices, list) or not choices:
                log.warning("%s API response missing choices", self.name)
                return None
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            if not isinstance(message, dict):
                log.warning("%s API response missing message content", self.name)
                return None
            return message.get("content")
        except requests.exceptions.Timeout as exc:
            log.warning("%s API timeout after %ss: %s", self.name, self.timeout_seconds, exc)
        except requests.exceptions.ConnectionError as exc:
            log.warning("%s API connection error: %s", self.name, exc)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            log.warning("%s API HTTP error (%s): %s", self.name, status, exc)
        except requests.exceptions.RequestException as exc:
            log.warning("%s API call failed: %s", self.name, exc)
        return None


class GeminiProvider:
    """Wrapper to expose Gemini as a provider when the API is available."""

    def __init__(self, orchestrator: GeminiOrchestrator) -> None:
        self.orchestrator = orchestrator

    @property
    def available(self) -> bool:
        return getattr(self.orchestrator, "_model", None) is not None

    def analyse_market_context(self, *args, **kwargs):
        return self.orchestrator.analyse_market_context(*args, **kwargs)

    def recommend_leverage(self, *args, **kwargs):
        return self.orchestrator.recommend_leverage(*args, **kwargs)

    def review_performance(self, *args, **kwargs):
        return self.orchestrator.review_performance(*args, **kwargs)


class MultiAIOrchestrator:
    """Orchestrates Gemini, OpenAI, OpenRouter, and Groq providers with fallback."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._fallback = GeminiOrchestrator(config)
        self._providers = self._build_providers(config)

    def analyse_market_context(
        self,
        symbol: str,
        ml_signal: Dict[str, Any],
        market_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        responses = self._collect(
            "analyse_market_context", symbol, ml_signal, market_snapshot
        )
        if not responses:
            return self._fallback.analyse_market_context(symbol, ml_signal, market_snapshot)
        return _merge_market_context(responses)

    def recommend_leverage(
        self,
        symbol: str,
        ml_confidence: float,
        regime: str,
        current_leverage: int,
        recent_performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        responses = self._collect(
            "recommend_leverage",
            symbol,
            ml_confidence,
            regime,
            current_leverage,
            recent_performance,
        )
        if not responses:
            return self._fallback.recommend_leverage(
                symbol, ml_confidence, regime, current_leverage, recent_performance
            )
        merged = _merge_leverage(responses, current_leverage)
        lev_cfg = self.cfg.trading.leverage
        merged["recommended_leverage"] = max(
            lev_cfg.min, min(lev_cfg.max, int(merged.get("recommended_leverage", current_leverage)))
        )
        return merged

    def review_performance(
        self,
        recent_trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        responses = self._collect("review_performance", recent_trades, metrics)
        if not responses:
            return self._fallback.review_performance(recent_trades, metrics)
        return _merge_performance(responses)

    def _build_providers(self, config: AppConfig) -> List[Any]:
        providers: List[Any] = []
        gemini_provider = GeminiProvider(self._fallback)
        if gemini_provider.available:
            providers.append(gemini_provider)

        openrouter_provider = OpenAICompatibleOrchestrator("OpenRouter", config, config.openrouter)
        if openrouter_provider.available:
            providers.append(openrouter_provider)

        openai_provider = OpenAICompatibleOrchestrator("OpenAI", config, config.openai)
        if openai_provider.available:
            providers.append(openai_provider)

        groq_provider = OpenAICompatibleOrchestrator("Groq", config, config.groq)
        if groq_provider.available:
            providers.append(groq_provider)
        return providers

    def _collect(self, method: str, *args) -> List[Dict[str, Any]]:
        responses: List[Dict[str, Any]] = []
        for provider in self._providers:
            result = getattr(provider, method)(*args)
            if isinstance(result, dict) and result:
                responses.append(result)
        return responses


def _build_market_context_prompt(
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
    cfg: AppConfig,
    symbol: str,
    ml_confidence: float,
    regime: str,
    current_leverage: int,
    perf: Dict[str, Any],
) -> str:
    lev_cfg = cfg.trading.leverage
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


def _parse_json_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    extracted = _extract_json(text)
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        log.warning("Failed to parse JSON response from provider")
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json(text: str) -> str:
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


def _merge_market_context(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    signals = [resp.get("validated_signal", 0) for resp in responses]
    signal = Counter(signals).most_common(1)[0][0] if signals else 0
    regimes = [resp.get("regime", "unknown") for resp in responses]
    regime = Counter(regimes).most_common(1)[0][0] if regimes else "unknown"
    conf_adj = [resp.get("confidence_adjustment", 0.0) for resp in responses]
    avg_conf_adj = float(sum(conf_adj) / len(conf_adj)) if conf_adj else 0.0
    risk_flags = sorted({flag for resp in responses for flag in resp.get("risk_flags", [])})
    reasoning = "; ".join(
        [resp.get("reasoning", "") for resp in responses if resp.get("reasoning")]
    )
    return {
        "validated_signal": signal,
        "confidence_adjustment": avg_conf_adj,
        "regime": regime,
        "reasoning": reasoning or "Combined AI consensus",
        "risk_flags": risk_flags,
    }


def _merge_leverage(responses: List[Dict[str, Any]], current_leverage: int) -> Dict[str, Any]:
    leverages = [
        resp.get("recommended_leverage")
        for resp in responses
        if resp.get("recommended_leverage") is not None
    ]
    # Median reduces the impact of outlier leverage recommendations.
    avg_lev = (
        round(statistics.median(map(float, leverages)))
        if leverages
        else current_leverage
    )
    reasoning = "; ".join(
        [resp.get("reasoning", "") for resp in responses if resp.get("reasoning")]
    )
    return {
        "recommended_leverage": int(avg_lev),
        "reasoning": reasoning or "Combined AI consensus",
    }


def _merge_performance(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    pause = any(resp.get("pause_trading") for resp in responses)
    pause_reason = ""
    if pause:
        pause_reason = next(
            (resp.get("pause_reason") for resp in responses if resp.get("pause_reason")), ""
        )
    adjustments: List[Dict[str, Any]] = []
    for resp in responses:
        adjustments.extend(resp.get("adjustments", []) or [])
    assessments = [resp.get("overall_assessment", "") for resp in responses if resp.get("overall_assessment")]
    overall = "; ".join(assessments) if assessments else "Combined AI consensus"
    return {
        "adjustments": adjustments,
        "overall_assessment": overall,
        "pause_trading": pause,
        "pause_reason": pause_reason,
    }
