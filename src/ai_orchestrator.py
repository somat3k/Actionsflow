"""
Multi-provider AI orchestrator for Groq (Agent), OpenRouter, and OpenAI.
AgentOrchestrator (Groq-backed) is the primary provider and also the final
fallback/heuristic path when other providers are unavailable or return no response.
"""

from __future__ import annotations

import json
import statistics
import textwrap
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import requests

from src.config import AppConfig
from src.agent_orchestrator import AgentOrchestrator
from src.gemini_orchestrator import _SYSTEM_PROMPT, build_market_context_prompt
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


class AgentProvider:
    """Wrapper to expose AgentOrchestrator as a provider."""

    name = "Agent"

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self.orchestrator = orchestrator

    @property
    def available(self) -> bool:
        return self.orchestrator.available

    def analyse_market_context(self, *args, **kwargs):
        return self.orchestrator.analyse_market_context(*args, **kwargs)

    def recommend_leverage(self, *args, **kwargs):
        return self.orchestrator.recommend_leverage(*args, **kwargs)

    def review_performance(self, *args, **kwargs):
        return self.orchestrator.review_performance(*args, **kwargs)


class MultiAIOrchestrator:
    """Orchestrates Groq (primary), Agent (Groq), OpenRouter, and OpenAI providers with fallback.

    Groq is the primary/default provider for AI inference due to its low-latency
    API.  When Groq responds successfully the result is returned immediately
    without querying other providers, keeping decision-making latency minimal.
    Other providers are queried only when Groq is unavailable or returns no
    response, in the order: Agent → OpenRouter → OpenAI.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._fallback = AgentOrchestrator(config)
        self._providers = self._build_providers(config)
        # Name of the primary fast-path provider; responses from this provider
        # are used immediately without waiting for the full provider list.
        self._primary_provider_name: str = "Groq"

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
        """Build the ordered provider list with Groq as the first (primary) entry.

        Provider priority:
          1. Groq   – primary; low-latency inference, used as fast-path
          2. Agent  – secondary; AgentOrchestrator (Groq-backed) with heuristic fallback
          3. OpenRouter – additional fallback
          4. OpenAI     – final fallback before heuristic Agent path
        """
        providers: List[Any] = []

        # Groq is the primary provider – always first for fast inference.
        groq_provider = OpenAICompatibleOrchestrator("Groq", config, config.groq)
        if groq_provider.available:
            providers.append(groq_provider)

        agent_provider = AgentProvider(self._fallback)
        if agent_provider.available:
            providers.append(agent_provider)

        openrouter_provider = OpenAICompatibleOrchestrator("OpenRouter", config, config.openrouter)
        if openrouter_provider.available:
            providers.append(openrouter_provider)

        openai_provider = OpenAICompatibleOrchestrator("OpenAI", config, config.openai)
        if openai_provider.available:
            providers.append(openai_provider)

        return providers

    def _collect(self, method: str, *args) -> List[Dict[str, Any]]:
        """Query providers sequentially with a fast-path for the primary provider.

        When the primary provider (Groq) is the first in the list and returns a
        valid response, its result is returned immediately without querying the
        remaining providers.  This minimises end-to-end inference latency for
        the execute-analyse loop.

        If the primary provider fails or returns an empty response, all remaining
        providers are tried in order and their successful responses are accumulated
        into the returned list (used by the merge functions to combine signals).

        All provider objects expose a ``name`` attribute (``OpenAICompatibleOrchestrator``
        sets it in ``__init__``; ``AgentProvider`` defines it as a class attribute).
        """
        responses: List[Dict[str, Any]] = []
        for provider in self._providers:
            result = getattr(provider, method)(*args)
            if isinstance(result, dict) and result:
                responses.append(result)
                # Fast-path: primary provider responded – return immediately.
                if provider.name == self._primary_provider_name:
                    log.debug(
                        "Fast-path: %s responded successfully; skipping remaining providers",
                        self._primary_provider_name,
                    )
                    return responses
        return responses

    def health_check(self) -> List[Dict[str, Any]]:
        """Run a real-data inference probe on every configured AI provider.

        Sends a minimal synthetic market-context request to each provider
        and records whether a valid response is returned.  Providers that
        are not configured (no API key / library unavailable) are reported
        with ``status="skipped"``; those that raise or return an empty
        response are reported with ``status="error"``.

        Returns:
            List of per-provider result dicts, each with keys:

            - ``provider``   : provider name
            - ``status``     : ``"ok"`` | ``"error"`` | ``"skipped"``
            - ``latency_ms`` : round-trip latency in milliseconds (``None`` when skipped)
            - ``error``      : error message when ``status`` is not ``"ok"``
        """
        # Minimal synthetic market data used as the health probe payload.
        _probe_signal: Dict[str, Any] = {
            "signal": 0, "confidence": 0.5, "agreement": 0.8,
            "long_prob": 0.3, "short_prob": 0.2,
        }
        _probe_snapshot: Dict[str, Any] = {
            "funding": {
                "funding_rate": 0.0001, "open_interest": 1000.0, "mark_price": 50000.0,
            },
            "order_book": {"order_book_imbalance": 0.0, "bid_ask_spread_bps": 1.0},
            "trade_flow_imbalance": 0.0,
        }

        results: List[Dict[str, Any]] = []
        active_names: set = set()

        for provider in self._providers:
            name: str = getattr(provider, "name", type(provider).__name__)
            active_names.add(name)
            t0 = time.monotonic()
            try:
                response = provider.analyse_market_context(
                    "BTC", _probe_signal, _probe_snapshot
                )
                latency_ms = (time.monotonic() - t0) * 1000
                if isinstance(response, dict) and response:
                    results.append({
                        "provider": name, "status": "ok",
                        "latency_ms": round(latency_ms, 1), "error": "",
                    })
                    log.info(
                        "AI health check OK: %s (%.0fms)", name, latency_ms
                    )
                else:
                    results.append({
                        "provider": name, "status": "error",
                        "latency_ms": round(latency_ms, 1),
                        "error": "empty or invalid response",
                    })
                    log.warning(
                        "AI health check FAILED: %s – empty response", name
                    )
            except Exception as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                results.append({
                    "provider": name, "status": "error",
                    "latency_ms": round(latency_ms, 1), "error": str(exc),
                })
                log.warning("AI health check ERROR: %s – %s", name, exc)

        # Report unconfigured providers as skipped.
        _all_provider_keys = [
            ("Groq", self.cfg.groq.api_key),
            ("OpenRouter", self.cfg.openrouter.api_key),
            ("OpenAI", self.cfg.openai.api_key),
        ]
        for name, key in _all_provider_keys:
            if name not in active_names:
                results.append({
                    "provider": name, "status": "skipped",
                    "latency_ms": None, "error": "no API key configured",
                })
        return results

    def orchestration_probe(self) -> List[Dict[str, Any]]:
        """Run a full multi-step orchestration probe to validate the complete pipeline.

        Simulates the actual orchestration flow used during live trading by running
        three sequential steps per provider with a rich, realistic payload that includes
        ML model neural-network votes and technical indicators:

          1. **Market context analysis** – ML signal + indicators → regime + validated signal
          2. **Leverage recommendation** – regime + confidence → leverage
          3. **Performance review** – synthetic recent trades + metrics → adjustments

        The payload identifies this request as a "Quantum Trading System – inference probe"
        so providers can distinguish health probes from live trading requests.

        Returns:
            List of per-provider result dicts, each containing keys:

            - ``provider``      : provider name
            - ``step``          : ``"market_context"`` | ``"leverage"`` | ``"performance"``
            - ``status``        : ``"ok"`` | ``"error"`` | ``"skipped"``
            - ``latency_ms``    : round-trip latency in milliseconds (``None`` when skipped)
            - ``response``      : parsed response dict (``None`` when not ok)
            - ``error``         : error message when status is not ok
        """
        # ── Rich probe payload with ML / NN context ────────────────────────────
        _symbol = "BTC"
        _probe_signal: Dict[str, Any] = {
            # Identity: identifies this call as a health probe from the trading system
            "request_context": "Quantum Trading System – inference probe",
            "signal": 1,           # neural-network consensus: LONG
            "confidence": 0.72,
            "agreement": 0.83,     # 83% of sub-models agree on LONG
            "long_prob": 0.72,
            "short_prob": 0.14,
            # Technical indicators used as features by the ML models
            "indicators": {
                "rsi_14": 58.4,
                "rsi_7": 61.2,
                "macd": 0.0023,
                "macd_signal": 0.0018,
                "macd_hist": 0.0005,
                "ema_9": 42_150.0,
                "ema_21": 41_800.0,
                "ema_50": 41_200.0,
                "atr_14": 420.0,
                "adx_14": 32.5,
                "bb_upper": 43_100.0,
                "bb_lower": 41_000.0,
                "bb_bandwidth": 0.05,
                "stochastic_k": 67.3,
                "stochastic_d": 62.1,
            },
            # Per-model votes from the QuantumEnsemble sub-models
            "nn_model_votes": {
                "gradient_boost": "LONG",
                "random_forest": "LONG",
                "linear": "LONG",
                "extra_trees": "LONG",
                "xgboost": "FLAT",
            },
        }
        _probe_snapshot: Dict[str, Any] = {
            "funding": {
                "funding_rate": 0.00012,
                "open_interest": 45_000.0,
                "mark_price": 42_200.0,
            },
            "order_book": {
                "order_book_imbalance": 0.12,
                "bid_ask_spread_bps": 0.8,
            },
            "trade_flow_imbalance": 0.18,
        }
        _recent_trades = [
            {"symbol": _symbol, "side": "long", "pnl": 180.0, "pnl_pct": 0.018},
            {"symbol": _symbol, "side": "long", "pnl": -60.0, "pnl_pct": -0.006},
            {"symbol": _symbol, "side": "long", "pnl": 220.0, "pnl_pct": 0.022},
            {"symbol": _symbol, "side": "short", "pnl": 95.0, "pnl_pct": 0.0095},
            {"symbol": _symbol, "side": "long", "pnl": -40.0, "pnl_pct": -0.004},
        ]
        _recent_metrics: Dict[str, Any] = {
            "win_rate": 0.60,
            "sharpe_ratio": 1.45,
            "max_drawdown_pct": 0.08,
            "profit_factor": 1.80,
            "total_trades": 5,
        }
        _current_leverage = 15

        results: List[Dict[str, Any]] = []
        all_provider_names: set = set()

        providers_to_probe: List[Any] = list(self._providers)
        # Always probe the fallback (Agent heuristic) so it appears in results
        # even when no primary providers are configured.
        if not providers_to_probe:
            providers_to_probe = [AgentProvider(self._fallback)]

        for provider in providers_to_probe:
            name: str = getattr(provider, "name", type(provider).__name__)
            all_provider_names.add(name)

            if not getattr(provider, "available", False):
                for step in ("market_context", "leverage", "performance"):
                    results.append({
                        "provider": name, "step": step,
                        "status": "skipped", "latency_ms": None,
                        "response": None, "error": "provider not available",
                    })
                continue

            # Step 1 – market context analysis
            log.info("Orchestration probe [%s] step=market_context symbol=%s", name, _symbol)
            t0 = time.monotonic()
            step1_response: Optional[Dict[str, Any]] = None
            try:
                step1_response = provider.analyse_market_context(
                    _symbol, _probe_signal, _probe_snapshot
                )
                latency_ms = (time.monotonic() - t0) * 1000
                if isinstance(step1_response, dict) and step1_response:
                    regime = step1_response.get("regime", "unknown")
                    validated_signal = step1_response.get("validated_signal", 0)
                    adj = step1_response.get("confidence_adjustment", 0.0)
                    log.info(
                        "Orchestration probe [%s] market_context OK (%.0fms): "
                        "regime=%s validated_signal=%s confidence_adj=%+.3f",
                        name, latency_ms, regime, validated_signal, adj,
                    )
                    for flag in step1_response.get("risk_flags", []):
                        log.warning("Orchestration probe [%s] risk_flag: %s", name, flag)
                    results.append({
                        "provider": name, "step": "market_context",
                        "status": "ok", "latency_ms": round(latency_ms, 1),
                        "response": step1_response, "error": "",
                    })
                else:
                    log.warning(
                        "Orchestration probe [%s] market_context EMPTY response", name
                    )
                    results.append({
                        "provider": name, "step": "market_context",
                        "status": "error", "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                        "response": None, "error": "empty or invalid response",
                    })
                    step1_response = None
            except Exception as exc:
                log.warning("Orchestration probe [%s] market_context ERROR: %s", name, exc)
                results.append({
                    "provider": name, "step": "market_context",
                    "status": "error", "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "response": None, "error": str(exc),
                })
                step1_response = None

            # Step 2 – leverage recommendation (uses regime from step 1)
            regime_from_step1 = (
                step1_response.get("regime", "ranging")
                if isinstance(step1_response, dict)
                else "ranging"
            )
            probe_confidence = float(_probe_signal.get("confidence", 0.72))
            log.info(
                "Orchestration probe [%s] step=leverage symbol=%s regime=%s",
                name, _symbol, regime_from_step1,
            )
            t0 = time.monotonic()
            try:
                step2_response = provider.recommend_leverage(
                    _symbol,
                    probe_confidence,
                    regime_from_step1,
                    _current_leverage,
                    _recent_metrics,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                if isinstance(step2_response, dict) and step2_response:
                    recommended = step2_response.get("recommended_leverage", _current_leverage)
                    log.info(
                        "Orchestration probe [%s] leverage OK (%.0fms): "
                        "recommended=%sx (current=%sx)",
                        name, latency_ms, recommended, _current_leverage,
                    )
                    results.append({
                        "provider": name, "step": "leverage",
                        "status": "ok", "latency_ms": round(latency_ms, 1),
                        "response": step2_response, "error": "",
                    })
                else:
                    log.warning(
                        "Orchestration probe [%s] leverage EMPTY response", name
                    )
                    results.append({
                        "provider": name, "step": "leverage",
                        "status": "error", "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                        "response": None, "error": "empty or invalid response",
                    })
            except Exception as exc:
                log.warning("Orchestration probe [%s] leverage ERROR: %s", name, exc)
                results.append({
                    "provider": name, "step": "leverage",
                    "status": "error", "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "response": None, "error": str(exc),
                })

            # Step 3 – performance review
            log.info("Orchestration probe [%s] step=performance symbol=%s", name, _symbol)
            t0 = time.monotonic()
            try:
                step3_response = provider.review_performance(_recent_trades, _recent_metrics)
                latency_ms = (time.monotonic() - t0) * 1000
                if isinstance(step3_response, dict) and step3_response:
                    pause = step3_response.get("pause_trading", False)
                    adj_count = len(step3_response.get("adjustments", []))
                    if pause:
                        log.warning(
                            "Orchestration probe [%s] performance: pause_trading=True reason=%s",
                            name, step3_response.get("pause_reason", ""),
                        )
                    log.info(
                        "Orchestration probe [%s] performance OK (%.0fms): "
                        "adjustments=%d pause=%s",
                        name, latency_ms, adj_count, pause,
                    )
                    results.append({
                        "provider": name, "step": "performance",
                        "status": "ok", "latency_ms": round(latency_ms, 1),
                        "response": step3_response, "error": "",
                    })
                else:
                    log.warning(
                        "Orchestration probe [%s] performance EMPTY response", name
                    )
                    results.append({
                        "provider": name, "step": "performance",
                        "status": "error", "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                        "response": None, "error": "empty or invalid response",
                    })
            except Exception as exc:
                log.warning("Orchestration probe [%s] performance ERROR: %s", name, exc)
                results.append({
                    "provider": name, "step": "performance",
                    "status": "error", "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "response": None, "error": str(exc),
                })

        return results


def _build_market_context_prompt(
    symbol: str,
    ml_signal: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> str:
    return build_market_context_prompt(symbol, ml_signal, snapshot)


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
        Calibrate leverage to match market conditions and ML confidence.

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
