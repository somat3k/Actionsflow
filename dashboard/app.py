from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import AppConfig, load_config
from src.dashboard_controls import ADJUSTMENT_SPECS, parse_adjustment_request, parse_trade_request
from src.dashboard_data import load_dashboard_state
from src.database_manager import DatabaseManager
from src.utils import fmt_pct, fmt_usd

CHAT_SYSTEM_PROMPT = """
You are the Quantum Trading dashboard assistant powered by Groq.
Use the provided tool results, cached dashboard memory, and the latest trading
metrics (sourced from Hyperliquid) to answer questions about model performance,
trades, and risk. Keep short questions brief and conversational; for analytical
requests provide structured, data-driven analysis with clear conclusions.
If the user asks to adjust parameters or stage trades, respond with a concise
confirmation. When visuals are requested, say which chart type suits the data.
If data is missing, say what is unavailable instead of guessing.
Fallback to rule-based decision summaries when live data is unavailable.
""".strip()


def _call_groq(
    api_key: str,
    api_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: int = 30,
) -> Optional[str]:
    """Call Groq (or any OpenAI-compatible) chat endpoint with full message history."""
    if not api_key or not model or not api_url:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        # (5, timeout): 5-second connect timeout + configurable read timeout
        resp = requests.post(api_url, json=payload, headers=headers, timeout=(5, timeout))
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            return choices[0].get("message", {}).get("content")
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    return None


def _format_metric_value(key: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if "pct" in key or "rate" in key or "ratio" in key:
        return fmt_pct(float(value))
    if "equity" in key or "profit" in key or "loss" in key or "fees" in key:
        return fmt_usd(float(value))
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _select_tool(prompt: str) -> str:
    prompt = prompt.lower()
    if any(term in prompt for term in ["reinforcement", "decision tree", "training quality"]):
        return "reinforcement"
    if any(term in prompt for term in ["override", "parameter", "adjustment", "tune"]):
        return "overrides"
    if any(term in prompt for term in ["equity", "curve", "drawdown"]):
        return "equity_curve"
    if any(term in prompt for term in ["trade", "trades", "pnl", "win", "loss"]):
        return "trades"
    if any(term in prompt for term in ["position", "open", "exposure"]):
        return "positions"
    if any(term in prompt for term in ["model", "training", "score", "xgb", "lstm"]):
        return "models"
    return "metrics"


def _tool_payload(
    tool: str,
    data: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    if tool == "equity_curve":
        df = data["equity_curve_df"].copy()
        return {
            "summary": f"Equity curve based on {len(df)} closed trades.",
            "df": df,
            "chart": "line",
        }
    if tool == "trades":
        df = data["trades_df"].copy()
        return {
            "summary": f"Loaded {len(df)} closed trades.",
            "df": df.tail(25),
            "chart": "table",
        }
    if tool == "positions":
        df = data["positions_df"].copy()
        return {
            "summary": f"Open positions count: {len(df)}.",
            "df": df,
            "chart": "table",
        }
    if tool == "models":
        df = data["model_scores_df"].copy()
        return {
            "summary": f"Model score rows: {len(df)}.",
            "df": df,
            "chart": "bar",
        }
    if tool == "reinforcement":
        reinforcement = context.get("reinforcement", {})
        table = reinforcement.get("table")
        summary = reinforcement.get("summary", "Reinforcement snapshot loaded.")
        return {
            "summary": summary,
            "df": table,
            "chart": "table",
        }
    if tool == "overrides":
        overrides_table = context.get("overrides_table")
        return {
            "summary": "Active parameter overrides loaded.",
            "df": overrides_table,
            "chart": "table",
        }
    metrics = data["metrics"]
    return {
        "summary": "Performance metrics snapshot loaded.",
        "metrics": metrics,
    }


def _build_chat_prompt(
    user_prompt: str,
    tool_result: Dict[str, Any],
    metrics: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    summary = tool_result.get("summary", "")
    payload = tool_result.get("metrics") or {}
    lines = [
        "User question:",
        user_prompt,
        "",
        f"Tool summary: {summary}",
        "",
        "Key metrics (sourced from Hyperliquid/paper broker):",
    ]
    for key in ["total_return_pct", "sharpe_ratio", "win_rate", "profit_factor", "max_drawdown_pct"]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    overrides = context.get("overrides", {})
    if overrides:
        lines.append("")
        lines.append("Active parameter overrides:")
        lines.append(str(overrides))
    reinforcement = context.get("reinforcement", {})
    if reinforcement:
        lines.append("")
        lines.append("Reinforcement snapshot:")
        lines.append(str({k: v for k, v in reinforcement.items() if k != "table"}))
    hl_data = context.get("hyperliquid_data")
    if hl_data:
        lines.append("")
        lines.append("Hyperliquid live market snapshot:")
        lines.append(str(hl_data))
    if payload:
        lines.append("\nTool payload:")
        lines.append(str(payload))
    return "\n".join(lines)


def _wants_canvas(prompt: str) -> bool:
    prompt = prompt.lower()
    return any(term in prompt for term in ["canvas", "chart", "plot", "graph", "visual"])


def _render_chat_tool_result(tool_result: Dict[str, Any]) -> None:
    df = tool_result.get("df")
    chart = tool_result.get("chart")
    if df is None or df.empty:
        return
    if chart == "line":
        st.line_chart(df.set_index("exit_time")["equity"])
    elif chart == "bar":
        chart_df = df.set_index("symbol").dropna(axis=1, how="all")
        st.bar_chart(chart_df)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_resource
def _load_db(state_dir: str, db_file: str) -> DatabaseManager:
    return DatabaseManager(Path(state_dir) / db_file)


def _config_value(cfg: AppConfig, key: str) -> Any:
    value_map = {
        "ml.long_threshold": cfg.ml.long_threshold,
        "ml.short_threshold": cfg.ml.short_threshold,
        "ml.close_threshold": cfg.ml.close_threshold,
        "ml.min_ensemble_agreement": cfg.ml.min_ensemble_agreement,
        "ml.reinforcement_alpha": cfg.ml.reinforcement_alpha,
        "ml.training_epochs": cfg.ml.training_epochs,
        "ml.retrain_interval_hours": cfg.ml.retrain_interval_hours,
        "trading.leverage.min": cfg.trading.leverage.min,
        "trading.leverage.max": cfg.trading.leverage.max,
    }
    return value_map.get(key)


def _build_overrides_table(overrides: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for spec in ADJUSTMENT_SPECS:
        if spec.key in overrides:
            rows.append(
                {
                    "parameter": spec.label,
                    "value": overrides[spec.key],
                    "key": spec.key,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["parameter", "value", "key"])
    return pd.DataFrame(rows)


def _build_reinforcement_snapshot(
    model_scores: pd.DataFrame,
    cfg: AppConfig,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    scores = model_scores.drop(columns=["symbol"], errors="ignore")
    numeric_scores = scores.apply(pd.to_numeric, errors="coerce")
    avg_score = None
    if not numeric_scores.empty:
        avg_score = float(numeric_scores.mean().mean())
    reinforcement_alpha = overrides.get("ml.reinforcement_alpha", cfg.ml.reinforcement_alpha)
    table = pd.DataFrame(
        [
            {"metric": "Avg model score", "value": avg_score},
            {"metric": "Reinforcement alpha", "value": reinforcement_alpha},
            {"metric": "Training epochs", "value": overrides.get("ml.training_epochs", cfg.ml.training_epochs)},
        ]
    )
    summary = "Reinforcement snapshot based on latest training scores."
    return {
        "avg_model_score": avg_score,
        "reinforcement_alpha": reinforcement_alpha,
        "table": table,
        "summary": summary,
    }


def _get_cache_list(db: DatabaseManager, key: str) -> List[Dict[str, Any]]:
    cached = db.get_cache(key)
    if isinstance(cached, list):
        return cached
    return []


def _append_cache_list(
    db: DatabaseManager,
    key: str,
    entry: Dict[str, Any],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    history = _get_cache_list(db, key)
    history.append(entry)
    history = history[-limit:]
    db.set_cache(key, history)
    return history


def _fetch_hyperliquid_snapshot(cfg: AppConfig, symbol: str = "BTC") -> Optional[Dict[str, Any]]:
    """Fetch a lightweight live market snapshot from Hyperliquid."""
    try:
        resp = requests.post(
            cfg.data.hyperliquid_api_url,
            json={"type": "allMids"},
            timeout=5,
        )
        resp.raise_for_status()
        mids: Dict[str, str] = resp.json()
        # Hyperliquid may return symbols with or without '-PERP' suffix
        price = mids.get(symbol) or mids.get(f"{symbol}-PERP")
        return {"symbol": symbol, "mid_price": price, "all_mids_count": len(mids)}
    except Exception:
        return None


@st.cache_data(ttl=30)
def _fetch_hl_candles_cached(api_url: str, symbol: str, interval: str, n: int) -> Optional[pd.DataFrame]:
    """Fetch recent OHLCV candles from Hyperliquid for charting."""
    try:
        from src.utils import interval_to_ms, utc_now_ms
        end_ms = utc_now_ms()
        start_ms = end_ms - n * interval_to_ms(interval)
        resp = requests.post(
            api_url,
            json={"type": "candleSnapshot", "req": {"coin": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms}},
            timeout=10,
        )
        resp.raise_for_status()
        candles = resp.json()
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── App bootstrap ─────────────────────────────────────────────────────────────

cfg = load_config()
db = _load_db(cfg.system.state_dir, cfg.system.database_file)

st.set_page_config(page_title="Quantum Trader Dashboard", layout="wide")
st.markdown(
    """
<style>
div[data-testid="stAppViewContainer"] { font-size: 0.85rem; }
div[data-testid="stAppViewContainer"] p { font-size: 0.85rem; }
.frame-title { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #6c757d; }
.muted { color: #6c757d; }
div[data-testid="stMetric"] label { font-size: 0.7rem; color: #6c757d; }
div[data-testid="stMetric"] div { font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def _load_state() -> Dict[str, Any]:
    return load_dashboard_state(cfg)


state = _load_state()
metrics = state["metrics"]
model_scores = state["model_scores_df"]
overrides = db.get_cache("dashboard:config_overrides") or {}
if "config_overrides" not in st.session_state:
    st.session_state.config_overrides = overrides
overrides = st.session_state.config_overrides
overrides_table = _build_overrides_table(overrides)
trade_requests = _get_cache_list(db, "dashboard:trade_requests")
reinforcement_snapshot = _build_reinforcement_snapshot(state["model_scores_df"], cfg, overrides)

if "canvas_payload" not in st.session_state:
    st.session_state.canvas_payload = None

# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 Quantum Trader")
    st.caption(f"v{cfg.system.version} · {cfg.trading.mode}")
    st.divider()
    page = st.radio(
        "Navigate",
        ["💬 Chat & Canvas", "📈 Performance", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()
    groq_ready = bool(cfg.groq.api_key and cfg.groq.model)
    if groq_ready:
        st.success(f"Groq ready · `{cfg.groq.model}`")
    else:
        st.warning("Groq API key missing")
    if st.button("↺ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.toast("Data cache cleared.")

# ── Headline metrics (always visible) ────────────────────────────────────────

headline_metrics = [
    ("Total Return", "total_return_pct"),
    ("Sharpe Ratio", "sharpe_ratio"),
    ("Win Rate", "win_rate"),
    ("Profit Factor", "profit_factor"),
]
headline_cols = st.columns(len(headline_metrics))
for col, (label, key) in zip(headline_cols, headline_metrics):
    value = metrics.get(key)
    if key == "sharpe_ratio" and value is not None:
        display_value = f"{value:.2f}"
    else:
        display_value = _format_metric_value(key, value)
    col.metric(label, display_value)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Chat & Canvas
# ═════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat & Canvas":
    chat_col, canvas_col = st.columns([1.4, 2.6], gap="small")

    # ── Chat panel ────────────────────────────────────────────────────────────
    with chat_col:
        st.markdown('<div class="frame-title">Groq Chat</div>', unsafe_allow_html=True)
        if not groq_ready:
            st.warning("Set GROQ_API_KEY to enable Groq inference. Responses will use local summaries only.")
        else:
            st.caption(f"Model: `{cfg.groq.model}` · Hyperliquid data enabled")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = _get_cache_list(db, "dashboard:chat_history")

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("tool_result"):
                    _render_chat_tool_result(message["tool_result"])

        user_prompt = st.chat_input("Ask about metrics, trades, models, or live Hyperliquid data…")
        if user_prompt:
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            hl_snapshot = _fetch_hyperliquid_snapshot(cfg)
            context = {
                "overrides": overrides,
                "overrides_table": overrides_table,
                "reinforcement": reinforcement_snapshot,
                "trade_requests": trade_requests,
                "hyperliquid_data": hl_snapshot,
            }

            response_text = ""
            tool_result: Dict[str, Any] = {}
            adjustment = parse_adjustment_request(user_prompt)
            trade_request = parse_trade_request(
                user_prompt,
                symbols=[m.symbol for m in cfg.trading.markets],
            )
            if adjustment:
                overrides[adjustment["key"]] = adjustment["value"]
                st.session_state.config_overrides = overrides
                db.set_cache("dashboard:config_overrides", overrides)
                _append_cache_list(
                    db,
                    "dashboard:override_history",
                    {"timestamp": _utc_timestamp(), "overrides": {adjustment["key"]: adjustment["value"]}},
                )
                overrides_table = _build_overrides_table(overrides)
                tool_result = {
                    "summary": f"Adjustment stored: {adjustment['label']} → {adjustment['value']}",
                    "df": overrides_table,
                    "chart": "table",
                }
                response_text = tool_result["summary"]
            elif trade_request:
                trade_request["requested_at"] = _utc_timestamp()
                trade_requests = _append_cache_list(db, "dashboard:trade_requests", trade_request)
                tool_result = {
                    "summary": f"Trade request staged: {trade_request['side']} {trade_request['symbol']}",
                    "df": pd.DataFrame(trade_requests[-5:]),
                    "chart": "table",
                }
                response_text = tool_result["summary"]
            else:
                tool = _select_tool(user_prompt)
                tool_result = _tool_payload(tool, state, context)
                response_text = tool_result.get("summary", "")

                if groq_ready:
                    chat_prompt = _build_chat_prompt(user_prompt, tool_result, metrics, context)
                    # Build multi-turn message history for Groq
                    groq_messages: List[Dict[str, str]] = [
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT}
                    ]
                    for msg in st.session_state.chat_messages[:-1]:  # exclude current user msg
                        if msg["role"] in ("user", "assistant"):
                            groq_messages.append({"role": msg["role"], "content": msg["content"]})
                    groq_messages.append({"role": "user", "content": chat_prompt})
                    groq_reply = _call_groq(
                        api_key=cfg.groq.api_key,
                        api_url=cfg.groq.api_url,
                        model=cfg.groq.model,
                        messages=groq_messages,
                        temperature=cfg.groq.temperature,
                        max_tokens=cfg.groq.max_output_tokens,
                        timeout=cfg.groq.timeout_seconds,
                    )
                    if groq_reply:
                        response_text = groq_reply
                    else:
                        response_text = f"[Groq unavailable — local summary] {response_text}"

            assistant_entry = {
                "role": "assistant",
                "content": response_text,
                "tool_result": tool_result,
                "timestamp": _utc_timestamp(),
            }
            st.session_state.chat_messages.append(assistant_entry)
            # Persist full chat history to DB
            _append_cache_list(
                db,
                "dashboard:chat_history",
                {"role": "user", "content": user_prompt, "timestamp": _utc_timestamp()},
                limit=200,
            )
            _append_cache_list(
                db,
                "dashboard:chat_history",
                {"role": "assistant", "content": response_text, "timestamp": _utc_timestamp()},
                limit=200,
            )
            if tool_result and _wants_canvas(user_prompt):
                canvas_df = tool_result.get("df")
                if canvas_df is not None and not getattr(canvas_df, "empty", True):
                    st.session_state.canvas_payload = tool_result
            with st.chat_message("assistant"):
                st.markdown(response_text)
                _render_chat_tool_result(tool_result)

        if st.session_state.chat_messages:
            if st.button("Clear chat history", key="clear_chat"):
                st.session_state.chat_messages = []
                db.set_cache("dashboard:chat_history", [])
                st.toast("Chat history cleared.")

    # ── Canvas panel ──────────────────────────────────────────────────────────
    with canvas_col:
        st.markdown('<div class="frame-title">Canvas</div>', unsafe_allow_html=True)

        canvas_payload = st.session_state.get("canvas_payload")
        if canvas_payload:
            st.caption("📌 Pinned visual from chat")
            _render_chat_tool_result(canvas_payload)
            if st.button("Clear pinned canvas", key="clear_canvas"):
                st.session_state.canvas_payload = None
        else:
            st.caption("Ask Groq for a chart to pin it here, or select a view below.")

        canvas_view = st.selectbox(
            "Canvas view",
            [
                "Performance",
                "Model Scores",
                "Hyperliquid Live",
                "Positions",
                "Trades",
                "AI Inference",
                "Reinforcement",
                "System Snapshot",
            ],
            key="canvas_view",
        )

        if canvas_view == "Performance":
            st.markdown("**Performance Overview**")
            metric_cols = st.columns(4)
            key_metrics = [
                ("Max Drawdown", "max_drawdown_pct"),
                ("Total Trades", "total_trades"),
                ("Final Equity", "final_equity"),
                ("Avg Trade PnL", "avg_trade_pnl_usd"),
            ]
            for idx, (label, key) in enumerate(key_metrics):
                metric_cols[idx % 4].metric(label, _format_metric_value(key, metrics.get(key)))
            equity_curve = state["equity_curve_df"]
            if equity_curve.empty:
                st.info("No closed trades yet. Equity curve will appear after the first cycle.")
            else:
                st.line_chart(equity_curve.set_index("exit_time")["equity"], height=240)
                drawdown = equity_curve.copy()
                drawdown["running_max"] = drawdown["equity"].cummax()
                drawdown["drawdown_pct"] = (
                    drawdown["equity"] - drawdown["running_max"]
                ) / drawdown["running_max"].replace(0, pd.NA)
                st.area_chart(drawdown.set_index("exit_time")["drawdown_pct"], height=140)
            adjustments = state["adjustments"]
            if adjustments:
                st.markdown("**Latest Auto-Adjustments**")
                st.dataframe(pd.DataFrame(adjustments), use_container_width=True, hide_index=True)

        elif canvas_view == "Model Scores":
            st.markdown("**Model Training Scores**")
            if model_scores.empty:
                st.info("No training scores found. Run the model training workflow to populate data.")
            else:
                st.dataframe(model_scores, use_container_width=True, hide_index=True)
                chart_df = model_scores.set_index("symbol").dropna(axis=1, how="all")
                st.bar_chart(chart_df, height=220)

        elif canvas_view == "Hyperliquid Live":
            st.markdown("**Hyperliquid Live Market Data**")
            hl_symbols = [m.symbol for m in cfg.trading.markets] or ["BTC"]
            hl_sym = st.selectbox("Symbol", hl_symbols, key="hl_symbol_canvas")
            hl_interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h"], key="hl_interval_canvas")
            hl_df = _fetch_hl_candles_cached(cfg.data.hyperliquid_api_url, hl_sym, hl_interval, 100)
            if hl_df is None or hl_df.empty:
                st.info(f"No live candle data available for {hl_sym} ({hl_interval}). Check connectivity.")
            else:
                st.line_chart(hl_df.set_index("time")["close"], height=240)
                vol_df = hl_df.set_index("time")[["volume"]]
                st.bar_chart(vol_df, height=120)
                st.dataframe(hl_df.tail(10), use_container_width=True, hide_index=True)

        elif canvas_view == "Positions":
            st.markdown("**Open Positions**")
            positions = state["positions_df"]
            if positions.empty:
                st.info("No open positions recorded in paper broker state.")
            else:
                st.dataframe(positions, use_container_width=True, hide_index=True)

        elif canvas_view == "Trades":
            st.markdown("**Closed Trades**")
            trades = state["trades_df"]
            if trades.empty:
                st.info("No closed trades recorded yet.")
            else:
                st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
                pnl_by_symbol = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
                st.bar_chart(pnl_by_symbol, height=180)

        elif canvas_view == "AI Inference":
            st.markdown("**AI Inference — Groq Analysis**")
            last_ai = _get_cache_list(db, "dashboard:chat_history")
            ai_entries = [m for m in last_ai if m.get("role") == "assistant"]
            if not ai_entries:
                st.info("No AI inference history yet. Ask Groq a question in the Chat panel.")
            else:
                latest = ai_entries[-1]
                st.caption(f"Latest inference · {latest.get('timestamp', '')}")
                st.markdown(latest["content"])
                st.divider()
                st.markdown(f"**Total inference messages:** {len(ai_entries)}")
                st.dataframe(
                    pd.DataFrame([{"timestamp": m.get("timestamp"), "preview": (m.get("content") or "")[:120]} for m in ai_entries[-10:]]),
                    use_container_width=True,
                    hide_index=True,
                )

        elif canvas_view == "Reinforcement":
            st.markdown("**Reinforcement Snapshot**")
            st.dataframe(reinforcement_snapshot["table"], use_container_width=True, hide_index=True)
            st.caption("Use chat to adjust reinforcement alpha or training epochs.")

        else:
            st.markdown("**System Snapshot**")
            st.write(
                {
                    "system_name": cfg.system.name,
                    "version": cfg.system.version,
                    "trading_mode": cfg.trading.mode,
                    "markets": [m.symbol for m in cfg.trading.markets],
                    "auto_adjust_enabled": cfg.evaluation.auto_adjust_enabled,
                    "groq_model": cfg.groq.model,
                    "hyperliquid_api": cfg.data.hyperliquid_api_url,
                }
            )
            st.markdown("**Raw Metrics**")
            st.json(metrics)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.markdown("## Performance")
    tab_perf, tab_models, tab_trades, tab_inferences = st.tabs(
        ["📊 Overview", "🧠 Models", "🔁 Trades", "🤖 AI Inferences"]
    )

    with tab_perf:
        st.markdown("**Key Metrics**")
        perf_cols = st.columns(4)
        all_metrics = [
            ("Max Drawdown", "max_drawdown_pct"),
            ("Total Trades", "total_trades"),
            ("Final Equity", "final_equity"),
            ("Avg Trade PnL", "avg_trade_pnl_usd"),
        ]
        for idx, (label, key) in enumerate(all_metrics):
            perf_cols[idx % 4].metric(label, _format_metric_value(key, metrics.get(key)))
        equity_curve = state["equity_curve_df"]
        if equity_curve.empty:
            st.info("No closed trades yet.")
        else:
            st.markdown("**Equity Curve**")
            st.line_chart(equity_curve.set_index("exit_time")["equity"], height=300)
            drawdown = equity_curve.copy()
            drawdown["running_max"] = drawdown["equity"].cummax()
            drawdown["drawdown_pct"] = (
                drawdown["equity"] - drawdown["running_max"]
            ) / drawdown["running_max"].replace(0, pd.NA)
            st.markdown("**Drawdown %**")
            st.area_chart(drawdown.set_index("exit_time")["drawdown_pct"], height=150)
        adjustments = state["adjustments"]
        if adjustments:
            st.markdown("**Auto-Adjustments**")
            st.dataframe(pd.DataFrame(adjustments), use_container_width=True, hide_index=True)

    with tab_models:
        st.markdown("**Model Training Scores**")
        if model_scores.empty:
            st.info("No training scores found. Run the model training workflow to populate data.")
        else:
            st.dataframe(model_scores, use_container_width=True, hide_index=True)
            chart_df = model_scores.set_index("symbol").dropna(axis=1, how="all")
            st.bar_chart(chart_df, height=280)
        st.markdown("**Reinforcement Snapshot**")
        st.dataframe(reinforcement_snapshot["table"], use_container_width=True, hide_index=True)

    with tab_trades:
        trades = state["trades_df"]
        positions = state["positions_df"]
        st.markdown("**Open Positions**")
        if positions.empty:
            st.info("No open positions.")
        else:
            st.dataframe(positions, use_container_width=True, hide_index=True)
        st.markdown("**Closed Trades (last 50)**")
        if trades.empty:
            st.info("No closed trades recorded yet.")
        else:
            st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
            st.markdown("**PnL by Symbol**")
            pnl_by_symbol = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
            st.bar_chart(pnl_by_symbol, height=200)

    with tab_inferences:
        st.markdown("**Groq AI Inference History**")
        ai_history = _get_cache_list(db, "dashboard:chat_history")
        assistant_msgs = [m for m in ai_history if m.get("role") == "assistant"]
        if not assistant_msgs:
            st.info("No AI inferences yet. Use the Chat panel to interact with Groq.")
        else:
            st.metric("Total Inferences", len(assistant_msgs))
            for msg in reversed(assistant_msgs[-20:]):
                ts = msg.get("timestamp", "")
                with st.expander(f"🤖 {ts[:19] if len(ts) >= 19 else ts or 'unknown time'}"):
                    st.markdown(msg.get("content") or "")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("## Settings")
    tab_params, tab_model_actions, tab_data = st.tabs(
        ["🎚️ Parameters", "🔧 Model Actions", "📡 Data Sources"]
    )

    with tab_params:
        st.markdown("**Parameter Adjustments**")
        with st.form("parameter_controls"):
            param_cols = st.columns(2)
            updated_values: Dict[str, Any] = {}
            for idx, spec in enumerate(ADJUSTMENT_SPECS):
                col = param_cols[idx % 2]
                current = overrides.get(spec.key, _config_value(cfg, spec.key))
                with col:
                    if spec.value_type == "ratio":
                        updated_values[spec.key] = st.slider(
                            spec.label,
                            min_value=float(spec.min_value or 0.0),
                            max_value=float(spec.max_value or 1.0),
                            value=float(current),
                            step=0.01,
                        )
                    else:
                        updated_values[spec.key] = int(
                            st.number_input(
                                spec.label,
                                min_value=int(spec.min_value or 0),
                                max_value=int(spec.max_value or 999),
                                value=int(current),
                                step=1,
                            )
                        )
            save = st.form_submit_button("Save adjustments")
        if save:
            overrides.update(updated_values)
            st.session_state.config_overrides = overrides
            db.set_cache("dashboard:config_overrides", overrides)
            _append_cache_list(
                db,
                "dashboard:override_history",
                {"timestamp": _utc_timestamp(), "overrides": updated_values},
            )
            st.toast("Adjustments stored in dashboard memory.")
            overrides_table = _build_overrides_table(overrides)

        st.markdown("**Active Overrides**")
        if overrides_table.empty:
            st.caption("No overrides stored yet.")
        else:
            st.dataframe(overrides_table, use_container_width=True, hide_index=True)

    with tab_model_actions:
        st.markdown("**Model Actions**")
        model_reload = db.get_cache("dashboard:model_reload_requested") or {}
        if model_reload:
            st.caption(f"Last reload request: {model_reload.get('requested_at')}")
        if st.button("Reload ML/Groq caches"):
            st.cache_resource.clear()
            st.cache_data.clear()
            db.set_cache(
                "dashboard:model_reload_requested",
                {"requested_at": _utc_timestamp()},
            )
            st.toast("Reload request recorded.")
        if not model_scores.empty:
            csv_data = model_scores.to_csv(index=False)
            st.download_button(
                "Export model scores (CSV)",
                data=csv_data,
                file_name="model_scores_export.csv",
                mime="text/csv",
            )
        st.markdown("**Staged Trade Requests**")
        if trade_requests:
            st.dataframe(pd.DataFrame(trade_requests[-5:]), use_container_width=True, hide_index=True)
        else:
            st.caption("No trade requests stored from chat.")

    with tab_data:
        st.markdown("**Data Sources**")
        st.markdown(f"- **State dir:** `{cfg.system.state_dir}`")
        st.markdown(f"- **Results dir:** `{cfg.system.results_dir}`")
        st.markdown(f"- **Hyperliquid API:** `{cfg.data.hyperliquid_api_url}`")
        st.markdown(f"- **Groq model:** `{cfg.groq.model}`")
        broker_updated = state["broker_state"].get("updated_at")
        report_updated = state["evaluation_report"].get("updated_at")
        training_updated = state["training_scores"].get("updated_at")
        st.markdown("**Last Updated (UTC)**")
        st.write(
            {
                "broker_state": broker_updated,
                "evaluation_report": report_updated,
                "training_scores": training_updated,
            }
        )
        st.markdown("**Hyperliquid Live Snapshot**")
        hl_live = _fetch_hyperliquid_snapshot(cfg)
        if hl_live:
            st.json(hl_live)
        else:
            st.caption("Hyperliquid API not reachable from this environment.")
