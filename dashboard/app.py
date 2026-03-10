from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import load_config
from src.dashboard_data import load_dashboard_state
from src.utils import fmt_pct, fmt_usd

CHAT_SYSTEM_PROMPT = """
You are the Quantum Trading dashboard assistant. Use the provided tool results and
latest trading metrics to answer questions about model performance, trades, and
risk. If data is missing, say what is unavailable instead of guessing.
""".strip()


@st.cache_resource
def _load_gemini_model(api_key: str, model_name: str) -> Optional[Any]:
    try:
        import google.generativeai as genai
    except ImportError:
        return None
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=model_name, system_instruction=CHAT_SYSTEM_PROMPT)


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
    if any(term in prompt for term in ["equity", "curve", "drawdown"]):
        return "equity_curve"
    if any(term in prompt for term in ["trade", "trades", "pnl", "win", "loss"]):
        return "trades"
    if any(term in prompt for term in ["position", "open", "exposure"]):
        return "positions"
    if any(term in prompt for term in ["model", "training", "score", "xgb", "lstm"]):
        return "models"
    return "metrics"


def _tool_payload(tool: str, data: Dict[str, Any]) -> Dict[str, Any]:
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
    metrics = data["metrics"]
    return {
        "summary": "Performance metrics snapshot loaded.",
        "metrics": metrics,
    }


def _build_chat_prompt(user_prompt: str, tool_result: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    summary = tool_result.get("summary", "")
    payload = tool_result.get("metrics") or {}
    lines = [
        "User question:",
        user_prompt,
        "",
        f"Tool summary: {summary}",
        "",
        "Key metrics:",
    ]
    for key in ["total_return_pct", "sharpe_ratio", "win_rate", "profit_factor", "max_drawdown_pct"]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    if payload:
        lines.append("\nTool payload:")
        lines.append(str(payload))
    return "\n".join(lines)


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


cfg = load_config()

st.set_page_config(page_title="Quantum Trader Dashboard", layout="wide")
st.title("Quantum Trader Dashboard")

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

@st.cache_data(ttl=60)
def _load_state() -> Dict[str, Any]:
    return load_dashboard_state(cfg)

state = _load_state()
metrics = state["metrics"]

st.sidebar.markdown("### Data Sources")
st.sidebar.markdown(f"State directory: `{cfg.system.state_dir}`")
st.sidebar.markdown(f"Results directory: `{cfg.system.results_dir}`")

broker_updated = state["broker_state"].get("updated_at")
report_updated = state["evaluation_report"].get("updated_at")
training_updated = state["training_scores"].get("updated_at")

st.sidebar.markdown("### Last Updated (UTC)")
st.sidebar.write({
    "broker_state": broker_updated,
    "evaluation_report": report_updated,
    "training_scores": training_updated,
})

perf_tab, model_tab, position_tab, trade_tab, chat_tab, system_tab = st.tabs(
    [
        "📈 Performance",
        "🤖 Models",
        "🎯 Positions",
        "📊 Trade History",
        "💬 Gemini Chat",
        "🛠 System",
    ]
)

with perf_tab:
    st.subheader("Performance Overview")
    columns = st.columns(4)
    key_metrics = [
        ("Total Return", "total_return_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Win Rate", "win_rate"),
        ("Profit Factor", "profit_factor"),
        ("Max Drawdown", "max_drawdown_pct"),
        ("Total Trades", "total_trades"),
        ("Final Equity", "final_equity"),
        ("Avg Trade PnL", "avg_trade_pnl_usd"),
    ]
    for idx, (label, key) in enumerate(key_metrics):
        columns[idx % 4].metric(label, _format_metric_value(key, metrics.get(key)))

    equity_curve = state["equity_curve_df"]
    if equity_curve.empty:
        st.info("No closed trades yet. Equity curve will appear after the first cycle.")
    else:
        st.line_chart(equity_curve.set_index("exit_time")["equity"], height=280)
        drawdown = equity_curve.copy()
        drawdown["running_max"] = drawdown["equity"].cummax()
        drawdown["drawdown_pct"] = (
            drawdown["equity"] - drawdown["running_max"]
        ) / drawdown["running_max"].replace(0, pd.NA)
        st.area_chart(drawdown.set_index("exit_time")["drawdown_pct"], height=160)

    adjustments = state["adjustments"]
    if adjustments:
        st.markdown("### Latest Auto-Adjustments")
        st.dataframe(pd.DataFrame(adjustments), use_container_width=True, hide_index=True)

with model_tab:
    st.subheader("Model Training Scores")
    model_scores = state["model_scores_df"]
    if model_scores.empty:
        st.info("No training scores found. Run the model training workflow to populate data.")
    else:
        st.dataframe(model_scores, use_container_width=True, hide_index=True)
        chart_df = model_scores.set_index("symbol").dropna(axis=1, how="all")
        st.bar_chart(chart_df, height=240)

with position_tab:
    st.subheader("Open Positions")
    positions = state["positions_df"]
    if positions.empty:
        st.info("No open positions recorded in paper broker state.")
    else:
        st.dataframe(positions, use_container_width=True, hide_index=True)

with trade_tab:
    st.subheader("Closed Trades")
    trades = state["trades_df"]
    if trades.empty:
        st.info("No closed trades recorded yet.")
    else:
        st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
        pnl_by_symbol = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        st.bar_chart(pnl_by_symbol, height=200)

with chat_tab:
    st.subheader("Gemini Chat with Tools")
    model = _load_gemini_model(cfg.gemini.api_key, cfg.gemini.model)
    if model is None:
        st.warning("Gemini API key or library missing. Responses will use local summaries only.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_result"):
                _render_chat_tool_result(message["tool_result"])

    user_prompt = st.chat_input("Ask about metrics, trades, or model scores")
    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        tool = _select_tool(user_prompt)
        tool_result = _tool_payload(tool, state)

        response_text = tool_result.get("summary", "")
        if model is not None:
            prompt = _build_chat_prompt(user_prompt, tool_result, metrics)
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": cfg.gemini.temperature,
                        "max_output_tokens": cfg.gemini.max_output_tokens,
                    },
                )
                if response and response.text:
                    response_text = response.text
            except Exception as exc:
                response_text = f"Gemini request failed: {exc}.\n\n{response_text}"

        assistant_entry = {
            "role": "assistant",
            "content": response_text,
            "tool_result": tool_result,
        }
        st.session_state.chat_messages.append(assistant_entry)
        with st.chat_message("assistant"):
            st.markdown(response_text)
            _render_chat_tool_result(tool_result)

with system_tab:
    st.subheader("System Snapshot")
    st.write(
        {
            "system_name": cfg.system.name,
            "version": cfg.system.version,
            "trading_mode": cfg.trading.mode,
            "markets": [m.symbol for m in cfg.trading.markets],
            "auto_adjust_enabled": cfg.evaluation.auto_adjust_enabled,
            "gemini_model": cfg.gemini.model,
        }
    )
    st.markdown("### Raw Metrics")
    st.json(metrics)

