from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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
You are the Quantum Trading dashboard assistant. Use the provided tool results,
cached dashboard memory, and the latest trading metrics to answer questions about
model performance, trades, and risk. If the user asks to adjust parameters or
stage trades, respond with a concise confirmation. When visuals are requested,
highlight charts for the Gemini Canvas. If data is missing, say what is
unavailable instead of guessing.
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
        "Key metrics:",
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


def _gemini_profiles(cfg: AppConfig) -> Dict[str, Dict[str, Any]]:
    profiles = {
        "Primary (flash)": {"api_key": cfg.gemini.api_key, "model": cfg.gemini.model},
    }
    if cfg.gemini.api_key_2:
        profiles["Secondary (pro)"] = {
            "api_key": cfg.gemini.api_key_2,
            "model": cfg.gemini.model_2 or cfg.gemini.model,
        }
    if len(profiles) > 1:
        profiles["Auto (primary → secondary)"] = {"auto": True}
    return profiles


def _load_gemini_profile(
    profiles: Dict[str, Dict[str, Any]],
    selected: str,
) -> tuple[Optional[Any], str]:
    profile = profiles.get(selected)
    if profile is None:
        selected = next(iter(profiles.keys()))
        profile = profiles[selected]
    if profile.get("auto"):
        for name, candidate in profiles.items():
            if candidate.get("auto"):
                continue
            model = _load_gemini_model(candidate.get("api_key", ""), candidate.get("model", ""))
            if model is not None:
                return model, name
        return None, "Unavailable"
    return _load_gemini_model(profile.get("api_key", ""), profile.get("model", "")), selected


cfg = load_config()
db = _load_db(cfg.system.state_dir, cfg.system.database_file)

st.set_page_config(page_title="Quantum Trader Dashboard", layout="wide")
st.markdown(
    """
<style>
html, body, [class*="css"] { font-size: 0.85rem; }
.frame-title { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #6c757d; }
.muted { color: #6c757d; }
div[data-testid="stMetric"] label { font-size: 0.7rem; color: #6c757d; }
div[data-testid="stMetric"] div { font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## Quantum Trader Command Deck")
st.caption("Compact single-page control frame, Gemini Canvas, and chat panel.")


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

headline_metrics = [
    ("Total Return", "total_return_pct"),
    ("Sharpe Ratio", "sharpe_ratio"),
    ("Win Rate", "win_rate"),
    ("Profit Factor", "profit_factor"),
]
headline_cols = st.columns(len(headline_metrics))
for col, (label, key) in zip(headline_cols, headline_metrics):
    col.metric(label, _format_metric_value(key, metrics.get(key)))

profiles = _gemini_profiles(cfg)
control_col, canvas_col, chat_col = st.columns([1.2, 2.4, 1.4], gap="small")

with control_col:
    st.markdown('<div class="frame-title">Control Frame</div>', unsafe_allow_html=True)
    if st.button("Refresh data cache"):
        st.cache_data.clear()
        st.toast("Data cache cleared.")

    st.markdown("**Data Sources**")
    st.markdown(f"- State: `{cfg.system.state_dir}`")
    st.markdown(f"- Results: `{cfg.system.results_dir}`")

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

    selected_profile = st.selectbox(
        "Gemini profile",
        list(profiles.keys()),
        key="gemini_profile",
    )
    model_reload = db.get_cache("dashboard:model_reload_requested") or {}
    if model_reload:
        st.caption(f"Last reload request: {model_reload.get('requested_at')}")

    with st.form("parameter_controls"):
        st.markdown("**Parameter Adjustments**")
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
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overrides": updated_values,
            },
        )
        st.toast("Adjustments stored in dashboard memory.")
        overrides_table = _build_overrides_table(overrides)

    st.markdown("**Active Overrides**")
    if overrides_table.empty:
        st.caption("No overrides stored yet.")
    else:
        st.dataframe(overrides_table, use_container_width=True, hide_index=True)

    st.markdown("**Model Actions**")
    if st.button("Reload Gemini/ML caches"):
        st.cache_resource.clear()
        st.cache_data.clear()
        db.set_cache(
            "dashboard:model_reload_requested",
            {"requested_at": datetime.now(timezone.utc).isoformat()},
        )
        st.toast("Reload request recorded.")
    if not model_scores.empty:
        csv_data = model_scores.to_csv(index=False)
        st.download_button(
            "Re-export model scores",
            data=csv_data,
            file_name="model_scores_export.csv",
            mime="text/csv",
        )

    st.markdown("**Trade Requests (DB memory)**")
    if trade_requests:
        st.dataframe(pd.DataFrame(trade_requests[-5:]), use_container_width=True, hide_index=True)
    else:
        st.caption("No trade requests stored from chat.")

with canvas_col:
    st.markdown('<div class="frame-title">Gemini Canvas</div>', unsafe_allow_html=True)
    canvas_payload = st.session_state.get("canvas_payload")
    if canvas_payload:
        st.caption("Pinned visual from chat")
        _render_chat_tool_result(canvas_payload)
        if st.button("Clear canvas", key="clear_canvas"):
            st.session_state.canvas_payload = None
    else:
        st.caption("Ask Gemini for a chart to pin it here.")

    canvas_view = st.selectbox(
        "Canvas view",
        [
            "Performance",
            "Model Scores",
            "Positions",
            "Trades",
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
                "gemini_model": cfg.gemini.model,
            }
        )
        st.markdown("**Raw Metrics**")
        st.json(metrics)

with chat_col:
    st.markdown('<div class="frame-title">Gemini Chat</div>', unsafe_allow_html=True)
    selected_profile = st.session_state.get("gemini_profile", list(profiles.keys())[0])
    model, active_profile = _load_gemini_profile(profiles, selected_profile)
    if model is None:
        st.warning("Gemini API key or library missing. Responses will use local summaries only.")
    else:
        st.caption(f"Active profile: {active_profile}")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_result"):
                _render_chat_tool_result(message["tool_result"])

    user_prompt = st.chat_input("Ask about metrics, trades, models, or adjustments")
    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        context = {
            "overrides": overrides,
            "overrides_table": overrides_table,
            "reinforcement": reinforcement_snapshot,
            "trade_requests": trade_requests,
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
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "overrides": {adjustment["key"]: adjustment["value"]},
                },
            )
            overrides_table = _build_overrides_table(overrides)
            tool_result = {
                "summary": f"Adjustment stored: {adjustment['label']} → {adjustment['value']}",
                "df": overrides_table,
                "chart": "table",
            }
            response_text = tool_result["summary"]
        elif trade_request:
            trade_request["requested_at"] = datetime.now(timezone.utc).isoformat()
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
            if model is not None:
                prompt = _build_chat_prompt(user_prompt, tool_result, metrics, context)
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
        if tool_result and _wants_canvas(user_prompt):
            st.session_state.canvas_payload = tool_result
        with st.chat_message("assistant"):
            st.markdown(response_text)
            _render_chat_tool_result(tool_result)
