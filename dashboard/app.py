from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(
    page_title="Quantum Trader",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── World-class dark trading terminal CSS ─────────────────────────────────────
st.markdown(
    """
<style>
/* ── Root palette ─────────────────────────────────────────────────────────── */
:root {
  --bg-base:       #080c14;
  --bg-card:       #0d1117;
  --bg-elevated:   #141b2d;
  --bg-hover:      #1a2235;
  --border:        rgba(0,212,255,0.12);
  --border-strong: rgba(0,212,255,0.28);
  --accent-cyan:   #00d4ff;
  --accent-purple: #7b61ff;
  --accent-green:  #00e676;
  --accent-red:    #ff5252;
  --accent-gold:   #ffd54f;
  --text-primary:  #e8eaf0;
  --text-muted:    #5a6a85;
  --text-label:    #8899aa;
  --shadow-card:   0 4px 24px rgba(0,0,0,0.6), 0 1px 0 rgba(0,212,255,0.06);
  --shadow-glow:   0 0 20px rgba(0,212,255,0.15);
  --radius:        10px;
  --radius-sm:     6px;
}

/* ── App-wide base ────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 0.875rem;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0f1e 0%, #080c14 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio > label { color: var(--text-label) !important; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] div { border-radius: var(--radius-sm); transition: background 0.15s; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] div:hover { background: var(--bg-hover); }

/* ── Streamlit main block padding ────────────────────────────────────────── */
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: none !important; }

/* ── Metric widgets ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem !important;
  box-shadow: var(--shadow-card);
  transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--border-strong); box-shadow: var(--shadow-glow); }
[data-testid="stMetric"] label { font-size: 0.65rem !important; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted) !important; font-weight: 600; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; font-variant-numeric: tabular-nums; color: var(--text-primary) !important; }
[data-testid="stMetricDelta"] { font-size: 0.7rem !important; }

/* ── Cards / panels ───────────────────────────────────────────────────────── */
.qt-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  box-shadow: var(--shadow-card);
  margin-bottom: 1rem;
}
.qt-card-header {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 6px;
}
.qt-card-header::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 14px;
  background: var(--accent-cyan);
  border-radius: 2px;
}

/* ── Section labels (frame-title) ─────────────────────────────────────────── */
.frame-title {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  margin-bottom: 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── Status pills ─────────────────────────────────────────────────────────── */
.qt-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}
.qt-pill-green  { background: rgba(0,230,118,0.12); color: var(--accent-green); border: 1px solid rgba(0,230,118,0.25); }
.qt-pill-cyan   { background: rgba(0,212,255,0.10); color: var(--accent-cyan);  border: 1px solid rgba(0,212,255,0.25); }
.qt-pill-red    { background: rgba(255,82,82,0.12); color: var(--accent-red);   border: 1px solid rgba(255,82,82,0.25); }
.qt-pill-purple { background: rgba(123,97,255,0.12); color: var(--accent-purple); border: 1px solid rgba(123,97,255,0.25); }
.qt-pill-gold   { background: rgba(255,213,79,0.12); color: var(--accent-gold); border: 1px solid rgba(255,213,79,0.25); }

/* ── System status dot ────────────────────────────────────────────────────── */
.qt-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
}
.qt-dot-live { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); animation: pulse 2s infinite; }
.qt-dot-warn { background: var(--accent-gold); box-shadow: 0 0 6px var(--accent-gold); }
.qt-dot-err  { background: var(--accent-red); }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--bg-card) !important;
  border-radius: var(--radius) var(--radius) 0 0 !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  padding: 0.6rem 1.1rem !important;
  transition: color 0.15s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: var(--text-primary) !important; background: var(--bg-hover) !important; }
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent-cyan) !important;
  border-bottom: 2px solid var(--accent-cyan) !important;
  background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
  padding: 1.25rem 1.5rem !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
[data-testid="baseButton-secondary"] {
  background: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="baseButton-secondary"]:hover {
  border-color: var(--border-strong) !important;
  box-shadow: 0 0 12px rgba(0,212,255,0.2) !important;
}
[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
  color: #080c14 !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
}

/* ── Selectbox / inputs ───────────────────────────────────────────────────── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-size: 0.8rem !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-size: 0.8rem !important;
}
[data-testid="stSlider"] { accent-color: var(--accent-cyan); }

/* ── Chat messages ────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  margin-bottom: 0.5rem !important;
}
[data-testid="stChatInputTextArea"] textarea {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-size: 0.8rem !important;
}
[data-testid="stChatInputTextArea"] textarea:focus {
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Dataframes / tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  overflow: hidden;
}
[data-testid="stDataFrame"] thead th {
  background: var(--bg-elevated) !important;
  color: var(--text-muted) !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tbody tr:hover { background: var(--bg-hover) !important; }
[data-testid="stDataFrame"] td { color: var(--text-primary) !important; font-size: 0.8rem !important; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary { color: var(--text-label) !important; font-size: 0.75rem !important; }

/* ── Alerts / info boxes ─────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-sm) !important;
  font-size: 0.8rem !important;
  border-left-width: 3px !important;
}

/* ── Form ────────────────────────────────────────────────────────────────── */
[data-testid="stForm"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1.25rem 1.5rem !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

/* ── Typography helpers ──────────────────────────────────────────────────── */
.muted { color: var(--text-muted) !important; }
.qt-mono { font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 0.85em; }
.qt-section-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.qt-section-title span.accent { color: var(--accent-cyan); }
.qt-divider {
  height: 1px;
  background: linear-gradient(90deg, var(--accent-cyan) 0%, transparent 60%);
  margin: 1rem 0;
  opacity: 0.4;
}
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
    st.markdown(
        """
        <div style="padding:0.25rem 0 1rem;">
          <div style="font-size:1.15rem;font-weight:800;letter-spacing:-0.02em;color:#e8eaf0;display:flex;align-items:center;gap:8px;">
            <span style="font-size:1.3rem;">⚡</span> Quantum Trader
          </div>
          <div style="font-size:0.62rem;color:#5a6a85;letter-spacing:0.08em;margin-top:2px;">
            ALGORITHMIC TRADING SYSTEM
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["💬 Chat & Canvas", "📈 Performance", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()
    groq_ready = bool(cfg.groq.api_key and cfg.groq.model and cfg.groq.api_url)
    mode_color = {"live": "#00e676", "paper": "#ffd54f", "test": "#00d4ff"}.get(cfg.trading.mode, "#8899aa")
    st.markdown(
        f"""
        <div style="margin-bottom:0.75rem;">
          <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#5a6a85;margin-bottom:4px;">System</div>
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
            <span class="qt-dot qt-dot-live"></span>
            <span style="color:#e8eaf0;font-size:0.75rem;font-weight:600;">{cfg.system.name}</span>
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <span class="qt-pill qt-pill-cyan" style="font-size:0.6rem;padding:2px 8px;">v{cfg.system.version}</span>
            <span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;font-size:0.6rem;font-weight:600;background:rgba(0,0,0,0.3);color:{mode_color};border:1px solid {mode_color}33;">{cfg.trading.mode.upper()}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="margin-bottom:0.75rem;">
          <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#5a6a85;margin-bottom:4px;">AI Engine</div>
          {"<span class='qt-pill qt-pill-green' style='font-size:0.6rem;padding:2px 8px;'>● Groq ready · " + (cfg.groq.model or "") + "</span>"
           if groq_ready else
           "<span class='qt-pill qt-pill-gold' style='font-size:0.6rem;padding:2px 8px;'>⚠ API key missing</span>"}
        </div>
        """,
        unsafe_allow_html=True,
    )
    markets_str = " · ".join(m.symbol for m in cfg.trading.markets[:6])
    if markets_str:
        st.markdown(
            f"""
            <div style="margin-bottom:0.75rem;">
              <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#5a6a85;margin-bottom:4px;">Markets</div>
              <div style="font-size:0.7rem;color:#8899aa;font-family:monospace;">{markets_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    if st.button("↺ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.toast("✓ Data cache cleared.", icon="✓")

# ── Headline metrics (always visible) ────────────────────────────────────────

_HEADLINE_METRICS = [
    ("Total Return", "total_return_pct", "📈"),
    ("Sharpe Ratio", "sharpe_ratio", "⚡"),
    ("Win Rate", "win_rate", "🎯"),
    ("Profit Factor", "profit_factor", "💎"),
]

_metric_raw_values = {key: metrics.get(key) for _, key, _ in _HEADLINE_METRICS}

# Build display values
_metric_displays: Dict[str, str] = {}
for _label, _key, _icon in _HEADLINE_METRICS:
    _val = _metric_raw_values[_key]
    if _key == "sharpe_ratio" and _val is not None:
        _metric_displays[_key] = f"{_val:.2f}"
    else:
        _metric_displays[_key] = _format_metric_value(_key, _val)

headline_cols = st.columns(len(_HEADLINE_METRICS), gap="small")
for col, (label, key, icon) in zip(headline_cols, _HEADLINE_METRICS):
    col.metric(label, _metric_displays[key])

# ── Plotly theme helper ───────────────────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8899aa", family="Inter,SF Pro Display,-apple-system,sans-serif", size=11),
    xaxis=dict(gridcolor="rgba(0,212,255,0.07)", zerolinecolor="rgba(0,212,255,0.12)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,212,255,0.07)", zerolinecolor="rgba(0,212,255,0.12)", tickfont=dict(size=10)),
    margin=dict(l=12, r=12, t=28, b=12),
    hoverlabel=dict(bgcolor="#141b2d", bordercolor="#00d4ff", font_color="#e8eaf0", font_size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
)


def _equity_figure(df: pd.DataFrame) -> go.Figure:
    """Styled Plotly equity curve."""
    fig = go.Figure()
    x = df["exit_time"]
    y = df["equity"]
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            name="Equity",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:,.2f}</b><extra></extra>",
        )
    )
    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text="Equity Curve", font_size=12, font_color="#8899aa", x=0))
    return fig


def _drawdown_figure(df: pd.DataFrame) -> go.Figure:
    """Styled drawdown area chart."""
    dd = df.copy()
    dd["running_max"] = dd["equity"].cummax()
    dd["drawdown_pct"] = (dd["equity"] - dd["running_max"]) / dd["running_max"].replace(0, pd.NA) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd["exit_time"], y=dd["drawdown_pct"],
            mode="lines",
            name="Drawdown %",
            line=dict(color="#ff5252", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,82,82,0.08)",
            hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.2f}%</b><extra></extra>",
        )
    )
    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text="Drawdown %", font_size=12, font_color="#8899aa", x=0))
    return fig


def _model_bar_figure(df: pd.DataFrame) -> go.Figure:
    """Styled model scores bar chart."""
    chart_df = df.set_index("symbol").dropna(axis=1, how="all")
    colors = ["#00d4ff", "#7b61ff", "#00e676", "#ffd54f", "#ff5252"]
    fig = go.Figure()
    for i, col in enumerate(chart_df.columns):
        fig.add_trace(
            go.Bar(
                name=col,
                x=chart_df.index.tolist(),
                y=chart_df[col].tolist(),
                marker_color=colors[i % len(colors)],
                marker_line_width=0,
                opacity=0.85,
                hovertemplate=f"<b>{col}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(**_PLOTLY_LAYOUT, barmode="group", title=dict(text="Model Scores by Symbol", font_size=12, font_color="#8899aa", x=0))
    return fig


def _pnl_bar_figure(series: "pd.Series[float]") -> go.Figure:
    """PnL by symbol bar chart."""
    colors = ["#00e676" if v >= 0 else "#ff5252" for v in series.values]
    fig = go.Figure(
        go.Bar(
            x=series.index.tolist(),
            y=series.values.tolist(),
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text="PnL by Symbol", font_size=12, font_color="#8899aa", x=0))
    return fig


def _candle_figure(df: pd.DataFrame, symbol: str, interval: str) -> go.Figure:
    """Plotly candlestick chart for Hyperliquid live data."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=symbol,
            increasing_line_color="#00e676", increasing_fillcolor="rgba(0,230,118,0.7)",
            decreasing_line_color="#ff5252", decreasing_fillcolor="rgba(255,82,82,0.7)",
            hovertext=[
                f"O:{o:.4f} H:{h:.4f} L:{lv:.4f} C:{c:.4f}"
                for o, h, lv, c in zip(df["open"], df["high"], df["low"], df["close"])
            ],
        )
    )
    layout = dict(**_PLOTLY_LAYOUT)
    layout["xaxis"]["rangeslider"] = {"visible": False}
    fig.update_layout(**layout, title=dict(text=f"{symbol} · {interval}", font_size=12, font_color="#8899aa", x=0))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Chat & Canvas
# ═════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat & Canvas":
    chat_col, canvas_col = st.columns([1.4, 2.6], gap="small")

    # ── Chat panel ────────────────────────────────────────────────────────────
    with chat_col:
        st.markdown('<div class="frame-title">💬 Groq Chat</div>', unsafe_allow_html=True)
        if not groq_ready:
            missing = "GROQ_API_KEY" if not cfg.groq.api_key else ("GROQ_API_URL" if not cfg.groq.api_url else "GROQ_MODEL")
            st.warning(f"⚠ Set `{missing}` to enable Groq inference — using local summaries only.")
        else:
            st.markdown(
                f'<div style="display:flex;gap:6px;align-items:center;margin-bottom:0.5rem;">'
                f'<span class="qt-pill qt-pill-green">● Live</span>'
                f'<span class="qt-pill qt-pill-cyan" style="font-family:monospace;">{cfg.groq.model}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

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
        st.markdown('<div class="frame-title">🖥 Canvas</div>', unsafe_allow_html=True)

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
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:700;color:#8899aa;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Performance Overview</div>',
                unsafe_allow_html=True,
            )
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
                st.plotly_chart(_equity_figure(equity_curve), use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(_drawdown_figure(equity_curve), use_container_width=True, config={"displayModeBar": False})
            adjustments = state["adjustments"]
            if adjustments:
                st.markdown('<div class="frame-title" style="margin-top:0.5rem;">⚙ Latest Auto-Adjustments</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(adjustments), use_container_width=True, hide_index=True)

        elif canvas_view == "Model Scores":
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:700;color:#8899aa;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Model Training Scores</div>',
                unsafe_allow_html=True,
            )
            if model_scores.empty:
                st.info("No training scores found. Run the model training workflow to populate data.")
            else:
                st.dataframe(model_scores, use_container_width=True, hide_index=True)
                st.plotly_chart(_model_bar_figure(model_scores), use_container_width=True, config={"displayModeBar": False})

        elif canvas_view == "Hyperliquid Live":
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:700;color:#8899aa;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Hyperliquid Live Market Data</div>',
                unsafe_allow_html=True,
            )
            hl_symbols = [m.symbol for m in cfg.trading.markets] or ["BTC"]
            hl_col1, hl_col2 = st.columns([2, 1])
            hl_sym = hl_col1.selectbox("Symbol", hl_symbols, key="hl_symbol_canvas")
            hl_interval = hl_col2.selectbox("Interval", ["1m", "5m", "15m", "1h"], key="hl_interval_canvas")
            hl_df = _fetch_hl_candles_cached(cfg.data.hyperliquid_api_url, hl_sym, hl_interval, 100)
            if hl_df is None or hl_df.empty:
                st.info(f"No live candle data available for {hl_sym} ({hl_interval}). Check connectivity.")
            else:
                st.plotly_chart(_candle_figure(hl_df, hl_sym, hl_interval), use_container_width=True, config={"displayModeBar": False})
                vol_fig = go.Figure(
                    go.Bar(
                        x=hl_df["time"].tolist(), y=hl_df["volume"].tolist(),
                        marker_color="rgba(0,212,255,0.35)", marker_line_width=0,
                        name="Volume",
                        hovertemplate="%{x|%H:%M}<br>Vol: %{y:,.0f}<extra></extra>",
                    )
                )
                vol_fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text="Volume", font_size=11, font_color="#8899aa", x=0), height=130)
                st.plotly_chart(vol_fig, use_container_width=True, config={"displayModeBar": False})
                st.dataframe(hl_df.tail(8), use_container_width=True, hide_index=True)

        elif canvas_view == "Positions":
            st.markdown("**Open Positions**")
            positions = state["positions_df"]
            if positions.empty:
                st.info("No open positions recorded in paper broker state.")
            else:
                st.dataframe(positions, use_container_width=True, hide_index=True)

        elif canvas_view == "Trades":
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:700;color:#8899aa;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Closed Trades</div>',
                unsafe_allow_html=True,
            )
            trades = state["trades_df"]
            if trades.empty:
                st.info("No closed trades recorded yet.")
            else:
                st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
                pnl_by_symbol = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
                st.plotly_chart(_pnl_bar_figure(pnl_by_symbol), use_container_width=True, config={"displayModeBar": False})

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
    st.markdown(
        '<div class="qt-section-title">📈 Performance <span class="accent">Analytics</span></div>',
        unsafe_allow_html=True,
    )
    tab_perf, tab_models, tab_trades, tab_inferences = st.tabs(
        ["📊 Overview", "🧠 Models", "🔁 Trades", "🤖 AI Inferences"]
    )

    with tab_perf:
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
            st.plotly_chart(_equity_figure(equity_curve), use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(_drawdown_figure(equity_curve), use_container_width=True, config={"displayModeBar": False})
        adjustments = state["adjustments"]
        if adjustments:
            st.markdown('<div class="frame-title" style="margin-top:0.5rem;">⚙ Auto-Adjustments</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(adjustments), use_container_width=True, hide_index=True)

    with tab_models:
        if model_scores.empty:
            st.info("No training scores found. Run the model training workflow to populate data.")
        else:
            st.dataframe(model_scores, use_container_width=True, hide_index=True)
            st.plotly_chart(_model_bar_figure(model_scores), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="frame-title" style="margin-top:0.75rem;">⚙ Reinforcement Snapshot</div>', unsafe_allow_html=True)
        st.dataframe(reinforcement_snapshot["table"], use_container_width=True, hide_index=True)

    with tab_trades:
        trades = state["trades_df"]
        positions = state["positions_df"]
        st.markdown('<div class="frame-title">📍 Open Positions</div>', unsafe_allow_html=True)
        if positions.empty:
            st.info("No open positions.")
        else:
            st.dataframe(positions, use_container_width=True, hide_index=True)
        st.markdown('<div class="frame-title" style="margin-top:0.75rem;">🔁 Closed Trades (last 50)</div>', unsafe_allow_html=True)
        if trades.empty:
            st.info("No closed trades recorded yet.")
        else:
            st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
            pnl_by_symbol = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
            st.plotly_chart(_pnl_bar_figure(pnl_by_symbol), use_container_width=True, config={"displayModeBar": False})

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
    st.markdown(
        '<div class="qt-section-title">⚙️ System <span class="accent">Settings</span></div>',
        unsafe_allow_html=True,
    )
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
        st.markdown('<div class="frame-title">📡 Data Sources</div>', unsafe_allow_html=True)
        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown(
                f"""
                <div class="qt-card">
                  <div class="qt-card-header">Directories</div>
                  <div style="font-size:0.75rem;color:#8899aa;line-height:1.8;">
                    <div><span style="color:#5a6a85;">state&nbsp;dir&nbsp;&nbsp;</span> <span class="qt-mono">{cfg.system.state_dir}</span></div>
                    <div><span style="color:#5a6a85;">results&nbsp;dir</span> <span class="qt-mono">{cfg.system.results_dir}</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_cols[1]:
            st.markdown(
                f"""
                <div class="qt-card">
                  <div class="qt-card-header">API Endpoints</div>
                  <div style="font-size:0.75rem;color:#8899aa;line-height:1.8;">
                    <div><span style="color:#5a6a85;">hyperliquid</span> <span class="qt-mono">{cfg.data.hyperliquid_api_url}</span></div>
                    <div><span style="color:#5a6a85;">groq model &nbsp;</span> <span class="qt-mono">{cfg.groq.model}</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        broker_updated = state["broker_state"].get("updated_at")
        report_updated = state["evaluation_report"].get("updated_at")
        training_updated = state["training_scores"].get("updated_at")
        st.markdown('<div class="frame-title" style="margin-top:0.5rem;">🕒 Last Updated (UTC)</div>', unsafe_allow_html=True)
        ts_cols = st.columns(3)
        ts_cols[0].metric("Broker State", broker_updated or "—")
        ts_cols[1].metric("Eval Report", report_updated or "—")
        ts_cols[2].metric("Training Scores", training_updated or "—")
        st.markdown('<div class="frame-title" style="margin-top:0.75rem;">⚡ Hyperliquid Snapshot</div>', unsafe_allow_html=True)
        if st.button("Fetch live snapshot", key="hl_live_snap"):
            hl_live = _fetch_hyperliquid_snapshot(cfg)
            if hl_live:
                st.json(hl_live)
            else:
                st.caption("Hyperliquid API not reachable from this environment.")
