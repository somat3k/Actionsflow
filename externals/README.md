# Externals – MetaTrader & cTrader Bridges

This folder contains **standalone templates** for integrating external trading platforms with the
Quantum Trading System’s **ML ensemble** and **Gemini orchestrator**. The integrations are isolated
from the core repository and communicate with a local Python bridge via HTTP.

## Shared JSON Contract

All three platforms use the same request/response schema so that signals match across MQL4,
MQL5, and cTrader cBot (cAlgo).

### Request

```json
{
  "platform": "mql4|mql5|ctrader",
  "symbol": "EURUSD",
  "timeframe": "M1",
  "candles": [
    {
      "time": "2026-03-11T06:00:00Z",
      "open": 1.0834,
      "high": 1.0839,
      "low": 1.0832,
      "close": 1.0837,
      "volume": 1452
    }
  ],
  "account": {
    "equity": 10000,
    "balance": 10000,
    "leverage": 30
  },
  "initial_equity": 10000,
  "positions": {
    "long": 0,
    "short": 0
  },
  "market_snapshot": {
    "funding": {
      "mark_price": 1.0837,
      "funding_rate": 0.0001,
      "open_interest": 1200000
    },
    "order_book": {
      "order_book_imbalance": 0.12,
      "bid_ask_spread_bps": 1.5
    },
    "trade_flow_imbalance": -0.08
  },
  "recent_trades": [
    {
      "pnl": 42.5,
      "fee_usd": 0.7,
      "leverage": 20,
      "duration_ms": 3600000,
      "entry_time_ms": 1710146400000,
      "exit_time_ms": 1710150000000
    }
  }
}
```

### Response

```json
{
  "status": "ok",
  "symbol": "EURUSD",
  "signal": 1,
  "confidence": 0.74,
  "regime": "trending_up",
  "model": {
    "signal": 1,
    "confidence": 0.70
  },
  "gemini": {
    "validated_signal": 1,
    "confidence_adjustment": 0.04,
    "regime": "trending_up",
    "reasoning": "Momentum aligned with model signal.",
    "risk_flags": []
  },
  "leverage": {
    "current": 30,
    "recommended": 28,
    "final": 29,
    "reasoning": "Blended ML/Gemini leverage recommendation."
  },
  "position": {
    "symbol": "EURUSD",
    "side": "long",
    "entry_price": 1.0837,
    "size_usd": 1200.0,
    "size_contracts": 1107.5,
    "leverage": 29,
    "stop_loss": 1.0812,
    "take_profit": 1.0869,
    "trailing_stop_pct": 0.015,
    "risk_usd": 27.6,
    "allowed": true,
    "reject_reason": ""
  },
  "metrics": {
    "total_return_pct": 0.012,
    "annualized_return_pct": 0.18,
    "sharpe_ratio": 1.12,
    "sortino_ratio": 1.45,
    "calmar_ratio": 0.95,
    "max_drawdown_pct": 0.08,
    "win_rate": 0.55,
    "profit_factor": 1.24,
    "avg_trade_pnl_usd": 12.4,
    "avg_win_usd": 38.1,
    "avg_loss_usd": 21.3,
    "avg_trade_duration_hours": 1.4,
    "trades_per_day": 3.2,
    "total_trades": 28,
    "winning_trades": 15,
    "losing_trades": 13,
    "avg_leverage_used": 18.5,
    "avg_confidence": 0.68,
    "gross_profit": 571.4,
    "gross_loss": 460.2,
    "total_fees": 12.3,
    "final_equity": 10000,
    "initial_equity": 9880
  }
}
```

Signal mapping:

- `0` = FLAT / HOLD
- `1` = LONG
- `2` = SHORT

Notes:
- Candle timestamps should be ISO-8601 strings. The MT4/MT5 templates format broker server time
  as ISO-8601 with a trailing `Z`.
- `market_snapshot` and `recent_trades` are optional. When omitted, the bridge defaults to
  zero microstructure values and empty performance metrics.
- Each entry in `recent_trades` should include a `pnl` field to be counted in metrics.
- `positions.long` / `positions.short` are counts of open positions for the symbol.
- `initial_equity` is optional and lets you override the performance baseline used for metrics.
- Set `reset_daily: true` to reset the risk manager daily tracking when you start a new session.
- `position` is `null` when the final signal is `0` or the model is unavailable. It uses the same
  RiskManager logic as the core project.

## Python Bridge (shared)

Start the bridge server (uses the repository’s ML models and Gemini orchestrator):

```bash
python externals/python/gemini_bridge_server.py
```

Environment variables:

- `TRADING_CONFIG_PATH` (optional): path to `config/trading_config.yaml`
- `BRIDGE_HOST` (default `127.0.0.1`)
- `BRIDGE_PORT` (default `8001`)
- `BRIDGE_TOKEN` (optional): require `X-Bridge-Token` header for `/signal` requests
- `BRIDGE_MAX_BODY_BYTES` (default `1000000`): max request body size in bytes

> Binding the bridge to a non-loopback interface requires `BRIDGE_TOKEN` to be set.

> Ensure models are trained and saved before running the bridge:
>
> ```bash
> python -m src.main --run-type train-models --mode paper
> ```

## MetaTrader 4 (MQL4)

- Add `http://127.0.0.1:8001` to **Tools → Options → Expert Advisors → Allow WebRequest**.
- Compile `externals/mql4/QuantumGeminiBridge.mq4` and attach it to a chart.
- If `BRIDGE_TOKEN` is set, configure the EA’s `BridgeToken` input to match.

## MetaTrader 5 (MQL5)

- Add `http://127.0.0.1:8001` to **Tools → Options → Expert Advisors → Allow WebRequest**.
- Compile `externals/mql5/QuantumGeminiBridge.mq5` and attach it to a chart.
- If `BRIDGE_TOKEN` is set, configure the EA’s `BridgeToken` input to match.

## cTrader cBot (cAlgo)

- Create a new cBot and replace its content with `externals/ctrader/QuantumGeminiBridge.cs`.
- The cBot requires `AccessRights.Internet` to call the local bridge.
- If `BRIDGE_TOKEN` is set, configure the cBot’s `BridgeToken` parameter to match.
