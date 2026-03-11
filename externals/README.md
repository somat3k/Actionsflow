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
  "positions": {
    "long": 0.0,
    "short": 0.0
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
  }
}
```

Signal mapping:

- `0` = FLAT / HOLD
- `1` = LONG
- `2` = SHORT

## Python Bridge (shared)

Start the bridge server (uses the repository’s ML models and Gemini orchestrator):

```bash
python externals/python/gemini_bridge_server.py
```

Environment variables:

- `TRADING_CONFIG_PATH` (optional): path to `config/trading_config.yaml`
- `BRIDGE_HOST` (default `127.0.0.1`)
- `BRIDGE_PORT` (default `8001`)

> Ensure models are trained and saved before running the bridge:
>
> ```bash
> python -m src.main --run-type train-models --mode paper
> ```

## MetaTrader 4 (MQL4)

- Add `http://127.0.0.1:8001` to **Tools → Options → Expert Advisors → Allow WebRequest**.
- Compile `externals/mql4/QuantumGeminiBridge.mq4` and attach it to a chart.

## MetaTrader 5 (MQL5)

- Add `http://127.0.0.1:8001` to **Tools → Options → Expert Advisors → Allow WebRequest**.
- Compile `externals/mql5/QuantumGeminiBridge.mq5` and attach it to a chart.

## cTrader cBot (cAlgo)

- Create a new cBot and replace its content with `externals/ctrader/QuantumGeminiBridge.cs`.
- The cBot requires `AccessRights.Internet` to call the local bridge.
