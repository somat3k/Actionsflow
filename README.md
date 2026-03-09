# Quantum Trading System

A fully automated, production-grade perpetuals trading system built on **GitHub Actions** that combines:

- 🔗 **[Hyperliquid](https://hyperliquid.xyz)** – decentralised perpetuals exchange for data fetching and live execution
- 🤖 **ML Ensemble** – LSTM, XGBoost, Gradient Boosting, and Random Forest models for signal generation
- 🧠 **[Gemini AI](https://ai.google.dev)** – orchestration, market regime detection, leverage recommendation, and performance review
- 📄 **Paper Broker** – realistic simulation with fees, funding rates, slippage, and liquidations
- ⚡ **Live Trader Commander** – isolated-margin trade execution on Hyperliquid with safety guards
- 📊 **Evaluator** – comprehensive metrics (Sharpe, Sortino, Calmar, win rate, profit factor) with auto-adjustments

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 GitHub Actions Orchestration                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Schedule   │  │  Paper Train │  │  Evaluation & Adjust  │  │
│  │  (*/15 min)  │  │   (Daily)    │  │      (Hourly)         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │
│         │                 │                     │               │
│  ┌──────▼─────────────────▼─────────────────────▼────────────┐  │
│  │                Gemini AI Orchestrator                      │  │
│  │   • Validates ML signals    • Recommends leverage (10–35x) │  │
│  │   • Detects market regime   • Reviews performance          │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼─────────────────────────────────┐  │
│  │               ML Ensemble (4 models)                       │  │
│  │    LSTM (35%)  XGBoost (35%)  GBM (15%)  RF (15%)          │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼─────────────────────────────────┐  │
│  │             Hyperliquid Data Feed                          │  │
│  │  • OHLCV  • Order book  • Funding rates  • Open interest   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| [Quantum Trading Orchestrator](.github/workflows/quantum-trading.yml) | Every 15 min | Main signal cycle: paper or live |
| [Paper Trading – Model Training](.github/workflows/paper-training.yml) | Daily 00:00 UTC | Train / retrain all ML models |
| [Live Trader Commander](.github/workflows/live-trading.yml) | On dispatch | Execute real trades (guarded) |
| [Model Evaluation & Adjustments](.github/workflows/model-evaluation.yml) | Every hour | Evaluate performance, emit adjustments |
| [CI – Tests & Lint](.github/workflows/ci.yml) | Push / PR | Unit tests on Python 3.10 & 3.11 |

---

## Trading Configuration

Key parameters in [`config/trading_config.yaml`](config/trading_config.yaml):

| Parameter | Default | Description |
|---|---|---|
| `trading.mode` | `paper` | `paper` \| `live` |
| `trading.initial_equity` | `10000.0` | Starting equity (USD) |
| `trading.leverage.min` | `10` | Minimum leverage |
| `trading.leverage.max` | `35` | Maximum leverage |
| `trading.leverage.default` | `15` | Starting leverage |
| `trading.markets` | BTC, ETH, SOL, ARB | Symbols to trade (isolated margin) |
| `trading.risk.max_drawdown_pct` | `0.20` | 20% max drawdown before halt |
| `ml.signals.long_threshold` | `0.60` | P(long) to enter |
| `gemini.model` | `gemini-1.5-pro` | Gemini model for orchestration |

---

## Repository Secrets & Variables

### Required Secrets

| Secret | Required For | Description |
|---|---|---|
| `GEMINI_API_KEY` | All modes | Google Gemini AI API key |
| `HYPERLIQUID_PRIVATE_KEY` | Live trading | Ethereum private key for signing orders |
| `HYPERLIQUID_WALLET_ADDRESS` | Live trading | Wallet address |

### Required Variables (for live trading)

| Variable | Value | Description |
|---|---|---|
| `LIVE_TRADING_ENABLED` | `true` | Master kill-switch for live trading |

---

## Quick Start

### Paper Trading (no secrets required beyond Gemini)

1. **Fork this repository**
2. Add `GEMINI_API_KEY` to repository secrets
3. Go to **Actions → Paper Trading – Model Training** → Run workflow
4. Go to **Actions → Quantum Trading Orchestrator** → Run workflow (mode: `paper`)

### Live Trading (requires Hyperliquid account)

1. Complete paper trading setup above
2. Add `HYPERLIQUID_PRIVATE_KEY` and `HYPERLIQUID_WALLET_ADDRESS` secrets
3. Set repository variable `LIVE_TRADING_ENABLED = true`
4. Go to **Actions → Live Trader Commander** → Run workflow
   - Type `CONFIRM_LIVE` in the confirmation field

---

## Project Structure

```
.
├── .github/workflows/
│   ├── quantum-trading.yml         # Main orchestration (every 15 min)
│   ├── paper-training.yml          # ML model training (daily)
│   ├── live-trading.yml            # Live trader commander
│   ├── model-evaluation.yml        # Evaluation & adjustments (hourly)
│   └── ci.yml                      # CI tests
├── src/
│   ├── config.py                   # Configuration management
│   ├── data_fetcher.py             # Hyperliquid data fetching
│   ├── ml_models.py                # ML ensemble (LSTM, XGBoost, GB, RF)
│   ├── gemini_orchestrator.py      # Gemini AI orchestration
│   ├── paper_broker.py             # Paper trading simulator
│   ├── live_trader.py              # Live trading commander
│   ├── risk_manager.py             # Risk & leverage management
│   ├── evaluator.py                # Performance evaluation
│   ├── utils.py                    # Shared utilities & indicators
│   └── main.py                     # Entry point
├── tests/
│   ├── test_data_fetcher.py
│   ├── test_paper_broker.py
│   ├── test_evaluator.py
│   └── test_risk_manager.py
├── config/
│   └── trading_config.yaml         # Full trading configuration
├── requirements.txt
└── pyproject.toml
```

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Manual training run (paper mode)
python -m src.main --run-type train-models --mode paper

# Manual signal cycle (paper mode)
python -m src.main --run-type signal --mode paper

# Evaluate performance
python -m src.main --run-type evaluate --mode paper
```

---

## Risk Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency perpetuals trading with leverage involves substantial risk of loss. Never trade with money you cannot afford to lose. Past paper trading results do not guarantee future live performance.