# Goldmap — Gold Market Intelligence Platform for MT5

An institutional-grade analytical platform that reproduces Bookmap concepts for XAUUSD (Gold) trading using MetaTrader 5 data.

## Important Disclaimer

This system relies on **inference models** — NOT real order book data.
Every output clearly distinguishes between:
- **Observed data** (ticks, OHLC, bid/ask)
- **Derived data** (calculated from observed)
- **Estimated data** (inferred/modeled)

## Features

- Volume Engine (relative volume, spikes, imbalance)
- Liquidity Engine (zones, strength, ranking)
- Absorption Engine (high volume + low movement detection)
- Stop Hunt Engine (breakout + reversal detection)
- Fake Breakout Engine (failed breakout detection)
- Pseudo-Bookmap Heatmap Visualization
- Real-time Dashboard with Dash/Plotly

## Requirements

- Python 3.12+
- MetaTrader 5 terminal (running)
- Windows OS (MT5 Python API requirement)

## Installation

```bash
cd goldmap
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml`:
```yaml
mt5:
  symbol: "XAUUSD"
  timeframe: "M1"
  login: YOUR_LOGIN
  password: YOUR_PASSWORD
  server: YOUR_SERVER
```

## Running

### Start Backend API
```bash
python -m api.main
```

### Start Dashboard
```bash
python -m frontend.app
```

### Start Data Collector (background)
```bash
python -m core.collector
```

## Architecture

```
MT5 Connector
│
▼
Tick Collector
│
▼
Data Normalization Layer
│
▼
Feature Extraction Engine
│
├── Volume Engine
├── Liquidity Engine
├── Absorption Engine
├── Stop Hunt Engine
└── Fake Breakout Engine
│
▼
Signal Engine
│
▼
Analytics Engine
│
▼
Visualization Engine (Dash/Plotly)
│
▼
Dashboard
```

## Project Structure

```
goldmap/
├── api/              # FastAPI REST + WebSocket API
├── core/             # Core data collection and processing
├── engines/          # Intelligence engines
├── models/           # Pydantic data models
├── storage/          # SQLite database layer
├── frontend/         # Dash/Plotly dashboard
├── utils/            # Utility functions
├── config/           # Configuration files
├── tests/            # Test suite
└── requirements.txt
```