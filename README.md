# hftbot_fairprice
HFT-Bot for sniping Fair price spikes, can be used on multiple crypto exchenges.
# HFT Bot — Fair Price Momentum Strategy

Asyncio-based perpetual futures trading bot.
Supports: **GateIO**, **MEXC**, **Binance USDM**.

## Strategy

Monitors exchange WebSocket order-book feed for rapid "fair price" moves.
When price moves sharply within a short window (configurable threshold), the
bot opens a position in the direction of the move and closes it after a hold
period or trailing stop.

## Project structure

    core/               Core engine, config, risk manager, paper engine, signal
    exchanges/          Exchange-specific WebSocket + REST execution adapters
      gateio/           GateIO futures
      mexc/             MEXC futures
      binance/          Binance USDM futures
    run_gateio.py       Entry point — GateIO
    run_mexc.py         Entry point — MEXC
    run_binance.py      Entry point — Binance
    config.*.json       Example configs (copy & edit)
    tests/              Unit & integration tests (pytest)

## Quick start

### 1. Python 3.12+

    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### 2. Create log/data directories

    mkdir -p logs/gateio/paper logs/mexc/paper logs/binance/paper data

### 3. Paper trading (no API keys needed)

    python run_gateio.py --config config.gateio.paper.json

### 4. Live trading

Copy `.env.example` to `.env` and fill in your API keys:

    cp .env.example .env

Set `"paper_mode": false` in the config, then:

    python run_gateio.py --config config.gateio.paper.json

## Key config parameters

| Section  | Parameter                  | Description                                          |
|----------|---------------------------|------------------------------------------------------|
| strategy | fair_rise_threshold        | Min price move to trigger signal (0.035 = 3.5%)      |
| strategy | fair_move_window_seconds   | Rolling window for measuring the move                |
| strategy | hold_seconds               | Max hold time before forced close                    |
| strategy | quote_size_usdt            | Position size in USDT                                |
| strategy | max_spread_pct             | Skip if spread exceeds this %                        |
| strategy | max_open_positions         | Maximum simultaneous open positions                  |
| risk     | trailing_enabled           | Enable trailing stop                                 |
| risk     | trailing_activation_pct_*  | Profit % to activate trailing                        |
| risk     | trailing_lock_pct          | Fraction of profit locked when trailing activates    |
| risk     | stop_loss_margin_fraction_*| Stop loss as fraction of initial margin              |
| runtime  | paper_mode                 | true = paper trading, false = live                   |
| runtime  | paper_balance_usdt         | Virtual balance for paper mode                       |

## Tests

    pytest tests/ -v

## Notes

- Never commit your `.env` file — it contains your API keys
- Start with `paper_mode: true` to verify behavior before going live
- Positions survive restarts via `position_state_file`
- The bot uses exponential backoff on WebSocket reconnects
- Symbols in `symbols_denylist` are skipped (exclude high-cap coins with low volatility)
