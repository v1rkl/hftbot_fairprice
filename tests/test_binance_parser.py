"""Regression tests for BinanceMessageParser.

Bug: state.last_price was set only on the first mark tick and then never updated,
because Binance paper doesn't subscribe to bookTicker/aggTrade. Stale last_price
caused trailing/TP math to explode (e.g. entry=4.88 while last=22.87 from an
earlier rally → +368% fake best_pnl → instant TP close at -0.10% from spread).
"""
from __future__ import annotations

from core.models import SymbolState
from exchanges.binance.message_parser import BinanceMessageParser


def _make_state(symbol: str = "RAVE/USDT:USDT") -> SymbolState:
    return SymbolState(symbol=symbol, tick_size=0.0001, max_leverage=20.0)


def test_mark_price_updates_last_price_every_tick() -> None:
    """last_price must track every markPriceUpdate (not just the first)."""
    parser = BinanceMessageParser()
    state = _make_state()

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "22.87"})
    assert state.last_price == 22.87

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "4.88"})
    assert state.last_price == 4.88, (
        "last_price must update on every mark tick, otherwise trailing/TP math "
        "sees a stale historical price and explodes"
    )


def test_mark_price_updates_bid_ask_every_tick() -> None:
    parser = BinanceMessageParser()
    state = _make_state()

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "10.0"})
    assert state.book.bid is not None and abs(state.book.bid - 9.995) < 1e-9
    assert state.book.ask is not None and abs(state.book.ask - 10.005) < 1e-9

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "20.0"})
    assert state.book.bid is not None and abs(state.book.bid - 19.99) < 1e-9
    assert state.book.ask is not None and abs(state.book.ask - 20.01) < 1e-9


def test_invalid_price_does_not_overwrite_last_price() -> None:
    parser = BinanceMessageParser()
    state = _make_state()

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "10.0"})
    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "not-a-number"})
    assert state.last_price == 10.0

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "-1"})
    assert state.last_price == 10.0

    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "0"})
    assert state.last_price == 10.0


def test_fair_price_also_tracks_mark() -> None:
    parser = BinanceMessageParser()
    state = _make_state()
    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "4.88"})
    assert state.fair_price == 4.88
    parser.ingest(state, {"e": "markPriceUpdate", "s": "RAVEUSDT", "p": "4.90"})
    assert state.fair_price == 4.90
