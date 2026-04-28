"""Tests for MexcMessageParser.

Covers:
- last_price set from trade (push.deal) messages only
- fair_price set from push.fair.price messages only (NOT last_price)
- push.ticker channel sets both (it carries lastPrice + fairPrice)
- order_book depth parsing
- Robust to list-shaped data and unexpected shapes
"""
from __future__ import annotations

import math

import pytest

from core.models import SymbolState
from exchanges.mexc.message_parser import MexcMessageParser


def _make_state(symbol: str = "BTC/USDT:USDT") -> SymbolState:
    return SymbolState(symbol=symbol, tick_size=0.01, max_leverage=25.0)


class TestExtractSymbol:
    def test_underscore_symbol_converted_to_slash(self) -> None:
        p = MexcMessageParser()
        assert p.extract_symbol({"symbol": "BTC_USDT"}) == "BTC/USDT:USDT"

    def test_symbol_inside_data_dict(self) -> None:
        p = MexcMessageParser()
        assert p.extract_symbol({"data": {"symbol": "ETH_USDT"}}) == "ETH/USDT:USDT"

    def test_symbol_inside_param_dict(self) -> None:
        p = MexcMessageParser()
        assert p.extract_symbol({"param": {"symbol": "SOL_USDT"}}) == "SOL/USDT:USDT"

    def test_missing_symbol_returns_none(self) -> None:
        p = MexcMessageParser()
        assert p.extract_symbol({"channel": "push.ping"}) is None

    def test_empty_dict_returns_none(self) -> None:
        p = MexcMessageParser()
        assert p.extract_symbol({}) is None


class TestLastPriceFromDealChannel:
    """push.deal (trades) is the ONLY authoritative source of last_price."""

    def test_deal_message_sets_last_price(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {
            "channel": "push.deal",
            "symbol": "BTC_USDT",
            "data": {"p": 70000.5, "v": 10, "T": 1},
        })
        assert s.last_price == 70000.5

    def test_deal_message_does_not_set_fair(self) -> None:
        """push.deal must never touch fair_price."""
        p = MexcMessageParser()
        s = _make_state()
        s.fair_price = 65000.0  # pre-existing
        p.ingest(s, {
            "channel": "push.deal",
            "symbol": "BTC_USDT",
            "data": {"p": 70000.5},
        })
        assert s.fair_price == 65000.0


class TestFairPriceFromFairChannel:
    """push.fair.price carries {"price": X} — must set fair_price ONLY, NOT last_price.

    Regression: older parser read 'price' key for last_price first, then again for fair_price.
    Result: last_price got overwritten with fair value on every fair.price tick, so fair==last
    always, defeating the whole fair-vs-last signal on MEXC.
    """

    def test_fair_channel_sets_fair_price(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {
            "channel": "push.fair.price",
            "symbol": "BTC_USDT",
            "data": {"price": 70123.4},
        })
        assert s.fair_price == 70123.4

    def test_fair_channel_does_not_overwrite_last_price(self) -> None:
        """Regression: fair.price must NOT touch last_price even though payload key is 'price'."""
        p = MexcMessageParser()
        s = _make_state()
        s.last_price = 70050.0  # set by an earlier deal tick
        p.ingest(s, {
            "channel": "push.fair.price",
            "symbol": "BTC_USDT",
            "data": {"price": 70123.4},
        })
        assert s.last_price == 70050.0, (
            "push.fair.price payload {'price': X} carries the FAIR price. "
            "Treating it as last_price makes fair == last on every tick and "
            "destroys the fair-vs-last dislocation signal."
        )
        assert s.fair_price == 70123.4

    def test_fair_channel_updates_fair_price_timestamp(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        s.fair_price_updated_monotonic = 0.0
        p.ingest(s, {
            "channel": "push.fair.price",
            "symbol": "BTC_USDT",
            "data": {"price": 70123.4},
        })
        assert s.fair_price_updated_monotonic > 0.0


class TestIndexChannel:
    def test_index_price_sets_fair(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {
            "channel": "push.index.price",
            "symbol": "BTC_USDT",
            "data": {"price": 69900.0},
        })
        assert s.fair_price == 69900.0

    def test_index_does_not_touch_last(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        s.last_price = 70050.0
        p.ingest(s, {
            "channel": "push.index.price",
            "symbol": "BTC_USDT",
            "data": {"price": 69900.0},
        })
        assert s.last_price == 70050.0


class TestTickerChannel:
    """push.ticker carries both fairPrice and lastPrice in one message."""

    def test_ticker_sets_both(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {
            "channel": "push.ticker",
            "symbol": "BTC_USDT",
            "data": {"lastPrice": 70010.0, "fairPrice": 70000.0},
        })
        assert s.last_price == 70010.0
        assert s.fair_price == 70000.0


class TestOrderBook:
    def test_depth_message_updates_bids_and_asks(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {
            "channel": "push.depth.full",
            "symbol": "BTC_USDT",
            "data": {
                "bids": [[70000.0, 1.5], [69999.5, 2.0]],
                "asks": [[70001.0, 1.0], [70001.5, 2.5]],
            },
        })
        assert s.book.bid == 70000.0
        assert s.book.ask == 70001.0
        assert len(s.book.bids) == 2
        assert len(s.book.asks) == 2

    def test_invalid_levels_skipped(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {
            "channel": "push.depth.full",
            "symbol": "BTC_USDT",
            "data": {
                "bids": [[70000.0, 1.5], [-1, 2.0], ["bad", "bad"]],
                "asks": [[70001.0, 1.0], [70001.5, -1]],
            },
        })
        # Only the first bid/ask should be parsed
        assert s.book.bid == 70000.0
        assert len(s.book.bids) == 1
        assert s.book.ask == 70001.0
        assert len(s.book.asks) == 1


class TestInvalidPayloads:
    def test_empty_data_dict_no_crash(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {"channel": "push.deal", "symbol": "BTC_USDT", "data": {}})
        assert s.last_price is None
        assert s.fair_price is None

    def test_non_finite_price_rejected(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {"channel": "push.deal", "symbol": "BTC_USDT", "data": {"p": math.nan}})
        assert s.last_price is None

    def test_negative_price_rejected(self) -> None:
        p = MexcMessageParser()
        s = _make_state()
        p.ingest(s, {"channel": "push.deal", "symbol": "BTC_USDT", "data": {"p": -1.0}})
        assert s.last_price is None
