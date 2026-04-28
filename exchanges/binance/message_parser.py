from __future__ import annotations

import math
import time
from typing import Any

from core.interfaces import AbstractMessageParser
from core.models import SymbolState


def _is_valid_price(value: float) -> bool:
    return math.isfinite(value) and value > 0


class BinanceMessageParser(AbstractMessageParser):
    """Parse Binance USDT-M futures WebSocket messages into SymbolState.

    Binance stream formats:
      markPriceUpdate: {"e":"markPriceUpdate","s":"BTCUSDT","p":"<mark>"}
      bookTicker:      {"e":"bookTicker","s":"BTCUSDT","b":"<bid>","a":"<ask>"}
    """

    def extract_symbol(self, msg: dict[str, Any]) -> str | None:
        raw = msg.get("s")
        if not raw or not isinstance(raw, str):
            return None
        upper = raw.upper()
        if upper.endswith("USDT"):
            base = upper[:-4]
            return f"{base}/USDT:USDT"
        return None

    def ingest(self, state: SymbolState, msg: dict[str, Any]) -> None:
        event = msg.get("e", "")

        if event == "markPriceUpdate":
            p = msg.get("p")
            if p is not None:
                try:
                    val = float(p)
                    if _is_valid_price(val):
                        state.fair_price = val
                        state.fair_price_updated_monotonic = time.monotonic()
                        # No aggTrade/bookTicker — mark is our only price signal.
                        # Must update every tick: if we only set on first tick,
                        # last_price goes stale (historical value) and trailing/TP
                        # math explodes (entry=4.88, stale last=22.87 → fake +368%).
                        state.last_price = val
                        # Synthesize bid/ask from mark price so liquidity checks pass.
                        # We don't subscribe to bookTicker (too many msgs/sec).
                        # 0.05% half-spread is conservative for Binance perps.
                        half = val * 0.0005
                        state.book.bid = val - half
                        state.book.ask = val + half
                        # Synthetic depth: 10000 USDT per side, always clears depth gate.
                        qty = 10000.0 / val
                        state.book.bids = [[state.book.bid, qty]]
                        state.book.asks = [[state.book.ask, qty]]
                except (TypeError, ValueError):
                    pass

        elif event == "bookTicker":
            bid_raw = msg.get("b")
            ask_raw = msg.get("a")
            if bid_raw is not None:
                try:
                    val = float(bid_raw)
                    if _is_valid_price(val):
                        state.book.bid = val
                except (TypeError, ValueError):
                    pass
            if ask_raw is not None:
                try:
                    val = float(ask_raw)
                    if _is_valid_price(val):
                        state.book.ask = val
                except (TypeError, ValueError):
                    pass
            if state.book.bid and state.book.ask:
                state.last_price = (state.book.bid + state.book.ask) / 2.0

        elif event == "aggTrade":
            p = msg.get("p")
            if p is not None:
                try:
                    val = float(p)
                    if _is_valid_price(val):
                        state.last_price = val
                except (TypeError, ValueError):
                    pass

        state.last_update_monotonic = time.monotonic()
