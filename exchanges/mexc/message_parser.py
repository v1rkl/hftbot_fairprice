from __future__ import annotations

import math
import time
from typing import Any

from core.interfaces import AbstractMessageParser
from core.models import SymbolState


def _is_valid_price(value: float) -> bool:
    """Return True if value is a finite positive number."""
    return math.isfinite(value) and value > 0


def _try_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if _is_valid_price(val) else None


class MexcMessageParser(AbstractMessageParser):
    """Parse MEXC contract WebSocket messages into SymbolState.

    Channel layout (push.*):
      push.deal         — trade prints -> last_price
      push.fair.price   — {"price": X}  fair/mark price -> fair_price
      push.index.price  — {"price": X}  index price     -> fair_price
      push.ticker       — consolidated (lastPrice + fairPrice) -> both
      push.depth / push.depth.full — order book -> book.bids/asks

    NOTE: push.fair.price payload has {"price": X} — must be routed to fair_price only;
    earlier versions treated 'price' as last_price first and then again as fair, which
    made last == fair on every tick and killed the dislocation signal.
    """

    def extract_symbol(self, msg: dict[str, Any]) -> str | None:
        raw: Any = None
        data = msg.get("data") or msg
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        if isinstance(data, dict):
            raw = data.get("symbol") or data.get("s")
        if not raw:
            raw = msg.get("symbol") or msg.get("s")
        if not raw:
            param = msg.get("param")
            if isinstance(param, dict):
                raw = param.get("symbol")
        if not raw:
            return None
        text = str(raw).replace("_", "/")
        if ":" not in text and text.endswith("/USDT"):
            text = f"{text}:USDT"
        return text

    def ingest(self, state: SymbolState, msg: dict[str, Any]) -> None:
        channel = str(msg.get("channel") or msg.get("c") or "")
        raw_data = msg.get("data") or msg

        data: dict[str, Any] | None = None
        if isinstance(raw_data, dict):
            data = raw_data
        elif isinstance(raw_data, list):
            if raw_data and isinstance(raw_data[0], dict):
                data = raw_data[0]
            elif raw_data and isinstance(raw_data[0], (list, tuple)) and raw_data[0]:
                val = _try_float(raw_data[0][0])
                if val is not None:
                    state.last_price = val
                data = {}
            else:
                data = {}
        else:
            return

        # -- Route by channel --
        is_fair_channel = "fair" in channel or "index" in channel
        is_trade_channel = "deal" in channel or "trade" in channel

        if data:
            if is_fair_channel:
                # push.fair.price / push.index.price — {"price": X} is FAIR. Do not touch last_price.
                for k in ("fairPrice", "markPrice", "indexPrice", "price", "p"):
                    val = _try_float(data.get(k))
                    if val is not None:
                        state.fair_price = val
                        state.fair_price_updated_monotonic = time.monotonic()
                        break
            elif is_trade_channel:
                # push.deal — last trade price. Do not touch fair_price.
                for k in ("p", "price", "lastPrice", "last"):
                    val = _try_float(data.get(k))
                    if val is not None:
                        state.last_price = val
                        break
            else:
                # push.ticker and generic messages: may carry both.
                last_val: float | None = None
                for k in ("lastPrice", "last", "p", "price"):
                    last_val = _try_float(data.get(k))
                    if last_val is not None:
                        break
                if last_val is not None:
                    state.last_price = last_val

                fair_val: float | None = None
                for k in ("fairPrice", "markPrice", "mark", "indexPrice"):
                    fair_val = _try_float(data.get(k))
                    if fair_val is not None:
                        break
                if fair_val is not None:
                    state.fair_price = fair_val
                    state.fair_price_updated_monotonic = time.monotonic()

            # -- Order book (any channel may carry it, e.g. push.depth.full) --
            bids = data.get("bids") or data.get("b")
            asks = data.get("asks") or data.get("a")
            if isinstance(bids, list) and bids:
                parsed_bids: list[list[float]] = []
                for lvl in bids:
                    if isinstance(lvl, list) and len(lvl) >= 2:
                        p = _try_float(lvl[0])
                        s_raw = lvl[1]
                        try:
                            s = float(s_raw)
                        except (TypeError, ValueError):
                            continue
                        if p is not None and math.isfinite(s) and s >= 0:
                            parsed_bids.append([p, s])
                if parsed_bids:
                    state.book.bid = parsed_bids[0][0]
                    state.book.bids = parsed_bids
            if isinstance(asks, list) and asks:
                parsed_asks: list[list[float]] = []
                for lvl in asks:
                    if isinstance(lvl, list) and len(lvl) >= 2:
                        p = _try_float(lvl[0])
                        s_raw = lvl[1]
                        try:
                            s = float(s_raw)
                        except (TypeError, ValueError):
                            continue
                        if p is not None and math.isfinite(s) and s >= 0:
                            parsed_asks.append([p, s])
                if parsed_asks:
                    state.book.ask = parsed_asks[0][0]
                    state.book.asks = parsed_asks

        state.last_update_monotonic = time.monotonic()
