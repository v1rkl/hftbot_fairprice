from __future__ import annotations

import math
import time
from typing import Any

from core.interfaces import AbstractMessageParser
from core.models import SymbolState


def _is_valid_price(value: float) -> bool:
    """Return True if value is a finite positive number."""
    return math.isfinite(value) and value > 0


class GateioMessageParser(AbstractMessageParser):
    """
    Parse Gate.io futures WebSocket messages into SymbolState.

    Gate.io message structure:
      {
        "time": <epoch>,
        "channel": "futures.trades" | "futures.order_book" | "futures.tickers",
        "event": "update",
        "result": { ... }   <-- data is here, NOT in "data"
      }

    Key differences from MEXC:
    - Data lives in msg["result"], not msg["data"]
    - Symbol key is result["contract"] (format: "BTC_USDT")
    - Prices arrive as strings -> need float() cast
    - Order book levels are objects {"p": "price", "s": size}
    - futures.tickers contains both mark_price and last price
    """

    def extract_symbol(self, msg: dict[str, Any]) -> str | None:
        result = msg.get("result")
        if result is None:
            return None

        raw: str | None = None

        # result can be a dict (tickers, order_book) or list (trades)
        if isinstance(result, list):
            if result and isinstance(result[0], dict):
                raw = result[0].get("contract")
        elif isinstance(result, dict):
            raw = result.get("contract") or result.get("s")

        if not raw:
            return None

        # Gate.io uses BTC_USDT -> convert to BTC/USDT:USDT
        raw = raw.replace("_", "/")
        if ":" not in raw and raw.endswith("/USDT"):
            raw = f"{raw}:USDT"
        return raw

    def ingest(self, state: SymbolState, msg: dict[str, Any]) -> None:
        channel = str(msg.get("channel") or "")
        result = msg.get("result")
        if result is None:
            return

        # -- futures.trades --
        if channel == "futures.trades":
            trades = result if isinstance(result, list) else [result]
            if trades and isinstance(trades[-1], dict):
                price_str = trades[-1].get("price")
                if price_str is not None:
                    try:
                        val = float(price_str)
                        if _is_valid_price(val):
                            state.last_price = val
                    except (TypeError, ValueError):
                        pass

        # -- futures.tickers --
        elif channel == "futures.tickers":
            # Gate.io sends result as a list of ticker dicts.
            if isinstance(result, list) and result and isinstance(result[0], dict):
                data = result[0]
            elif isinstance(result, dict):
                data = result
            else:
                data = {}
            mark = data.get("mark_price") or data.get("mark")
            if mark is not None:
                try:
                    val = float(mark)
                    if _is_valid_price(val):
                        state.fair_price = val
                except (TypeError, ValueError):
                    pass
            last = data.get("last") or data.get("last_price")
            if last is not None:
                try:
                    val = float(last)
                    if _is_valid_price(val):
                        state.last_price = val
                except (TypeError, ValueError):
                    pass

        # -- futures.order_book --
        elif channel == "futures.order_book":
            data = result if isinstance(result, dict) else {}
            raw_bids = data.get("bids") or []
            raw_asks = data.get("asks") or []

            def parse_levels(raw: list[Any]) -> list[list[float]]:
                parsed: list[list[float]] = []
                for lvl in raw:
                    if isinstance(lvl, dict):
                        p_raw = lvl.get("p")
                        s_raw = lvl.get("s")
                    elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                        p_raw, s_raw = lvl[0], lvl[1]
                    else:
                        continue
                    if p_raw is None or s_raw is None:
                        continue
                    try:
                        p = float(p_raw)
                        s = float(s_raw)
                        if _is_valid_price(p) and math.isfinite(s) and s >= 0:
                            parsed.append([p, s])
                    except (TypeError, ValueError):
                        continue
                return parsed

            parsed_bids = parse_levels(raw_bids)
            parsed_asks = parse_levels(raw_asks)

            if parsed_bids:
                state.book.bid = parsed_bids[0][0]
                state.book.bids = parsed_bids
            if parsed_asks:
                state.book.ask = parsed_asks[0][0]
                state.book.asks = parsed_asks

        state.last_update_monotonic = time.monotonic()
