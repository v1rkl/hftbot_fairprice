"""
Gate.io API compatibility tests.

Covers issues found in the compatibility audit:
1. fetch_order returns filled=None → must compute from amount - remaining
2. WS ping response must include time field
3. Cross-margin positions (leverage=0) must be skipped in reconciliation

Note: stop-loss order placement and cancellation tests moved to
test_gateio_server_stop_loss.py (server-side stop with close=true).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import (
    BotConfig,
    DataConfig,
    ExchangeConfig,
    RiskConfig,
    RuntimeConfig,
    StrategyConfig,
)
from core.models import SymbolState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(paper: bool = True) -> BotConfig:
    return BotConfig(
        strategy=StrategyConfig(fill_timeout_ms=300, quote_size_usdt=5.0),
        risk=RiskConfig(stop_loss_margin_fraction=0.65),
        data=DataConfig(),
        exchange=ExchangeConfig(exchange_id="gateio", market_type="swap"),
        runtime=RuntimeConfig(paper_mode=paper),
    )


def _make_state(symbol: str = "W/USDT:USDT") -> SymbolState:
    s = SymbolState(symbol=symbol, tick_size=0.00001, max_leverage=50.0)
    s.last_price = 0.015
    s.fair_price = 0.015
    return s


def _make_engine(paper: bool = True):
    """Build GateioExecutionEngine with mocked exchange."""
    from exchanges.gateio.execution import GateioExecutionEngine

    cfg = _make_cfg(paper=paper)
    engine = GateioExecutionEngine.__new__(GateioExecutionEngine)
    engine.cfg = cfg
    engine.paper_account = None
    engine._extra_params = {"settle": "usdt"}
    engine.exchange = MagicMock()
    engine.exchange.price_to_precision = lambda sym, p: str(round(p, 8))
    engine.exchange.amount_to_precision = lambda sym, a: str(round(a, 2))
    return engine


# ---------------------------------------------------------------------------
# Issue 3: fetch_order filled=None → amount - remaining
# ---------------------------------------------------------------------------

class TestFetchOrderFilledNone:
    """Gate.io futures fetch_order returns filled=None; bot must compute from amount-remaining."""

    @pytest.mark.asyncio
    async def test_filled_computed_from_amount_minus_remaining_when_none(self):
        engine = _make_engine(paper=False)
        order_closed = {
            "id": "123",
            "status": "closed",
            "filled": None,          # Gate.io always returns None
            "amount": 1000.0,
            "remaining": 0.0,
            "average": 0.01500,
            "price": 0.01500,
        }
        engine.exchange.fetch_order = AsyncMock(return_value=order_closed)

        result = await engine._wait_fill_with_timeout("W/USDT:USDT", "123", 1000.0)

        assert result.filled == 1000.0, (
            "filled must be amount - remaining (1000 - 0) when filled=None"
        )
        assert result.avg_price == 0.015

    @pytest.mark.asyncio
    async def test_partial_fill_computed_correctly(self):
        engine = _make_engine(paper=False)
        # First poll: still open, partial fill
        order_open = {
            "id": "123", "status": "open",
            "filled": None, "amount": 1000.0, "remaining": 400.0,
            "average": None, "price": 0.015,
        }
        # Second poll: closed
        order_closed = {
            "id": "123", "status": "closed",
            "filled": None, "amount": 1000.0, "remaining": 0.0,
            "average": 0.01502, "price": 0.015,
        }
        engine.exchange.fetch_order = AsyncMock(side_effect=[order_open, order_closed])

        result = await engine._wait_fill_with_timeout("W/USDT:USDT", "123", 1000.0)

        assert result.filled == 1000.0
        assert result.avg_price == 0.01502

    @pytest.mark.asyncio
    async def test_filled_field_used_when_not_none(self):
        """When filled is a real number (non-Gate.io exchange), use it directly."""
        engine = _make_engine(paper=False)
        order_closed = {
            "id": "456", "status": "closed",
            "filled": 500.0,
            "amount": 1000.0, "remaining": 500.0,
            "average": 0.016, "price": 0.016,
        }
        engine.exchange.fetch_order = AsyncMock(return_value=order_closed)

        result = await engine._wait_fill_with_timeout("W/USDT:USDT", "456", 1000.0)

        assert result.filled == 500.0

    @pytest.mark.asyncio
    async def test_timeout_cancel_refetch_uses_amount_minus_remaining(self):
        """After timeout+cancel, re-fetched order with filled=None still computed correctly."""
        engine = _make_engine(paper=False)
        # All polls return open (triggers timeout path)
        order_open = {
            "id": "789", "status": "open",
            "filled": None, "amount": 500.0, "remaining": 500.0,
            "average": None, "price": 0.015,
        }
        # After cancel, partial fill
        order_after_cancel = {
            "id": "789", "status": "canceled",
            "filled": None, "amount": 500.0, "remaining": 300.0,
            "average": 0.0149, "price": 0.015,
        }
        polls = [order_open] * 3  # 3 polls = 300ms timeout
        engine.exchange.fetch_order = AsyncMock(side_effect=polls + [order_after_cancel])
        engine.exchange.cancel_order = AsyncMock()

        result = await engine._wait_fill_with_timeout("W/USDT:USDT", "789", 500.0)

        assert result.filled == 200.0  # 500 - 300
        assert result.avg_price == 0.0149


# ---------------------------------------------------------------------------
# WS ping response must include time field
# ---------------------------------------------------------------------------

class TestGateioWsPing:
    """Gate.io WS ping response must echo the time field from the ping message."""

    @pytest.mark.asyncio
    async def test_pong_includes_time_from_ping(self):
        from exchanges.gateio.market_data import _GateioWSShard

        cfg = _make_cfg()
        shard = _GateioWSShard(cfg, ["W/USDT:USDT"], lambda m: None, 0, "wss://test")

        sent_messages = []
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=lambda m: sent_messages.append(m))

        import orjson
        ping_msg = {"channel": "futures.ping", "time": 1700000000, "event": ""}

        # Simulate ping handling directly
        await shard._handle_ping(mock_ws, ping_msg)

        assert len(sent_messages) == 1
        pong = orjson.loads(sent_messages[0])
        assert pong["channel"] == "futures.pong"
        assert pong.get("time") == 1700000000, (
            "Pong must echo the time from ping so Gate.io accepts it"
        )

    @pytest.mark.asyncio
    async def test_pong_uses_current_time_when_ping_has_no_time(self):
        from exchanges.gateio.market_data import _GateioWSShard
        import orjson, time as time_mod

        cfg = _make_cfg()
        shard = _GateioWSShard(cfg, ["W/USDT:USDT"], lambda m: None, 0, "wss://test")

        sent_messages = []
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=lambda m: sent_messages.append(m))

        before = int(time_mod.time())
        await shard._handle_ping(mock_ws, {"channel": "futures.ping"})
        after = int(time_mod.time())

        pong = orjson.loads(sent_messages[0])
        assert before <= pong["time"] <= after + 1


# ---------------------------------------------------------------------------
# Issue 5: Cross-margin positions skipped in reconciliation
# ---------------------------------------------------------------------------

class TestCrossMarginReconciliation:
    """fetch_open_positions must skip cross-margin positions (leverage=0) and warn."""

    @pytest.mark.asyncio
    async def test_cross_margin_position_is_skipped(self):
        engine = _make_engine(paper=False)
        # leverage="0" in Gate.io response means cross-margin
        raw_positions = [
            {
                "symbol": "W/USDT:USDT",
                "side": "long",
                "contracts": 1000.0,
                "contractSize": None,
                "entryPrice": 0.015,
                "notional": 15.0,
                "leverage": 0.0,       # cross-margin indicator
                "unrealizedPnl": 0.1,
            }
        ]
        engine.exchange.fetch_positions = AsyncMock(return_value=raw_positions)

        import logging
        with patch.object(logging, "warning") as mock_warn:
            positions = await engine.fetch_open_positions()

        assert positions == [], (
            "Cross-margin position (leverage=0) must be skipped — "
            "the bot only operates in isolated margin"
        )
        warned = any("cross" in str(call).lower() for call in mock_warn.call_args_list)
        assert warned, "Must log a warning when a cross-margin position is found"

    @pytest.mark.asyncio
    async def test_isolated_position_is_kept(self):
        engine = _make_engine(paper=False)
        raw_positions = [
            {
                "symbol": "W/USDT:USDT",
                "side": "short",
                "contracts": 500.0,
                "contractSize": None,
                "entryPrice": 0.01500,
                "notional": 7.5,
                "leverage": 50.0,     # isolated
                "unrealizedPnl": -0.05,
            }
        ]
        engine.exchange.fetch_positions = AsyncMock(return_value=raw_positions)

        positions = await engine.fetch_open_positions()

        assert len(positions) == 1
        assert positions[0]["leverage"] == 50.0
        assert positions[0]["side"] == "sell"
