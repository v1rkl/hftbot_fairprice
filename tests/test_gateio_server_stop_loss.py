"""
TDD tests for Gate.io server-side stop-loss using price_orders API with close=true.

Gate.io /futures/usdt/price_orders endpoint with initial.close=true guarantees
the trigger order CLOSES an existing position rather than opening a new one.

Design:
  - _place_stop_loss_order: calls private_futures_post_settle_price_orders directly
  - _cancel_stop_loss_order: calls private_futures_delete_settle_price_orders_order_id
  - state.stop_loss_price always set (software backup in engine.py)
  - state.stop_loss_order_id set from API response (for cancel on normal exit)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

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
        risk=RiskConfig(
            stop_loss_margin_fraction=0.65,
            stop_loss_margin_fraction_tier1=0.45,
            stop_loss_margin_fraction_tier2=0.55,
            lev_tier1_max=30.0,
            lev_tier2_max=50.0,
            stop_loss_high_lev_threshold=100.0,
            stop_loss_high_lev_margin_fraction=0.90,
        ),
        data=DataConfig(),
        exchange=ExchangeConfig(exchange_id="gateio", market_type="swap"),
        runtime=RuntimeConfig(paper_mode=paper),
    )


def _make_state(symbol: str = "STO/USDT:USDT") -> SymbolState:
    s = SymbolState(symbol=symbol, tick_size=0.00001, max_leverage=25.0)
    s.last_price = 0.12152
    s.fair_price = 0.12152
    return s


def _make_engine(paper: bool = False):
    from exchanges.gateio.execution import GateioExecutionEngine

    cfg = _make_cfg(paper=paper)
    engine = GateioExecutionEngine.__new__(GateioExecutionEngine)
    engine.cfg = cfg
    engine.paper_account = None
    engine._extra_params = {"settle": "usdt"}
    engine.exchange = MagicMock()
    engine.exchange.price_to_precision = lambda sym, p: str(round(p, 8))
    engine.exchange.amount_to_precision = lambda sym, a: str(round(a, 2))
    engine.exchange.market = MagicMock(return_value={"id": "STO_USDT"})
    return engine


# ---------------------------------------------------------------------------
# _place_stop_loss_order: uses price_orders API (not create_order)
# ---------------------------------------------------------------------------

class TestPlaceStopLossUsesNativeAPI:
    """Stop-loss must use private_futures_post_settle_price_orders with close=true."""

    @pytest.mark.asyncio
    async def test_calls_price_orders_not_create_order(self):
        """Must use price_orders endpoint, NOT create_order."""
        engine = _make_engine()
        state = _make_state()
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock(
            return_value={"id": "88001"}
        )
        engine.exchange.create_order = AsyncMock()  # should NOT be called

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        engine.exchange.private_futures_post_settle_price_orders.assert_called_once()
        engine.exchange.create_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_body_contains_close_true(self):
        """initial.close must be True so Gate.io closes the position, not opens a new one."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88001"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        initial = captured.get("initial", {})
        assert initial.get("close") is True, (
            "initial.close must be True — this is the only safe way to close a "
            "Gate.io position via price-trigger order without risk of opening a new one"
        )

    @pytest.mark.asyncio
    async def test_initial_size_is_zero_with_close_true(self):
        """initial.size=0 with close=true means 'close entire position'."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88002"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        initial = captured.get("initial", {})
        assert int(initial.get("size", -1)) == 0, (
            "initial.size=0 combined with close=true tells Gate.io to close the full position"
        )

    @pytest.mark.asyncio
    async def test_initial_price_is_zero_for_market_execution(self):
        """initial.price='0' means market execution at trigger (not stop-limit)."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88003"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        initial = captured.get("initial", {})
        assert str(initial.get("price", "x")) == "0", (
            "initial.price='0' ensures market execution at trigger — "
            "any other value creates a stop-limit that may not fill during fast moves"
        )

    @pytest.mark.asyncio
    async def test_initial_tif_is_ioc(self):
        """initial.tif must be 'ioc' — Gate.io REQUIRES it for price=0 (market) orders.
        Without it Gate.io returns AUTO_INVALID_PARAM_INITIAL_TIF and the stop is never placed.
        """
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88003b"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        initial = captured.get("initial", {})
        assert initial.get("tif") == "ioc", (
            "Gate.io requires tif='ioc' for market orders (price='0'). "
            "Without it the API returns AUTO_INVALID_PARAM_INITIAL_TIF and the "
            "stop-loss falls back to software-only, leaving the position unprotected "
            "against fast price moves (as seen in the D/USDT liquidation incident)."
        )

    @pytest.mark.asyncio
    async def test_settle_param_sent(self):
        """Request body must include settle='usdt' for Gate.io USDT perpetuals."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88004"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        assert captured.get("settle") == "usdt"


# ---------------------------------------------------------------------------
# _place_stop_loss_order: trigger rule direction
# ---------------------------------------------------------------------------

class TestStopLossTriggerRule:
    """Gate.io rule=1 means price>=trigger, rule=2 means price<=trigger."""

    @pytest.mark.asyncio
    async def test_long_uses_rule_2_price_falls_to_stop(self):
        """LONG: stop is below entry. Trigger when price falls to stop → rule=2 (price<=)."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88010"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "buy", 500.0, 0.12152, 25.0)

        trigger = captured.get("trigger", {})
        assert trigger.get("rule") == 2, (
            "LONG stop: price falls to stop, so rule=2 (price <= trigger_price). "
            "rule=1 would trigger when price rises above stop — wrong direction for LONG."
        )

    @pytest.mark.asyncio
    async def test_short_uses_rule_1_price_rises_to_stop(self):
        """SHORT: stop is above entry. Trigger when price rises to stop → rule=1 (price>=)."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88011"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        trigger = captured.get("trigger", {})
        assert trigger.get("rule") == 1, (
            "SHORT stop: price rises to stop, so rule=1 (price >= trigger_price). "
            "rule=2 would trigger when price falls below stop — wrong direction for SHORT."
        )

    @pytest.mark.asyncio
    async def test_long_stop_price_is_below_entry(self):
        """LONG stop price must be below entry price."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88012"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        entry = 0.12152
        await engine._place_stop_loss_order(state, "buy", 500.0, entry, 25.0)

        trigger_price = float(captured.get("trigger", {}).get("price", 0))
        assert trigger_price < entry, (
            f"LONG stop trigger {trigger_price} must be below entry {entry}"
        )

    @pytest.mark.asyncio
    async def test_short_stop_price_is_above_entry(self):
        """SHORT stop price must be above entry price."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88013"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        entry = 0.12152
        await engine._place_stop_loss_order(state, "sell", 620.0, entry, 25.0)

        trigger_price = float(captured.get("trigger", {}).get("price", 0))
        assert trigger_price > entry, (
            f"SHORT stop trigger {trigger_price} must be above entry {entry}"
        )


# ---------------------------------------------------------------------------
# _place_stop_loss_order: state update
# ---------------------------------------------------------------------------

class TestStopLossStateUpdate:
    """After placing stop-loss, state must be updated correctly."""

    @pytest.mark.asyncio
    async def test_sets_stop_loss_price_on_state(self):
        """state.stop_loss_price must be set — used as software backup in engine.py."""
        engine = _make_engine()
        state = _make_state()
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock(
            return_value={"id": "88020"}
        )

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        assert state.stop_loss_price is not None
        assert state.stop_loss_price > 0.12152, "SHORT stop must be above entry"

    @pytest.mark.asyncio
    async def test_sets_stop_loss_order_id_from_response(self):
        """state.stop_loss_order_id must be set from API response for later cancellation."""
        engine = _make_engine()
        state = _make_state()
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock(
            return_value={"id": "88021"}
        )

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        assert state.stop_loss_order_id == "88021"

    @pytest.mark.asyncio
    async def test_stop_loss_price_uses_tier_aware_fraction(self):
        """SL fraction must come from _sl_fraction (tier-aware), not hardcoded value."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "88022"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        # tier1 leverage (< 30): should use stop_loss_margin_fraction_tier1=0.45
        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        from core.risk import _sl_fraction, compute_stop_loss_price
        expected_frac = _sl_fraction(engine.cfg.risk, 25.0)
        expected_price = compute_stop_loss_price(0.12152, 25.0, "sell", expected_frac)
        expected_formatted = float(engine.exchange.price_to_precision(state.symbol, expected_price))

        assert state.stop_loss_price == pytest.approx(expected_formatted, rel=1e-6)


# ---------------------------------------------------------------------------
# _place_stop_loss_order: API error fallback
# ---------------------------------------------------------------------------

class TestStopLossApiFallback:
    """On API error, fall back to software-only stop (still set state.stop_loss_price)."""

    @pytest.mark.asyncio
    async def test_api_error_falls_back_to_software_stop(self):
        """If price_orders API fails, stop_loss_price must still be set for software backup."""
        engine = _make_engine()
        state = _make_state()
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock(
            side_effect=Exception("GATE_API_ERROR")
        )

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        assert state.stop_loss_price is not None, (
            "Even when API fails, state.stop_loss_price must be set so the "
            "engine's software stop monitoring (engine.py:309) can protect the position"
        )
        assert state.stop_loss_order_id is None, (
            "No exchange order was created, order_id must be None"
        )

    @pytest.mark.asyncio
    async def test_api_error_logs_warning_not_exception(self):
        """API error must log WARNING, not raise — position should continue with software stop."""
        import logging

        engine = _make_engine()
        state = _make_state()
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock(
            side_effect=Exception("NETWORK_ERROR")
        )

        with patch.object(logging, "warning") as mock_warn:
            # Must not raise
            await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        warned = any("stop" in str(c).lower() or "fallback" in str(c).lower()
                     for c in mock_warn.call_args_list)
        assert warned, "Must log a warning when stop-loss API placement fails"

    @pytest.mark.asyncio
    async def test_paper_mode_no_api_call(self):
        """In paper mode, no API call — just set state.stop_loss_price."""
        engine = _make_engine(paper=True)
        state = _make_state()
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock()

        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        engine.exchange.private_futures_post_settle_price_orders.assert_not_called()
        assert state.stop_loss_price is not None

    @pytest.mark.asyncio
    async def test_api_error_does_not_force_close_timer(self):
        """Gate.io override must NOT set close_due_monotonic on API error — software stop is enough."""
        import time as time_mod

        engine = _make_engine()
        state = _make_state()
        state.close_due_monotonic = 0.0
        engine.exchange.private_futures_post_settle_price_orders = AsyncMock(
            side_effect=Exception("GATE_API_ERROR")
        )

        before = time_mod.monotonic()
        await engine._place_stop_loss_order(state, "sell", 620.0, 0.12152, 25.0)

        assert state.close_due_monotonic < before, (
            "Gate.io override must NOT force-close on SL API error — "
            "unlike base_execution.py, it relies on software stop (state.stop_loss_price) "
            "rather than immediately queuing a close"
        )


# ---------------------------------------------------------------------------
# _cancel_stop_loss_order: uses price_orders DELETE endpoint
# ---------------------------------------------------------------------------

class TestCancelStopLossOrder:
    """Cancel must use private_futures_delete_settle_price_orders_order_id."""

    @pytest.mark.asyncio
    async def test_calls_price_orders_delete_endpoint(self):
        """Must use DELETE price_orders endpoint, not cancel_order."""
        engine = _make_engine()
        state = _make_state()
        state.stop_loss_order_id = "88030"
        engine.exchange.private_futures_delete_settle_price_orders_order_id = AsyncMock(
            return_value={}
        )
        engine.exchange.cancel_order = AsyncMock()

        await engine._cancel_stop_loss_order(state)

        engine.exchange.private_futures_delete_settle_price_orders_order_id.assert_called_once()
        engine.exchange.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_correct_order_id_and_settle(self):
        """Must send correct order_id and settle='usdt' to DELETE endpoint."""
        engine = _make_engine()
        state = _make_state()
        state.stop_loss_order_id = "88031"
        captured = {}

        async def fake_delete(body):
            captured.update(body)
            return {}

        engine.exchange.private_futures_delete_settle_price_orders_order_id = fake_delete

        await engine._cancel_stop_loss_order(state)

        assert str(captured.get("order_id", "")) == "88031", \
            "Must pass order_id (snake_case) matching the implementation"
        assert captured.get("settle") == "usdt"

    @pytest.mark.asyncio
    async def test_clears_order_id_after_success(self):
        """stop_loss_order_id must be None after successful cancel."""
        engine = _make_engine()
        state = _make_state()
        state.stop_loss_order_id = "88032"
        engine.exchange.private_futures_delete_settle_price_orders_order_id = AsyncMock(
            return_value={}
        )

        await engine._cancel_stop_loss_order(state)

        assert state.stop_loss_order_id is None

    @pytest.mark.asyncio
    async def test_clears_order_id_even_on_error(self):
        """stop_loss_order_id must be cleared even if cancel API call fails."""
        engine = _make_engine()
        state = _make_state()
        state.stop_loss_order_id = "88033"
        engine.exchange.private_futures_delete_settle_price_orders_order_id = AsyncMock(
            side_effect=Exception("404 not found")
        )

        await engine._cancel_stop_loss_order(state)

        assert state.stop_loss_order_id is None

    @pytest.mark.asyncio
    async def test_already_filled_error_is_warning_not_error(self):
        """'Already closed/filled' cancel error must log WARNING, not ERROR."""
        import logging

        engine = _make_engine()
        state = _make_state()
        state.stop_loss_order_id = "88034"
        engine.exchange.private_futures_delete_settle_price_orders_order_id = AsyncMock(
            side_effect=Exception("ORDER_CLOSED: order already closed")
        )

        with patch.object(logging, "warning") as mock_warn, \
             patch.object(logging, "error") as mock_err:
            await engine._cancel_stop_loss_order(state)

        mock_err.assert_not_called()
        assert mock_warn.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_api_call_when_no_order_id(self):
        """No DELETE call when stop_loss_order_id is None."""
        engine = _make_engine()
        state = _make_state()
        state.stop_loss_order_id = None
        engine.exchange.private_futures_delete_settle_price_orders_order_id = AsyncMock()

        await engine._cancel_stop_loss_order(state)

        engine.exchange.private_futures_delete_settle_price_orders_order_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_cancel_in_paper_mode(self):
        """Paper mode: no API call."""
        engine = _make_engine(paper=True)
        state = _make_state()
        state.stop_loss_order_id = "88035"
        engine.exchange.private_futures_delete_settle_price_orders_order_id = AsyncMock()

        await engine._cancel_stop_loss_order(state)


# ---------------------------------------------------------------------------
# Trigger price_type must be Mark Price (not Last Price)
#
# Gate.io liquidates positions based on MARK PRICE, not last traded price.
# If we trigger the SL on last price, the mark price can already reach the
# liquidation threshold while last price is still above our stop — resulting
# in a liquidation before our SL fires (as happened with DRIFT/USDT).
#
# Fix: use price_type=1 (Mark Price) so the SL triggers on the same price
# feed the exchange uses for liquidation, guaranteeing SL fires first.
# ---------------------------------------------------------------------------

class TestStopLossTriggerPriceType:
    """SL trigger must use price_type=1 (Mark Price), not price_type=0 (Last Price)."""

    @pytest.mark.asyncio
    async def test_trigger_uses_mark_price_not_last_price(self):
        """price_type=1 (Mark Price) must be used so SL fires before liquidation.

        Root cause of DRIFT liquidation: price_type=0 (Last Price) was used.
        Mark price hit liquidation level while last price was still above our stop.
        The exchange liquidated before our trigger fired.
        """
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "99001"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "buy", 500.0, 0.12152, 25.0)

        trigger = captured.get("trigger", {})
        price_type = trigger.get("price_type")
        assert price_type == 1, (
            f"trigger.price_type must be 1 (Mark Price), got {price_type}. "
            "Gate.io liquidates via mark price; using last price (0) allowed the "
            "DRIFT/USDT position to be liquidated before the SL trigger fired."
        )

    @pytest.mark.asyncio
    async def test_short_trigger_also_uses_mark_price(self):
        """Mark Price trigger applies to SHORT positions too."""
        engine = _make_engine()
        state = _make_state()
        captured = {}

        async def fake_price_order(body):
            captured.update(body)
            return {"id": "99002"}

        engine.exchange.private_futures_post_settle_price_orders = fake_price_order

        await engine._place_stop_loss_order(state, "sell", 500.0, 0.12152, 25.0)

        trigger = captured.get("trigger", {})
        assert trigger.get("price_type") == 1, "SHORT SL trigger must also use mark price (price_type=1)"

        engine.exchange.private_futures_delete_settle_price_orders_order_id.assert_not_called()
