"""Tests for MexcExecutionEngine extensions.

Covers:
- env-var guards for live vs paper
- _set_isolated_leverage: MEXC-specific openType=1 (isolated) params
- _place_stop_loss_order: MEXC uses triggerPrice + reduceOnly
- _cancel_stop_loss_order: cancels or silently ignores already-gone orders
- paper-mode does not touch the exchange
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import (
    BotConfig, DataConfig, ExchangeConfig, RiskConfig, RuntimeConfig, StrategyConfig,
)
from core.models import SymbolState


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
        exchange=ExchangeConfig(exchange_id="mexc", market_type="swap"),
        runtime=RuntimeConfig(paper_mode=paper),
    )


def _make_state(symbol: str = "BTC/USDT:USDT") -> SymbolState:
    s = SymbolState(symbol=symbol, tick_size=0.01, max_leverage=25.0)
    s.last_price = 70000.0
    s.fair_price = 70000.0
    return s


def _make_engine(paper: bool = True, *, no_keys: bool = True):
    """Build engine with mocked exchange. Clears MEXC env vars by default."""
    from exchanges.mexc.execution import MexcExecutionEngine

    env_override = {"MEXC_API_KEY": "", "MEXC_SECRET": ""} if no_keys else {}
    with patch.dict(os.environ, env_override, clear=False):
        engine = MexcExecutionEngine(_make_cfg(paper=paper))

    engine.exchange = MagicMock()
    engine.exchange.price_to_precision = lambda sym, p: str(round(p, 8))
    engine.exchange.amount_to_precision = lambda sym, a: str(round(a, 2))
    engine.exchange.market = MagicMock(return_value={"id": "BTC_USDT", "contractSize": 1.0})
    return engine


class TestLiveRequiresApiKeys:
    """Live mode must fail fast in initialize() when API keys are missing."""

    @pytest.mark.asyncio
    async def test_initialize_raises_when_live_and_no_keys(self):
        from exchanges.mexc.execution import MexcExecutionEngine

        with patch.dict(os.environ, {"MEXC_API_KEY": "", "MEXC_SECRET": ""}, clear=False):
            engine = MexcExecutionEngine(_make_cfg(paper=False))
            engine.exchange = MagicMock()
            engine.exchange.load_markets = AsyncMock(return_value={})

            with pytest.raises(RuntimeError, match="MEXC_API_KEY"):
                await engine.initialize()

    @pytest.mark.asyncio
    async def test_initialize_ok_in_paper_without_keys(self):
        engine = _make_engine(paper=True)
        engine.exchange.load_markets = AsyncMock(return_value={"BTC/USDT:USDT": {}})
        result = await engine.initialize()
        assert isinstance(result, dict)


class TestSetIsolatedLeverage:
    """MEXC accepts openType=1 (isolated) + positionType via params, NOT set_margin_mode."""

    @pytest.mark.asyncio
    async def test_paper_mode_skips_api_calls(self):
        engine = _make_engine(paper=True)
        engine.exchange.set_leverage = AsyncMock()
        engine.exchange.set_margin_mode = AsyncMock()

        ok = await engine._set_isolated_leverage(_make_state(), 25.0)
        assert ok is True
        engine.exchange.set_leverage.assert_not_called()
        engine.exchange.set_margin_mode.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_sets_leverage_with_isolated_params(self):
        """MEXC set_leverage must include openType=1 (isolated) in params."""
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        captured: dict = {}

        async def fake_set_leverage(lev, symbol, **kw):
            captured["leverage"] = lev
            captured["symbol"] = symbol
            captured["params"] = kw.get("params", {})
            return {"leverage": lev}

        engine.exchange.set_leverage = fake_set_leverage

        ok = await engine._set_isolated_leverage(_make_state("BTC/USDT:USDT"), 25.0)

        assert ok is True
        assert captured["leverage"] == 25
        assert captured["symbol"] == "BTC/USDT:USDT"
        assert captured["params"].get("openType") == 1, (
            "MEXC openType=1 selects isolated margin. "
            "Without it the leverage is applied to cross margin (unsafe)."
        )

    @pytest.mark.asyncio
    async def test_leverage_failure_aborts(self):
        """If set_leverage raises, _set_isolated_leverage must return False."""
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.set_leverage = AsyncMock(side_effect=Exception("API_ERROR"))

        ok = await engine._set_isolated_leverage(_make_state(), 25.0)
        assert ok is False

    @pytest.mark.asyncio
    async def test_zero_leverage_aborts(self):
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.set_leverage = AsyncMock()

        ok = await engine._set_isolated_leverage(_make_state(), 0.0)
        assert ok is False


class TestPlaceStopLoss:
    """MEXC uses triggerPrice+reduceOnly via create_order with type='stop_market'."""

    @pytest.mark.asyncio
    async def test_paper_mode_no_api_call_but_sets_state(self):
        engine = _make_engine(paper=True)
        engine.exchange.create_order = AsyncMock()
        state = _make_state()

        await engine._place_stop_loss_order(state, "sell", 620.0, 70000.0, 25.0)
        engine.exchange.create_order.assert_not_called()
        assert state.stop_loss_price is not None

    @pytest.mark.asyncio
    async def test_live_sends_stop_market_reduce_only(self):
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        captured: dict = {}

        async def fake_create_order(**kwargs):
            captured.update(kwargs)
            return {"id": "mexc-sl-1"}

        engine.exchange.create_order = fake_create_order
        state = _make_state()

        await engine._place_stop_loss_order(state, "sell", 620.0, 70000.0, 25.0)

        assert captured["type"] == "stop_market"
        assert captured["side"] == "buy"  # close side for SHORT position
        assert captured["symbol"] == state.symbol
        params = captured.get("params", {}) or {}
        assert params.get("reduceOnly") is True, "reduceOnly=True required to close, not flip"
        assert params.get("triggerPrice") is not None and float(params["triggerPrice"]) > 0
        assert "stopLossPrice" not in params, "duplicate stopLossPrice must be absent (H1 fix)"

    @pytest.mark.asyncio
    async def test_long_stop_is_below_entry(self):
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        captured: dict = {}

        async def fake_create_order(**kwargs):
            captured.update(kwargs)
            return {"id": "mexc-sl-long"}

        engine.exchange.create_order = fake_create_order
        state = _make_state()

        entry = 70000.0
        await engine._place_stop_loss_order(state, "buy", 620.0, entry, 25.0)

        assert state.stop_loss_price is not None and state.stop_loss_price < entry

    @pytest.mark.asyncio
    async def test_short_stop_is_above_entry(self):
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        captured: dict = {}

        async def fake_create_order(**kwargs):
            captured.update(kwargs)
            return {"id": "mexc-sl-short"}

        engine.exchange.create_order = fake_create_order
        state = _make_state()

        entry = 70000.0
        await engine._place_stop_loss_order(state, "sell", 620.0, entry, 25.0)

        assert state.stop_loss_price is not None and state.stop_loss_price > entry

    @pytest.mark.asyncio
    async def test_api_error_falls_back_to_software_stop(self):
        """Like Gate.io: even if exchange SL placement fails, state.stop_loss_price is set
        so the engine's software tick-level stop still protects the position."""
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.create_order = AsyncMock(side_effect=Exception("MEXC_ERROR"))
        state = _make_state()

        await engine._place_stop_loss_order(state, "sell", 620.0, 70000.0, 25.0)

        assert state.stop_loss_price is not None
        assert state.stop_loss_order_id is None

    @pytest.mark.asyncio
    async def test_api_error_does_not_force_close_timer(self):
        """MEXC override must NOT force-close on SL API error — rely on software stop."""
        import time as time_mod

        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        state = _make_state()
        state.close_due_monotonic = 0.0
        engine.exchange.create_order = AsyncMock(side_effect=Exception("MEXC_ERROR"))

        before = time_mod.monotonic()
        await engine._place_stop_loss_order(state, "sell", 620.0, 70000.0, 25.0)

        assert state.close_due_monotonic < before, (
            "MEXC override must NOT force-close on SL API error — "
            "software stop (state.stop_loss_price) provides the safety net."
        )


class TestCancelStopLoss:
    """_cancel_stop_loss_order must clean up order ID and tolerate already-gone orders."""

    @pytest.mark.asyncio
    async def test_paper_mode_clears_order_id_without_api_call(self):
        engine = _make_engine(paper=True)
        engine.exchange.cancel_order = AsyncMock()
        state = _make_state()
        state.stop_loss_order_id = "paper-fake-id"

        await engine._cancel_stop_loss_order(state)

        engine.exchange.cancel_order.assert_not_called()
        assert state.stop_loss_order_id is None

    @pytest.mark.asyncio
    async def test_no_order_id_is_noop(self):
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.cancel_order = AsyncMock()
        state = _make_state()
        state.stop_loss_order_id = None

        await engine._cancel_stop_loss_order(state)

        engine.exchange.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_cancels_and_clears_order_id(self):
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.cancel_order = AsyncMock(return_value={"status": "canceled"})
        state = _make_state()
        state.stop_loss_order_id = "mexc-sl-99"

        await engine._cancel_stop_loss_order(state)

        engine.exchange.cancel_order.assert_called_once_with("mexc-sl-99", state.symbol)
        assert state.stop_loss_order_id is None

    @pytest.mark.asyncio
    async def test_already_filled_is_silently_ignored(self):
        """A 'filled' error means the SL already triggered — not a problem."""
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.cancel_order = AsyncMock(side_effect=Exception("order is already filled"))
        state = _make_state()
        state.stop_loss_order_id = "mexc-sl-100"

        await engine._cancel_stop_loss_order(state)

        assert state.stop_loss_order_id is None

    @pytest.mark.asyncio
    async def test_unexpected_error_logs_but_still_clears_order_id(self):
        """Unknown errors are logged (not re-raised); order ID is still cleared."""
        engine = _make_engine(paper=False)
        engine.cfg.runtime.paper_mode = False
        engine.exchange.cancel_order = AsyncMock(side_effect=Exception("network timeout"))
        state = _make_state()
        state.stop_loss_order_id = "mexc-sl-101"

        await engine._cancel_stop_loss_order(state)

        assert state.stop_loss_order_id is None
