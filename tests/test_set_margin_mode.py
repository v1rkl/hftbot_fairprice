"""
Tests for base_execution._set_isolated_leverage margin-mode error handling.

Covers the bug where ccxt.NotSupported from set_margin_mode caused the bot to
abort entry instead of silently skipping an unsupported operation.

Scenarios:
1. ccxt.NotSupported → skip (not abort)
2. String-matched known-harmless messages → skip
3. Genuine unknown error → abort
4. set_leverage failure → abort regardless of margin mode
"""
from __future__ import annotations

import pytest
import ccxt
from unittest.mock import AsyncMock, MagicMock

from core.config import (
    BotConfig, DataConfig, ExchangeConfig, RiskConfig, RuntimeConfig, StrategyConfig,
)
from core.models import SymbolState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(paper: bool = False) -> BotConfig:
    return BotConfig(
        strategy=StrategyConfig(fill_timeout_ms=300, quote_size_usdt=5.0),
        risk=RiskConfig(stop_loss_margin_fraction=0.65),
        data=DataConfig(),
        exchange=ExchangeConfig(exchange_id="testexchange", market_type="swap",
                                min_max_leverage=50.0),
        runtime=RuntimeConfig(paper_mode=paper),
    )


def _make_base_engine(paper: bool = False):
    """Build BaseExecutionEngine with mocked exchange."""
    from core.base_execution import BaseExecutionEngine

    cfg = _make_cfg(paper=paper)
    engine = BaseExecutionEngine.__new__(BaseExecutionEngine)
    engine.cfg = cfg
    engine.paper_account = None
    engine._extra_params = {}
    engine.exchange = MagicMock()
    return engine


def _make_state(symbol: str = "GUN/USDT:USDT") -> SymbolState:
    s = SymbolState(symbol=symbol, tick_size=0.0001, max_leverage=50.0)
    s.last_price = 1.0
    s.fair_price = 1.0
    return s


# ---------------------------------------------------------------------------
# 1. ccxt.NotSupported must be treated as "skip", not "abort"
# ---------------------------------------------------------------------------

class TestSetMarginModeNotSupported:
    """ccxt.NotSupported from set_margin_mode must be silently skipped."""

    @pytest.mark.asyncio
    async def test_not_supported_exception_does_not_abort_entry(self):
        """
        Regression: Gate.io raises ccxt.NotSupported for setMarginMode.
        The bot must skip margin-mode setting and continue to set_leverage.
        """
        engine = _make_base_engine()
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock(
            side_effect=ccxt.NotSupported("gateio setMarginMode() is not supported yet")
        )
        engine.exchange.set_leverage = AsyncMock(return_value={"leverage": 50})

        result = await engine._set_isolated_leverage(state, 50.0)

        assert result is True, (
            "ccxt.NotSupported from set_margin_mode must be silently skipped, "
            "not treated as a fatal error that aborts entry"
        )
        engine.exchange.set_leverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_supported_with_different_message_does_not_abort(self):
        """Any ccxt.NotSupported, regardless of message text, must be skipped."""
        engine = _make_base_engine()
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock(
            side_effect=ccxt.NotSupported("mexc does not support isolated margin on this pair")
        )
        engine.exchange.set_leverage = AsyncMock(return_value={"leverage": 20})

        result = await engine._set_isolated_leverage(state, 20.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_already_in_isolated_mode_does_not_abort(self):
        """'already' in error message → harmless, skip."""
        engine = _make_base_engine()
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock(
            side_effect=Exception("margin mode is already set to isolated")
        )
        engine.exchange.set_leverage = AsyncMock(return_value={"leverage": 50})

        result = await engine._set_isolated_leverage(state, 50.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_margin_mode_error_aborts_entry(self):
        """An unrecognised error from set_margin_mode must abort entry."""
        engine = _make_base_engine()
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock(
            side_effect=Exception("403 Forbidden: insufficient permissions")
        )
        engine.exchange.set_leverage = AsyncMock()

        result = await engine._set_isolated_leverage(state, 50.0)

        assert result is False, (
            "An unrecognised set_margin_mode error must abort entry to prevent "
            "opening a position in the wrong margin mode"
        )
        engine.exchange.set_leverage.assert_not_called()


# ---------------------------------------------------------------------------
# 2. set_leverage failure always aborts
# ---------------------------------------------------------------------------

class TestSetLeverageFailure:
    """set_leverage failure must abort entry regardless of margin mode outcome."""

    @pytest.mark.asyncio
    async def test_leverage_failure_after_successful_margin_mode_aborts(self):
        engine = _make_base_engine()
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock(return_value={})
        engine.exchange.set_leverage = AsyncMock(
            side_effect=Exception("leverage out of range")
        )

        result = await engine._set_isolated_leverage(state, 50.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_leverage_failure_after_skipped_margin_mode_aborts(self):
        engine = _make_base_engine()
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock(
            side_effect=ccxt.NotSupported("not supported yet")
        )
        engine.exchange.set_leverage = AsyncMock(
            side_effect=Exception("invalid leverage value")
        )

        result = await engine._set_isolated_leverage(state, 50.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_paper_mode_always_returns_true(self):
        """Paper mode must skip all exchange calls and return True."""
        engine = _make_base_engine(paper=True)
        state = _make_state()

        engine.exchange.set_margin_mode = AsyncMock()
        engine.exchange.set_leverage = AsyncMock()

        result = await engine._set_isolated_leverage(state, 50.0)

        assert result is True
        engine.exchange.set_margin_mode.assert_not_called()
        engine.exchange.set_leverage.assert_not_called()
