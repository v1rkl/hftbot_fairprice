"""
TDD tests for external close (liquidation/exchange-side SL) PnL recording bugs:

Bug 1: close_reason stays "hold_timeout" instead of "external_close" for
       positions that were closed externally on the exchange.

Bug 2: exit_price uses state.last_price (current market price at detection time)
       which can be far above entry_price after price recovers, producing fake profit.
       Fix: use state.stop_loss_price as the exit estimate (price was at/below SL
       when the external close happened). Fall back to last_price only if SL
       price is not set.
"""
from __future__ import annotations

import pytest

from core.trade_log import pnl_linear_usdt


# ---------------------------------------------------------------------------
# Helpers that replicate the GONE-detection exit price / close_reason logic.
# We test the corrected logic in isolation before wiring it into engine.py.
# ---------------------------------------------------------------------------

def _resolve_exit_price(last_price: float | None, stop_loss_price: float | None) -> float:
    """
    Corrected logic: prefer stop_loss_price over last_price for external closes.
    Returns 0.0 if neither is available.
    """
    if stop_loss_price is not None and stop_loss_price > 0:
        return float(stop_loss_price)
    if last_price is not None:
        return float(last_price)
    return 0.0


def _resolve_close_reason(state_close_reason: str) -> str:
    """
    Corrected logic: always return "external_close" regardless of what the
    state had queued (e.g. "hold_timeout").
    """
    return "external_close"


# ---------------------------------------------------------------------------
# Bug 1: close_reason must always be "external_close" for GONE positions
# ---------------------------------------------------------------------------

class TestCloseReasonOverride:
    def test_hold_timeout_overridden(self):
        assert _resolve_close_reason("hold_timeout") == "external_close"

    def test_empty_string_overridden(self):
        assert _resolve_close_reason("") == "external_close"

    def test_any_reason_overridden(self):
        for reason in ("trailing_stop", "tp1", "signal_expired", "manual"):
            assert _resolve_close_reason(reason) == "external_close"


# ---------------------------------------------------------------------------
# Bug 2: exit price must NOT be last_price when stop_loss_price is available
# ---------------------------------------------------------------------------

class TestExitPriceResolution:
    def test_prefers_stop_loss_price_over_last_price(self):
        """SL price wins even when last_price is higher (price recovered)."""
        result = _resolve_exit_price(last_price=0.05214, stop_loss_price=0.05109)
        assert result == pytest.approx(0.05109)

    def test_falls_back_to_last_price_when_no_sl(self):
        result = _resolve_exit_price(last_price=0.05214, stop_loss_price=None)
        assert result == pytest.approx(0.05214)

    def test_falls_back_to_last_price_when_sl_zero(self):
        result = _resolve_exit_price(last_price=0.05214, stop_loss_price=0.0)
        assert result == pytest.approx(0.05214)

    def test_returns_zero_when_both_none(self):
        assert _resolve_exit_price(last_price=None, stop_loss_price=None) == 0.0

    def test_uses_sl_price_exactly(self):
        result = _resolve_exit_price(last_price=None, stop_loss_price=0.04800)
        assert result == pytest.approx(0.04800)


# ---------------------------------------------------------------------------
# End-to-end: DRIFT liquidation scenario (real values from trade log)
# With corrected exit price the PnL must be a LOSS, not a gain.
# ---------------------------------------------------------------------------

class TestDriftLiquidationScenario:
    """
    DRIFT LONG position, 20x leverage.
    entry=0.05213, sl=0.05109, actual liquidation at 0.05077.
    At detection time last_price had recovered to 0.05214.

    Before fix: exit=0.05214 → PnL = +0.0115 (fake profit, risk not tracked)
    After fix:  exit=0.05109 (SL) → PnL < 0 (loss correctly recorded)
    """
    ENTRY = 0.05213
    LAST_PRICE = 0.05214   # price recovered above entry at detection time
    SL_PRICE = 0.05109     # where we placed the stop-loss
    QTY = 115.0            # contracts
    CONTRACT_SIZE = 1.0    # DRIFT contract size

    def test_old_logic_produces_fake_profit(self):
        """Demonstrates the original bug: using last_price gives positive PnL."""
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=self.QTY,
            entry_price=self.ENTRY,
            exit_price=self.LAST_PRICE,   # BUG: used last_price
            contract_size=self.CONTRACT_SIZE,
        )
        assert pnl > 0, "Old logic incorrectly records a profit for a liquidation"

    def test_new_logic_produces_loss(self):
        """After fix: using stop_loss_price gives negative PnL."""
        exit_price = _resolve_exit_price(
            last_price=self.LAST_PRICE,
            stop_loss_price=self.SL_PRICE,
        )
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=self.QTY,
            entry_price=self.ENTRY,
            exit_price=exit_price,
            contract_size=self.CONTRACT_SIZE,
        )
        assert pnl < 0, f"Expected loss, got pnl={pnl}"

    def test_new_logic_close_reason_is_external_close(self):
        close_reason = _resolve_close_reason("hold_timeout")
        assert close_reason == "external_close"

    def test_new_pnl_value(self):
        """PnL using SL price: 115 * (0.05109 - 0.05213) = -1.196 USDT."""
        exit_price = _resolve_exit_price(
            last_price=self.LAST_PRICE,
            stop_loss_price=self.SL_PRICE,
        )
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=self.QTY,
            entry_price=self.ENTRY,
            exit_price=exit_price,
            contract_size=self.CONTRACT_SIZE,
        )
        expected = 115.0 * (0.05109 - 0.05213)   # ≈ -1.196
        assert pnl == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# SHORT position: price spikes up, SL was above entry
# ---------------------------------------------------------------------------

class TestShortExternalClose:
    ENTRY = 1.200
    LAST_PRICE = 1.195   # price fell back after spike
    SL_PRICE = 1.240     # SL above entry for a short
    QTY = 10.0

    def test_short_old_logic_produces_fake_profit(self):
        """last_price < entry for short → old logic shows profit even if liquidated."""
        pnl = pnl_linear_usdt(
            side_open="sell",
            qty_base=self.QTY,
            entry_price=self.ENTRY,
            exit_price=self.LAST_PRICE,
            contract_size=1.0,
        )
        assert pnl > 0

    def test_short_new_logic_produces_loss(self):
        exit_price = _resolve_exit_price(
            last_price=self.LAST_PRICE,
            stop_loss_price=self.SL_PRICE,
        )
        pnl = pnl_linear_usdt(
            side_open="sell",
            qty_base=self.QTY,
            entry_price=self.ENTRY,
            exit_price=exit_price,
            contract_size=1.0,
        )
        assert pnl < 0, f"Expected loss for short liquidation, got {pnl}"
