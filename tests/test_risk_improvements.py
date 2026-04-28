"""
TDD tests for risk improvements:
  - Stop-loss at 65% of margin (not 99% near liquidation)
  - Stop-loss at 90% of margin for high-leverage coins (>= 100x)
  - Phase 1: trailing_min_buffer_pct — prevents noise-tick false triggers
  - Phase 3: entry_follow_timeout_seconds — early exit when price doesn't follow signal
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from core.config import (
    BotConfig, DataConfig, ExchangeConfig, RiskConfig, RuntimeConfig, StrategyConfig,
)
from core.models import SymbolState
from core.risk import compute_stop_loss_price, update_trailing_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk_cfg(**kwargs) -> RiskConfig:
    defaults = dict(
        stop_loss_margin_fraction=0.65,
        stop_loss_high_lev_margin_fraction=0.90,
        stop_loss_high_lev_threshold=100.0,
        trailing_enabled=True,
        trailing_activation_pct=0.015,
        trailing_lock_pct=0.30,
        trailing_min_buffer_pct=0.0,
        tp1_margin_multiplier=4.0,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


def _make_state(symbol: str = "FORM/USDT:USDT", leverage: float = 50.0) -> SymbolState:
    s = SymbolState(symbol=symbol, tick_size=0.00001, max_leverage=leverage)
    s.last_price = 0.2375
    s.fair_price = 0.2375
    return s


def _make_cfg(strategy_kwargs=None, risk_kwargs=None) -> BotConfig:
    s_kwargs = dict(
        fill_timeout_ms=300, quote_size_usdt=5.0,
        hold_seconds=25.0, entry_follow_timeout_seconds=0.0,
        entry_follow_min_pct=0.003,
    )
    if strategy_kwargs:
        s_kwargs.update(strategy_kwargs)
    r_kwargs = dict(
        stop_loss_margin_fraction=0.65,
        stop_loss_high_lev_margin_fraction=0.90,
        stop_loss_high_lev_threshold=100.0,
        trailing_min_buffer_pct=0.0,
    )
    if risk_kwargs:
        r_kwargs.update(risk_kwargs)
    return BotConfig(
        strategy=StrategyConfig(**s_kwargs),
        risk=RiskConfig(**r_kwargs),
        data=DataConfig(),
        exchange=ExchangeConfig(exchange_id="gateio", market_type="swap"),
        runtime=RuntimeConfig(paper_mode=True),
    )


# ===========================================================================
# STOP-LOSS MARGIN FRACTION
# ===========================================================================

class TestStopLossMarginFraction:
    """New stop-loss formula: loss = margin_fraction / leverage from entry."""

    # --- LONG positions ---

    def test_stop_loss_65pct_long_at_50x(self):
        """LONG 50x: stop = entry * (1 - 0.65/50) = entry * 0.987"""
        sl = compute_stop_loss_price(0.2375, leverage=50.0, side="buy",
                                     loss_fraction=0.65)
        expected = 0.2375 * (1 - 0.65 / 50)
        assert abs(sl - expected) < 1e-9
        margin_loss = (0.2375 - sl) / 0.2375 * 50
        assert abs(margin_loss - 0.65) < 1e-6, f"Expected 65% margin loss, got {margin_loss*100:.2f}%"

    def test_stop_loss_90pct_long_at_125x(self):
        """LONG 125x: stop = entry * (1 - 0.90/125)"""
        sl = compute_stop_loss_price(1.0, leverage=125.0, side="buy",
                                     loss_fraction=0.90)
        expected = 1.0 * (1 - 0.90 / 125)
        assert abs(sl - expected) < 1e-9
        margin_loss = (1.0 - sl) / 1.0 * 125
        assert abs(margin_loss - 0.90) < 1e-6

    # --- SHORT positions ---

    def test_stop_loss_65pct_short_at_50x(self):
        """SHORT 50x: stop = entry * (1 + 0.65/50) = entry * 1.013"""
        sl = compute_stop_loss_price(0.2375, leverage=50.0, side="sell",
                                     loss_fraction=0.65)
        expected = 0.2375 * (1 + 0.65 / 50)
        assert abs(sl - expected) < 1e-9
        margin_loss = (sl - 0.2375) / 0.2375 * 50
        assert abs(margin_loss - 0.65) < 1e-6

    def test_stop_loss_90pct_short_at_125x(self):
        """SHORT 125x: stop = entry * (1 + 0.90/125)"""
        sl = compute_stop_loss_price(100.0, leverage=125.0, side="sell",
                                     loss_fraction=0.90)
        expected = 100.0 * (1 + 0.90 / 125)
        assert abs(sl - expected) < 1e-9

    def test_stop_is_tighter_than_before(self):
        """65% stop must be closer to entry than old 99% (~liq) stop."""
        sl_new = compute_stop_loss_price(1.0, 50.0, "buy", loss_fraction=0.65)
        sl_old = compute_stop_loss_price(1.0, 50.0, "buy", loss_fraction=0.99)
        assert sl_new > sl_old, "65% stop must be closer to entry (higher price) than 99% stop for LONG"

    def test_stop_price_short_is_above_entry(self):
        sl = compute_stop_loss_price(0.2375, 50.0, "sell", loss_fraction=0.65)
        assert sl > 0.2375, "SHORT stop must be above entry"

    def test_stop_price_long_is_below_entry(self):
        sl = compute_stop_loss_price(0.2375, 50.0, "buy", loss_fraction=0.65)
        assert sl < 0.2375, "LONG stop must be below entry"

    def test_default_loss_fraction_is_65pct(self):
        """Default parameter should give 65% loss."""
        sl = compute_stop_loss_price(1.0, 50.0, "buy")
        expected = compute_stop_loss_price(1.0, 50.0, "buy", loss_fraction=0.65)
        assert abs(sl - expected) < 1e-9


class TestRiskManagerStopLossLeverageRouting:
    """RiskManager.mark_opened selects fraction by leverage threshold."""

    def _make_manager(self, **risk_kwargs):
        from core.risk import RiskManager
        cfg = _make_cfg(risk_kwargs=risk_kwargs)
        return RiskManager(cfg.risk, cooldown_seconds=10.0, start_balance=100.0)

    def test_50x_uses_standard_fraction(self):
        mgr = self._make_manager(
            stop_loss_margin_fraction=0.65,
            stop_loss_high_lev_margin_fraction=0.90,
            stop_loss_high_lev_threshold=100.0,
        )
        state = _make_state(leverage=50.0)
        state.entry_price = 1.0
        mgr.mark_opened(state, "buy", qty=100.0, hold_seconds=25.0)

        expected = compute_stop_loss_price(1.0, 50.0, "buy", loss_fraction=0.65)
        assert abs(state.stop_loss_price - expected) < 1e-9, (
            f"50x should use 65% fraction, got {state.stop_loss_price}"
        )

    def test_125x_uses_high_lev_fraction(self):
        mgr = self._make_manager(
            stop_loss_margin_fraction=0.65,
            stop_loss_high_lev_margin_fraction=0.90,
            stop_loss_high_lev_threshold=100.0,
        )
        state = _make_state(leverage=125.0)
        state.entry_price = 1.0
        mgr.mark_opened(state, "sell", qty=100.0, hold_seconds=25.0)

        expected = compute_stop_loss_price(1.0, 125.0, "sell", loss_fraction=0.90)
        assert abs(state.stop_loss_price - expected) < 1e-9, (
            f"125x should use 90% fraction, got {state.stop_loss_price}"
        )

    def test_100x_uses_high_lev_fraction(self):
        """Threshold is inclusive: 100x >= 100 -> high lev."""
        mgr = self._make_manager(
            stop_loss_margin_fraction=0.65,
            stop_loss_high_lev_margin_fraction=0.90,
            stop_loss_high_lev_threshold=100.0,
        )
        state = _make_state(leverage=100.0)
        state.entry_price = 1.0
        mgr.mark_opened(state, "buy", qty=100.0, hold_seconds=25.0)

        expected = compute_stop_loss_price(1.0, 100.0, "buy", loss_fraction=0.90)
        assert abs(state.stop_loss_price - expected) < 1e-9

    def test_99x_uses_standard_fraction(self):
        """99x < 100 threshold -> standard 65%."""
        mgr = self._make_manager(
            stop_loss_margin_fraction=0.65,
            stop_loss_high_lev_margin_fraction=0.90,
            stop_loss_high_lev_threshold=100.0,
        )
        state = _make_state(leverage=99.0)
        state.entry_price = 1.0
        mgr.mark_opened(state, "buy", qty=100.0, hold_seconds=25.0)

        expected = compute_stop_loss_price(1.0, 99.0, "buy", loss_fraction=0.65)
        assert abs(state.stop_loss_price - expected) < 1e-9


# ===========================================================================
# PHASE 1: TRAILING MIN BUFFER
# ===========================================================================

class TestTrailingMinBuffer:
    """trailing_min_buffer_pct prevents noise-tick false triggers."""

    def _open_position(self, state: SymbolState, entry: float, side: str) -> None:
        state.entry_price = entry
        state.side = side
        state.position_qty = 1000.0
        state.trailing_active = False
        state.best_pnl_pct = 0.0
        state.trailing_stop_price = None

    def _tick(self, state: SymbolState, price: float, cfg: RiskConfig) -> str:
        state.last_price = price
        return update_trailing_state(state, cfg)

    # --- Core: noise tick does NOT trigger with buffer ---

    def test_noise_tick_does_not_trigger_short_with_buffer(self):
        """
        FORM scenario: best_move=3.1%, trail=0.23528, noise tick at 0.23530.
        Without buffer: triggered (0.23530 >= 0.23528).
        With buffer 0.5%: effective_trail = max(0.23528, 0.23530*1.005) = 0.23648
                          0.23530 < 0.23648 -> NOT triggered.
        """
        cfg = _make_risk_cfg(
            trailing_activation_pct=0.015,
            trailing_lock_pct=0.30,
            trailing_min_buffer_pct=0.005,
        )
        entry = 0.2375
        state = _make_state(leverage=50.0)
        self._open_position(state, entry, "sell")

        # Drive price down to activate trailing and reach best_move=3.1%
        best_price = entry * (1 - 0.031)
        for p in [entry * 0.99, entry * 0.98, entry * 0.975, entry * 0.970, best_price]:
            self._tick(state, p, cfg)

        assert state.trailing_active, "Trailing must be active"
        assert state.trailing_stop_price is not None

        # Noise tick: tiny uptick barely above trail
        action = self._tick(state, 0.23530, cfg)  # barely above trail 0.23528
        assert action != "trailing_stop_hit", (
            "Noise tick (0.02% above trail) must NOT trigger with min_buffer=0.5%"
        )

    def test_real_pullback_triggers_short_with_buffer(self):
        """Price rising well above (trail + buffer) must trigger the stop.

        SHORT trail is ABOVE current price.  A big enough reversal pushes
        last above the effective trail.  With buffer=0.5%, the trigger is:
            effective_trail = max(pure_trail, last * 1.005)
        The stop fires when last >= effective_trail, i.e. when last >= pure_trail
        AND last >= last * 1.005.  The second condition can only be met trivially
        (last >= last * 1.005 is never true), so the buffer only applies when the
        price is BELOW the pure trail.  When last rises above the pure trail the
        stop fires immediately (pure_trail < last).

        Concretely: drive price to best, then pull back above the pure trail.
        """
        cfg = _make_risk_cfg(
            trailing_activation_pct=0.015,
            trailing_lock_pct=0.30,
            trailing_min_buffer_pct=0.005,
        )
        entry = 0.2375
        state = _make_state(leverage=50.0)
        self._open_position(state, entry, "sell")

        # Drive to best_move=3.1%
        best_price = entry * (1 - 0.031)
        for p in [entry * 0.99, entry * 0.98, entry * 0.975, entry * 0.970, best_price]:
            self._tick(state, p, cfg)

        assert state.trailing_active
        trail = state.trailing_stop_price
        assert trail is not None

        # Price reverses cleanly above pure trail (genuine reversal, not noise)
        # trail ≈ 0.23529, send price to trail + 0.002 (well above)
        reversal_price = trail + 0.002
        action = self._tick(state, reversal_price, cfg)
        assert action == "trailing_stop_hit", (
            f"Price {reversal_price:.5f} is above trail {trail:.5f} — must trigger"
        )

    def test_buffer_zero_keeps_original_behaviour(self):
        """min_buffer=0 must reproduce original hair-trigger behaviour."""
        cfg = _make_risk_cfg(
            trailing_activation_pct=0.015,
            trailing_lock_pct=0.30,
            trailing_min_buffer_pct=0.0,
        )
        entry = 0.2375
        state = _make_state(leverage=50.0)
        self._open_position(state, entry, "sell")

        # Same path as above
        best_price = entry * (1 - 0.031)
        for p in [entry * 0.99, entry * 0.98, entry * 0.975, entry * 0.970, best_price]:
            self._tick(state, p, cfg)

        assert state.trailing_active
        trail = state.trailing_stop_price
        assert trail is not None

        # Tick barely above trail (same as FORM noise tick)
        action = self._tick(state, trail + 0.00002, cfg)
        assert action == "trailing_stop_hit", (
            "Without buffer, tiny tick above trail must still trigger"
        )

    def test_buffer_does_not_affect_trail_price_stored(self):
        """trailing_stop_price stores the pure computed value (not buffer-inflated)."""
        cfg = _make_risk_cfg(
            trailing_activation_pct=0.015,
            trailing_lock_pct=0.30,
            trailing_min_buffer_pct=0.005,
        )
        entry = 1.0
        state = _make_state(leverage=50.0)
        self._open_position(state, entry, "sell")

        for p in [0.985, 0.975, 0.970]:
            self._tick(state, p, cfg)

        assert state.trailing_active
        pure_trail = entry - (entry - 0.970) * 0.30
        assert abs(state.trailing_stop_price - pure_trail) < 1e-8, (
            "trailing_stop_price must store pure computed value, not buffer-inflated"
        )

    def test_buffer_applies_to_long_positions(self):
        """For LONG, buffer prevents trigger when tiny dip barely touches trail."""
        cfg = _make_risk_cfg(
            trailing_activation_pct=0.015,
            trailing_lock_pct=0.30,
            trailing_min_buffer_pct=0.005,
        )
        entry = 1.0
        state = _make_state(leverage=50.0)
        self._open_position(state, entry, "buy")

        # Drive up to best_move=3%
        for p in [1.01, 1.02, 1.025, 1.030]:
            self._tick(state, p, cfg)

        assert state.trailing_active
        trail = state.trailing_stop_price
        assert trail is not None

        # Tiny dip just at trail: last = trail - 0.00001
        action = self._tick(state, trail - 0.00001, cfg)
        assert action != "trailing_stop_hit", (
            "Tiny dip barely below trail must NOT trigger for LONG with min_buffer=0.5%"
        )


# ===========================================================================
# PHASE 3: ENTRY FOLLOW TIMEOUT
# ===========================================================================

class TestEntryFollowTimeout:
    """entry_follow_timeout_seconds: close early if price doesn't follow signal."""

    def _setup_open_position(self, state: SymbolState, entry: float,
                              side: str, entry_ts: float) -> None:
        state.entry_price = entry
        state.side = side
        state.position_qty = 1000.0
        state.trailing_active = False
        state.best_pnl_pct = 0.0
        state.trailing_stop_price = None
        state.entry_ts_monotonic = entry_ts
        state.close_queued = False
        state.close_reason = ""

    def _check_follow(self, state: SymbolState, cfg: BotConfig, now: float) -> bool:
        """Simulate the engine's entry_follow_timeout check. Returns True if close triggered."""
        from core.engine import TradingEngine
        # Call the private method directly
        engine = TradingEngine.__new__(TradingEngine)
        engine.cfg = cfg
        return engine._check_entry_follow_timeout(state, now)

    def test_closes_when_price_not_following_after_timeout(self):
        """
        SHORT signal: price must fall by entry_follow_min_pct within timeout.
        If it hasn't, close the position.
        """
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=10.0,
            entry_follow_min_pct=0.003,
        ))
        entry = 0.2375
        state = _make_state()
        entry_ts = 1000.0
        self._setup_open_position(state, entry, "sell", entry_ts)

        # Price barely moved (only 0.1% down vs required 0.3%)
        state.last_price = entry * (1 - 0.001)
        now = entry_ts + 10.5  # past timeout

        triggered = self._check_follow(state, cfg, now)
        assert triggered is True, (
            "Must close early when price moved only 0.1% vs required 0.3% after timeout"
        )

    def test_does_not_close_when_price_following(self):
        """Price moved enough in our direction -> no early close."""
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=10.0,
            entry_follow_min_pct=0.003,
        ))
        entry = 0.2375
        state = _make_state()
        entry_ts = 1000.0
        self._setup_open_position(state, entry, "sell", entry_ts)

        # Price moved 0.5% down (> 0.3% threshold)
        state.last_price = entry * (1 - 0.005)
        now = entry_ts + 10.5

        triggered = self._check_follow(state, cfg, now)
        assert triggered is False, (
            "Must NOT close when price moved 0.5% (above 0.3% threshold)"
        )

    def test_does_not_close_before_timeout(self):
        """Within timeout window: no early close even if price not moving."""
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=10.0,
            entry_follow_min_pct=0.003,
        ))
        entry = 0.2375
        state = _make_state()
        entry_ts = 1000.0
        self._setup_open_position(state, entry, "sell", entry_ts)

        state.last_price = entry  # price flat
        now = entry_ts + 5.0  # within timeout

        triggered = self._check_follow(state, cfg, now)
        assert triggered is False, "Must NOT close before timeout period"

    def test_does_not_trigger_when_disabled(self):
        """entry_follow_timeout_seconds=0 disables the feature."""
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=0.0,
            entry_follow_min_pct=0.003,
        ))
        entry = 0.2375
        state = _make_state()
        entry_ts = 1000.0
        self._setup_open_position(state, entry, "sell", entry_ts)

        state.last_price = entry  # price flat, would normally trigger
        now = entry_ts + 60.0   # way past timeout

        triggered = self._check_follow(state, cfg, now)
        assert triggered is False, "Feature must be disabled when timeout=0"

    def test_does_not_trigger_when_trailing_already_active(self):
        """If trailing is active, position is already profitable — skip follow check."""
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=10.0,
            entry_follow_min_pct=0.003,
        ))
        entry = 0.2375
        state = _make_state()
        entry_ts = 1000.0
        self._setup_open_position(state, entry, "sell", entry_ts)
        state.trailing_active = True  # already in profit
        state.last_price = entry  # flat (irrelevant — trailing manages exit)
        now = entry_ts + 15.0

        triggered = self._check_follow(state, cfg, now)
        assert triggered is False, "Must not early-close when trailing is already active"

    def test_long_side_checks_upward_movement(self):
        """For LONG, price must rise by entry_follow_min_pct."""
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=10.0,
            entry_follow_min_pct=0.003,
        ))
        state = _make_state()
        entry_ts = 1000.0
        entry = 1.0
        self._setup_open_position(state, entry, "buy", entry_ts)

        # Price barely moved up (0.1% vs 0.3%)
        state.last_price = entry * 1.001
        now = entry_ts + 10.5

        triggered = self._check_follow(state, cfg, now)
        assert triggered is True, "LONG: close when upward movement insufficient"

    def test_long_side_no_close_when_price_up_enough(self):
        cfg = _make_cfg(strategy_kwargs=dict(
            entry_follow_timeout_seconds=10.0,
            entry_follow_min_pct=0.003,
        ))
        state = _make_state()
        entry_ts = 1000.0
        entry = 1.0
        self._setup_open_position(state, entry, "buy", entry_ts)

        state.last_price = entry * 1.005  # 0.5% up > 0.3%
        now = entry_ts + 10.5

        triggered = self._check_follow(state, cfg, now)
        assert triggered is False
