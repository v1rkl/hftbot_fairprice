"""
TDD tests for 3-tier leverage TP / trailing / stop-loss system.

Tiers:
  tier1: leverage < lev_tier1_max (default 30)  → low TP, tight SL, early trailing
  tier2: leverage < lev_tier2_max (default 50)  → mid TP, mid SL, mid trailing
  tier3: leverage >= lev_tier2_max              → standard TP/SL/trailing (existing behaviour)
"""
from __future__ import annotations

import pytest

from core.config import RiskConfig
from core.models import SymbolState
from core.risk import (
    _sl_fraction,
    _tp_multiplier,
    _trailing_activation,
    update_trailing_state,
)
from core.risk import RiskManager
from core.config import _validate, BotConfig, DataConfig, ExchangeConfig, RuntimeConfig, StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> RiskConfig:
    defaults = dict(
        daily_loss_limit_pct=30.0,
        position_size_pct=0.10,
        stop_loss_margin_fraction=0.65,
        stop_loss_margin_fraction_tier1=0.45,
        stop_loss_margin_fraction_tier2=0.55,
        stop_loss_high_lev_margin_fraction=0.90,
        stop_loss_high_lev_threshold=100.0,
        trailing_enabled=True,
        trailing_activation_pct=0.015,
        trailing_activation_pct_tier1=0.010,
        trailing_activation_pct_tier2=0.012,
        trailing_lock_pct=0.30,
        trailing_min_buffer_pct=0.0,
        tp1_margin_multiplier=4.0,
        tp1_margin_multiplier_tier1=1.6,  # 60% margin gain
        tp1_margin_multiplier_tier2=2.0,  # 100% margin gain
        lev_tier1_max=30.0,
        lev_tier2_max=50.0,
        symbol_max_consecutive_losses=2,
        symbol_ban_duration_seconds=3600.0,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _state(leverage: float, entry: float = 1.0, side: str = "buy") -> SymbolState:
    s = SymbolState(symbol="TEST/USDT:USDT", tick_size=0.0001, max_leverage=leverage)
    s.entry_price = entry
    s.last_price = entry
    s.fair_price = entry
    s.side = side
    s.position_qty = 10.0
    return s


# ---------------------------------------------------------------------------
# _tp_multiplier — unit tests
# ---------------------------------------------------------------------------

class TestTpMultiplier:
    def test_tier1_uses_tier1_multiplier(self):
        cfg = _cfg()
        assert _tp_multiplier(cfg, 20.0) == 1.6

    def test_tier1_upper_boundary_exclusive(self):
        """leverage=29.9 is still tier1."""
        cfg = _cfg()
        assert _tp_multiplier(cfg, 29.9) == 1.6

    def test_tier2_at_lower_boundary(self):
        """leverage=30.0 is tier2 (not tier1 because tier1 is strictly < 30)."""
        cfg = _cfg()
        assert _tp_multiplier(cfg, 30.0) == 2.0

    def test_tier2_mid(self):
        cfg = _cfg()
        assert _tp_multiplier(cfg, 40.0) == 2.0

    def test_tier2_upper_boundary_exclusive(self):
        cfg = _cfg()
        assert _tp_multiplier(cfg, 49.9) == 2.0

    def test_tier3_at_lower_boundary(self):
        """leverage=50.0 is tier3 (existing behaviour)."""
        cfg = _cfg()
        assert _tp_multiplier(cfg, 50.0) == 4.0

    def test_tier3_high_leverage(self):
        cfg = _cfg()
        assert _tp_multiplier(cfg, 125.0) == 4.0

    def test_custom_tier_boundaries(self):
        cfg = _cfg(lev_tier1_max=25.0, lev_tier2_max=75.0)
        assert _tp_multiplier(cfg, 20.0) == 1.6   # tier1
        assert _tp_multiplier(cfg, 25.0) == 2.0   # tier2
        assert _tp_multiplier(cfg, 74.9) == 2.0   # still tier2
        assert _tp_multiplier(cfg, 75.0) == 4.0   # tier3


# ---------------------------------------------------------------------------
# _trailing_activation — unit tests
# ---------------------------------------------------------------------------

class TestTrailingActivation:
    def test_tier1_activates_earlier(self):
        cfg = _cfg()
        assert _trailing_activation(cfg, 20.0) == 0.010

    def test_tier2_activates_mid(self):
        cfg = _cfg()
        assert _trailing_activation(cfg, 40.0) == 0.012

    def test_tier3_uses_default(self):
        cfg = _cfg()
        assert _trailing_activation(cfg, 50.0) == 0.015

    def test_tier1_boundary(self):
        cfg = _cfg()
        assert _trailing_activation(cfg, 29.9) == 0.010
        assert _trailing_activation(cfg, 30.0) == 0.012


# ---------------------------------------------------------------------------
# _sl_fraction — unit tests
# ---------------------------------------------------------------------------

class TestSlFraction:
    def test_tier1_tightest_sl(self):
        cfg = _cfg()
        assert _sl_fraction(cfg, 20.0) == 0.45

    def test_tier2_mid_sl(self):
        cfg = _cfg()
        assert _sl_fraction(cfg, 40.0) == 0.55

    def test_tier3_standard_sl(self):
        cfg = _cfg()
        assert _sl_fraction(cfg, 50.0) == 0.65

    def test_high_lev_overrides_all_tiers(self):
        """stop_loss_high_lev_threshold=100 overrides tier logic for 100x+."""
        cfg = _cfg()
        assert _sl_fraction(cfg, 100.0) == 0.90
        assert _sl_fraction(cfg, 125.0) == 0.90

    def test_tier1_boundary(self):
        cfg = _cfg()
        assert _sl_fraction(cfg, 29.9) == 0.45
        assert _sl_fraction(cfg, 30.0) == 0.55

    def test_gap_range_between_tier2_max_and_high_lev_threshold(self):
        """leverage in [50, 100) falls through to standard tier3 fraction (0.65)."""
        cfg = _cfg()
        assert _sl_fraction(cfg, 75.0) == 0.65
        assert _sl_fraction(cfg, 50.0) == 0.65
        assert _sl_fraction(cfg, 99.9) == 0.65


# ---------------------------------------------------------------------------
# update_trailing_state — integration tests (tier-aware activation)
# ---------------------------------------------------------------------------

class TestUpdateTrailingStateTiered:
    def test_tier1_trailing_activates_at_tier1_pct(self):
        """At 20x, trailing activates at 1% price move (not 1.5%)."""
        cfg = _cfg()
        state = _state(leverage=20.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        # 1.0% move — below tier3 activation (1.5%) but above tier1 (1.0%)
        state.last_price = 1.010
        result = update_trailing_state(state, cfg)
        assert state.trailing_active is True

    def test_tier3_trailing_not_activated_below_tier3_pct(self):
        """At 50x, trailing should NOT activate at 1.0% move (needs 1.5%)."""
        cfg = _cfg()
        state = _state(leverage=50.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        state.last_price = 1.010  # only 1% move
        result = update_trailing_state(state, cfg)
        assert state.trailing_active is False

    def test_tier1_tp_fires_at_tier1_multiplier(self):
        """At 20x with TP=1.6, need 3% price move → margin_mult = 1 + 20*0.03 = 1.6."""
        cfg = _cfg()
        state = _state(leverage=20.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        state.last_price = 1.03  # margin_mult = 1.6 >= 1.6 → fires
        result = update_trailing_state(state, cfg)
        assert result == "tp1_full"

    def test_tier1_tp_does_not_fire_below_threshold(self):
        """At 20x, 2% move gives margin_mult=1.4 < 1.6 → no TP."""
        cfg = _cfg()
        state = _state(leverage=20.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        state.last_price = 1.02  # margin_mult = 1 + 20*0.02 = 1.4 < 1.6
        result = update_trailing_state(state, cfg)
        assert result == "none"

    def test_tier3_tp_does_not_fire_at_tier1_price_move(self):
        """At 50x the same 3% move (margin_mult=2.5) should NOT hit TP=4.0."""
        cfg = _cfg()
        state = _state(leverage=50.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        state.last_price = 1.03  # margin_mult = 1 + 50*0.03 = 2.5 < 4.0
        result = update_trailing_state(state, cfg)
        assert result == "none"

    def test_tier2_tp_fires_at_tier2_multiplier(self):
        """At 40x with TP=2.0, 2.6% price move gives margin_mult=2.04 >= 2.0."""
        cfg = _cfg()
        state = _state(leverage=40.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        # Use 2.6% to avoid floating-point edge at exactly 2.0
        state.last_price = 1.026  # margin_mult = 1 + 40*0.026 = 2.04 >= 2.0
        result = update_trailing_state(state, cfg)
        assert result == "tp1_full"

    def test_tier2_tp_does_not_fire_below_threshold(self):
        """At 40x, 1% price move gives margin_mult=1.4 < 2.0 → no TP."""
        cfg = _cfg()
        state = _state(leverage=40.0, entry=1.0, side="buy")
        state.position_qty = 10.0
        state.last_price = 1.01  # margin_mult = 1 + 40*0.01 = 1.4 < 2.0
        result = update_trailing_state(state, cfg)
        assert result == "none"

    def test_sell_side_tier1_tp(self):
        """Tier1 TP works correctly for short positions (3% drop)."""
        cfg = _cfg()
        state = _state(leverage=20.0, entry=1.0, side="sell")
        state.position_qty = 10.0
        state.last_price = 0.97  # 3% drop → margin_mult = 1 + 20*0.03 = 1.6 >= 1.6
        result = update_trailing_state(state, cfg)
        assert result == "tp1_full"

    def test_sell_side_tier2_trailing_activates(self):
        """Tier2 trailing activates at 1.2% for short position."""
        cfg = _cfg()
        state = _state(leverage=40.0, entry=1.0, side="sell")
        state.position_qty = 10.0
        # 1.2% drop — above tier2 activation (1.2%), below tier3 (1.5%)
        state.last_price = 0.988
        update_trailing_state(state, cfg)
        assert state.trailing_active is True

    def test_tier3_trailing_not_activated_at_tier2_pct(self):
        """At 50x, 1.2% move does NOT activate trailing (needs 1.5%)."""
        cfg = _cfg()
        state = _state(leverage=50.0, entry=1.0, side="sell")
        state.position_qty = 10.0
        state.last_price = 0.988  # only 1.2% move
        update_trailing_state(state, cfg)
        assert state.trailing_active is False


# ---------------------------------------------------------------------------
# RiskManager.mark_opened — tier-aware stop loss
# ---------------------------------------------------------------------------

class TestMarkOpenedTieredSL:
    def test_tier1_sl_price_uses_tier1_fraction(self):
        """At 20x, SL fraction 0.45 → stop is 2.25% from entry."""
        cfg = _cfg()
        rm = RiskManager(cfg, cooldown_seconds=30.0, start_balance=100.0)
        state = _state(leverage=20.0, entry=1.0, side="buy")
        state.position_qty = 0.0  # mark_opened sets it
        state.stop_loss_price = None
        rm.mark_opened(state, "buy", 10.0, 25.0)
        # expected stop = 1.0 * (1 - 0.45/20) = 1 - 0.0225 = 0.9775
        assert state.stop_loss_price is not None
        assert abs(state.stop_loss_price - 0.9775) < 1e-8

    def test_tier2_sl_price_uses_tier2_fraction(self):
        """At 40x, SL fraction 0.55 → stop is 1.375% from entry."""
        cfg = _cfg()
        rm = RiskManager(cfg, cooldown_seconds=30.0, start_balance=100.0)
        state = _state(leverage=40.0, entry=1.0, side="buy")
        state.stop_loss_price = None
        rm.mark_opened(state, "buy", 10.0, 25.0)
        # expected stop = 1.0 * (1 - 0.55/40) = 1 - 0.01375 = 0.98625
        assert state.stop_loss_price is not None
        assert abs(state.stop_loss_price - 0.98625) < 1e-8

    def test_tier3_sl_price_uses_standard_fraction(self):
        """At 50x, SL fraction 0.65 → stop is 1.3% from entry."""
        cfg = _cfg()
        rm = RiskManager(cfg, cooldown_seconds=30.0, start_balance=100.0)
        state = _state(leverage=50.0, entry=1.0, side="buy")
        state.stop_loss_price = None
        rm.mark_opened(state, "buy", 10.0, 25.0)
        # expected stop = 1.0 * (1 - 0.65/50) = 1 - 0.013 = 0.987
        assert state.stop_loss_price is not None
        assert abs(state.stop_loss_price - 0.987) < 1e-8

    def test_high_lev_overrides_tiers(self):
        """At 125x, high-lev fraction 0.90 takes priority over all tiers."""
        cfg = _cfg()
        rm = RiskManager(cfg, cooldown_seconds=30.0, start_balance=100.0)
        state = _state(leverage=125.0, entry=1.0, side="sell")
        state.stop_loss_price = None
        rm.mark_opened(state, "sell", 10.0, 25.0)
        # expected stop = 1.0 * (1 + 0.90/125) = 1 + 0.0072 = 1.0072
        assert state.stop_loss_price is not None
        assert abs(state.stop_loss_price - 1.0072) < 1e-8

    def test_gap_range_75x_uses_standard_sl_fraction(self):
        """At 75x (gap range: >= lev_tier2_max=50, < high_lev_threshold=100), uses tier3 SL."""
        cfg = _cfg()
        rm = RiskManager(cfg, cooldown_seconds=30.0, start_balance=100.0)
        state = _state(leverage=75.0, entry=1.0, side="buy")
        state.stop_loss_price = None
        rm.mark_opened(state, "buy", 10.0, 25.0)
        # expected stop = 1.0 * (1 - 0.65/75) = 1 - 0.00867 = 0.99133
        assert state.stop_loss_price is not None
        assert abs(state.stop_loss_price - (1.0 - 0.65 / 75.0)) < 1e-8


# ---------------------------------------------------------------------------
# Config validation — tier parameter checks
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def _make_full_cfg(self, **risk_overrides) -> BotConfig:
        risk = _cfg(**risk_overrides)
        return BotConfig(
            strategy=StrategyConfig(
                fill_timeout_ms=300, quote_size_usdt=5.0, hold_seconds=10.0,
                fair_rise_threshold=0.035,
            ),
            data=DataConfig(),
            exchange=ExchangeConfig(min_max_leverage=20.0),
            runtime=RuntimeConfig(paper_mode=True),
            risk=risk,
        )

    def test_tp_multiplier_tier1_below_1_raises(self):
        with pytest.raises(ValueError, match="tp1_margin_multiplier_tier1 must be >= 1.0"):
            _validate(self._make_full_cfg(tp1_margin_multiplier_tier1=0.6))

    def test_tp_multiplier_tier2_below_1_raises(self):
        with pytest.raises(ValueError, match="tp1_margin_multiplier_tier2 must be >= 1.0"):
            _validate(self._make_full_cfg(tp1_margin_multiplier_tier2=0.9))

    def test_lev_tier2_max_below_tier1_raises(self):
        with pytest.raises(ValueError, match="lev_tier2_max"):
            _validate(self._make_full_cfg(lev_tier1_max=50.0, lev_tier2_max=30.0))

    def test_high_lev_threshold_below_tier2_max_raises(self):
        with pytest.raises(ValueError, match="stop_loss_high_lev_threshold"):
            _validate(self._make_full_cfg(lev_tier2_max=50.0, stop_loss_high_lev_threshold=40.0))

    def test_trailing_activation_tier1_zero_raises(self):
        with pytest.raises(ValueError, match="trailing_activation_pct_tier1"):
            _validate(self._make_full_cfg(trailing_activation_pct_tier1=0.0))

    def test_trailing_activation_tier2_above_1_raises(self):
        with pytest.raises(ValueError, match="trailing_activation_pct_tier2"):
            _validate(self._make_full_cfg(trailing_activation_pct_tier2=1.5))
