"""
TDD tests: stop-loss price must be clamped to fire BEFORE exchange liquidation.

Root cause:
  Gate.io maintenance_rate for D/USDT = 0.04 (4%).
  At 20x isolated, initial margin = 5%, maintenance = 4% → liquidation at +1% move.
  Our loss_fraction=0.45 → SL at 2.25% — beyond the liquidation point.
  Position gets liquidated before the SL triggers.

Fix:
  compute_stop_loss_price accepts maintenance_rate=0.0.
  When > 0, clamp the move_pct to:
    max_safe_move = (1/leverage - maintenance_rate) * SAFETY_FACTOR (0.8)
  Use min(fraction_based_move, max_safe_move).

Safety factor 0.8 = SL fires at 80% of the distance to liquidation,
leaving 20% buffer for slippage / mark-price divergence.
"""
from __future__ import annotations

import pytest

from core.risk import compute_stop_loss_price

# Gate.io real-world values
D_USDT_MM   = 0.04    # D/USDT maintenance_rate
STO_MM       = 0.025  # STO / DRIFT / RLS maintenance_rate
BTC_MM       = 0.003  # BTC/USDT maintenance_rate (low — not a constraint)
SAFETY       = 0.8    # internal safety factor


# ---------------------------------------------------------------------------
# compute_stop_loss_price: maintenance_rate parameter
# ---------------------------------------------------------------------------

class TestNoMaintenanceRate:
    """maintenance_rate=0 (default) preserves existing behaviour."""

    def test_short_no_mm(self):
        sl = compute_stop_loss_price(1.0, 20.0, "sell", loss_fraction=0.45,
                                     maintenance_rate=0.0)
        assert sl == pytest.approx(1.0 * (1 + 0.45 / 20))

    def test_long_no_mm(self):
        sl = compute_stop_loss_price(1.0, 20.0, "buy", loss_fraction=0.45,
                                     maintenance_rate=0.0)
        assert sl == pytest.approx(1.0 * (1 - 0.45 / 20))

    def test_default_param_unchanged(self):
        sl_default = compute_stop_loss_price(1.0, 50.0, "buy", loss_fraction=0.65)
        sl_explicit = compute_stop_loss_price(1.0, 50.0, "buy", loss_fraction=0.65,
                                               maintenance_rate=0.0)
        assert sl_default == pytest.approx(sl_explicit)


class TestDUSDTLiquidationClamp:
    """D/USDT (mm=4%) at 20x: loss_fraction SL is beyond liquidation → must clamp."""

    ENTRY = 0.016144
    LEV   = 20.0
    FRAC  = 0.45

    def _safe_move(self) -> float:
        return (1.0 / self.LEV - D_USDT_MM) * SAFETY  # (5%-4%)*0.8 = 0.8%

    def test_short_sl_is_closer_than_fraction_based(self):
        sl_clamped = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=D_USDT_MM,
        )
        sl_unclamped = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=0.0,
        )
        # Clamped SL must be closer to entry (smaller than unclamped for SHORT)
        assert sl_clamped < sl_unclamped, (
            f"Clamped SL {sl_clamped} should be < unclamped {sl_unclamped} for SHORT"
        )

    def test_short_sl_below_liquidation_price(self):
        sl = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=D_USDT_MM,
        )
        liq_price = self.ENTRY * (1 + 1.0 / self.LEV)  # rough liquidation
        assert sl < liq_price, (
            f"SL {sl:.8f} must be below liquidation {liq_price:.8f}"
        )

    def test_short_sl_at_80pct_of_liq_distance(self):
        """SL should be entry * (1 + safe_move)."""
        expected_sl = self.ENTRY * (1 + self._safe_move())
        sl = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=D_USDT_MM,
        )
        assert sl == pytest.approx(expected_sl, rel=1e-6)

    def test_long_sl_at_80pct_of_liq_distance(self):
        expected_sl = self.ENTRY * (1 - self._safe_move())
        sl = compute_stop_loss_price(
            self.ENTRY, self.LEV, "buy", self.FRAC, maintenance_rate=D_USDT_MM,
        )
        assert sl == pytest.approx(expected_sl, rel=1e-6)


class TestSTOLiquidationClamp:
    """STO/DRIFT/RLS (mm=2.5%) at 20x: loss_fraction=0.45 gives 2.25% but safe is 2.0% → clamp."""

    ENTRY = 0.12
    LEV   = 20.0
    FRAC  = 0.45

    def _safe_move(self) -> float:
        return (1.0 / self.LEV - STO_MM) * SAFETY  # (5%-2.5%)*0.8 = 2.0%

    def test_fraction_based_exceeds_safe_distance(self):
        """Sanity-check: unclamped move (2.25%) > safe move (2.0%)."""
        fraction_move = self.FRAC / self.LEV
        assert fraction_move > self._safe_move(), (
            f"Unclamped {fraction_move:.4f} should exceed safe {self._safe_move():.4f}"
        )

    def test_short_sl_clamped_to_safe_distance(self):
        expected_sl = self.ENTRY * (1 + self._safe_move())
        sl = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=STO_MM,
        )
        assert sl == pytest.approx(expected_sl, rel=1e-6)


class TestBTCNoClamp:
    """BTC (mm=0.3%) at 20x: fraction-based SL (2.25%) is tighter than safe (3.76%) → no clamp."""

    ENTRY = 100_000.0
    LEV   = 20.0
    FRAC  = 0.45

    def _safe_move(self) -> float:
        return (1.0 / self.LEV - BTC_MM) * SAFETY  # (5%-0.3%)*0.8 = 3.76%

    def test_fraction_based_is_tighter_than_safe(self):
        fraction_move = self.FRAC / self.LEV
        assert fraction_move < self._safe_move()

    def test_short_sl_not_clamped(self):
        """BTC SL should equal the fraction-based SL (no clamping needed)."""
        sl_with_mm = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=BTC_MM,
        )
        sl_without_mm = compute_stop_loss_price(
            self.ENTRY, self.LEV, "sell", self.FRAC, maintenance_rate=0.0,
        )
        assert sl_with_mm == pytest.approx(sl_without_mm, rel=1e-9)


class TestMaintenanceRateEdgeCases:
    def test_very_high_mm_rate_still_positive_sl_distance(self):
        """Even if mm_rate is very close to initial_margin, SL stays sensible."""
        sl = compute_stop_loss_price(1.0, 10.0, "sell", 0.5, maintenance_rate=0.09)
        # initial=10%, mm=9% → safe_move=(10%-9%)*80%=0.8%
        # SL should be above entry for SHORT
        assert sl > 1.0

    def test_mm_rate_zero_initial_margin_edge(self):
        """maintenance_rate=0 always falls back to fraction-based."""
        sl = compute_stop_loss_price(1.0, 20.0, "buy", 0.45, maintenance_rate=0.0)
        assert sl == pytest.approx(1.0 * (1 - 0.45 / 20))
