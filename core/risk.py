from __future__ import annotations

import logging
import time

from .config import RiskConfig
from .models import SymbolState


def compute_liquidation_price(entry_price: float, leverage: float, side: str) -> float:
    """Compute approximate liquidation price for isolated margin.

    LONG:  liq = entry * (1 - 1/leverage)
    SHORT: liq = entry * (1 + 1/leverage)
    """
    if leverage <= 0:
        logging.error(
            "compute_liquidation_price called with invalid leverage=%.6f — defaulting to 1.0",
            leverage,
        )
        leverage = 1.0
    if side == "buy":
        return entry_price * (1.0 - 1.0 / leverage)
    else:
        return entry_price * (1.0 + 1.0 / leverage)


_SL_LIQUIDATION_SAFETY = 0.8  # Place SL at 75% of distance to liquidation (0.6 was too tight — caused best=0.00% stops on low-lev microcaps; 0.8 was too loose → FOLKS liquidation)

def compute_stop_loss_price(
    entry_price: float,
    leverage: float,
    side: str,
    loss_fraction: float = 0.65,
    maintenance_rate: float = 0.0,
) -> float:
    """Stop-loss price that limits margin loss to loss_fraction of initial margin.

    move_pct = loss_fraction / leverage
    LONG:  stop = entry * (1 - move_pct)
    SHORT: stop = entry * (1 + move_pct)

    When maintenance_rate > 0 (fetched from exchange market data), the move_pct
    is clamped so the SL fires before the exchange liquidates the position:

      max_safe_move = (initial_margin_rate - maintenance_rate) * SAFETY_FACTOR
      move_pct = min(fraction_based_move, max_safe_move)

    Example: D/USDT mm=4%, 20x → initial=5%, safe=(5%-4%)*80%=0.8% → SL 0.8% away.
    Example: BTC mm=0.3%, 20x → safe=(5%-0.3%)*80%=3.76% > fraction 2.25% → no clamp.
    """
    if leverage <= 0:
        logging.error(
            "compute_stop_loss_price called with invalid leverage=%.6f — defaulting to 1.0",
            leverage,
        )
        leverage = 1.0
    move_pct = loss_fraction / leverage
    if maintenance_rate > 0:
        initial_margin_rate = 1.0 / leverage
        max_safe_move = (initial_margin_rate - maintenance_rate) * _SL_LIQUIDATION_SAFETY
        if max_safe_move > 0:
            move_pct = min(move_pct, max_safe_move)
    if side == "buy":
        return entry_price * (1.0 - move_pct)
    else:
        return entry_price * (1.0 + move_pct)


def compute_trailing_stop_price(
    entry_price: float, best_price: float, side: str, lock_pct: float,
) -> float:
    """Trailing stop that locks in lock_pct of peak profit from entry.

    LONG:  trailing_stop = entry + (best - entry) * lock_pct
    SHORT: trailing_stop = entry - (entry - best) * lock_pct
    """
    if side == "buy":
        return entry_price + (best_price - entry_price) * lock_pct
    else:
        return entry_price - (entry_price - best_price) * lock_pct


def compute_margin_multiplier(
    entry_price: float, current_price: float, side: str, leverage: float,
) -> float:
    """Compute current margin multiplier.

    margin_multiplier = 1 + leverage * pnl_pct_on_notional
    At 3x: margin has tripled. At 6x: margin has sextupled.
    """
    if entry_price <= 0:
        return 1.0
    if side == "buy":
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    return 1.0 + leverage * pnl_pct


def _tp_multiplier(cfg: RiskConfig, leverage: float) -> float:
    """Return the TP margin multiplier for the given leverage tier."""
    if leverage < cfg.lev_tier1_max:
        return cfg.tp1_margin_multiplier_tier1
    if leverage < cfg.lev_tier2_max:
        return cfg.tp1_margin_multiplier_tier2
    return cfg.tp1_margin_multiplier


def _trailing_activation(cfg: RiskConfig, leverage: float) -> float:
    """Return the trailing activation % threshold for the given leverage tier."""
    if leverage < cfg.lev_tier1_max:
        return cfg.trailing_activation_pct_tier1
    if leverage < cfg.lev_tier2_max:
        return cfg.trailing_activation_pct_tier2
    return cfg.trailing_activation_pct


def _sl_fraction(cfg: RiskConfig, leverage: float) -> float:
    """Return the stop-loss margin fraction for the given leverage tier.

    Priority: high-lev threshold overrides tier logic.
    """
    if leverage >= cfg.stop_loss_high_lev_threshold:
        return cfg.stop_loss_high_lev_margin_fraction
    if leverage < cfg.lev_tier1_max:
        return cfg.stop_loss_margin_fraction_tier1
    if leverage < cfg.lev_tier2_max:
        return cfg.stop_loss_margin_fraction_tier2
    return cfg.stop_loss_margin_fraction


def update_trailing_state(state: SymbolState, cfg: RiskConfig) -> str:
    """Update trailing stop state and return action to take.

    Called every tick (50ms) under lock, no I/O.
    Returns: "none", "trailing_stop_hit", "tp1_full"
    """
    if not cfg.trailing_enabled:
        return "none"
    entry = state.entry_price
    last = state.last_price
    side = state.side
    if entry is None or last is None or side is None or entry <= 0:
        return "none"

    # 1. Compute current PnL as price move fraction.
    if side == "buy":
        current_pnl_pct = (last - entry) / entry
    else:
        current_pnl_pct = (entry - last) / entry

    # 2. Track best PnL seen.
    if current_pnl_pct > state.best_pnl_pct:
        state.best_pnl_pct = current_pnl_pct

    # 3. Compute best price from best_pnl_pct.
    if side == "buy":
        best_price = entry * (1.0 + state.best_pnl_pct)
    else:
        best_price = entry * (1.0 - state.best_pnl_pct)

    # 4. Activation check: tier-aware price move threshold.
    leverage = max(state.max_leverage, 1.0)
    activation_pct = _trailing_activation(cfg, leverage)
    if state.best_pnl_pct >= activation_pct and not state.trailing_active:
        state.trailing_active = True
        state.trailing_stop_price = compute_trailing_stop_price(
            entry, best_price, side, cfg.trailing_lock_pct,
        )
        logging.info(
            "TRAILING ACTIVATED %s side=%s best_move=%.4f%% trail_stop=%.8f",
            state.symbol, side.upper(), state.best_pnl_pct * 100.0,
            state.trailing_stop_price,
        )

    # 5. Trailing update: recalculate stop (only moves in favorable direction).
    if state.trailing_active:
        new_stop = compute_trailing_stop_price(
            entry, best_price, side, cfg.trailing_lock_pct,
        )
        if side == "buy":
            if state.trailing_stop_price is None or new_stop > state.trailing_stop_price:
                state.trailing_stop_price = new_stop
        else:
            if state.trailing_stop_price is None or new_stop < state.trailing_stop_price:
                state.trailing_stop_price = new_stop

    # 6. Trailing stop hit check (evaluated before TP so close_reason is accurate).
    if state.trailing_active and state.trailing_stop_price is not None:
        # Phase 1: apply min_buffer so noise ticks don't trigger the stop.
        # The stored trailing_stop_price is the pure computed value; we derive
        # an effective trigger price that is at least min_buffer away from last.
        min_buf = cfg.trailing_min_buffer_pct if cfg.trailing_min_buffer_pct > 0 else 0.0
        if min_buf > 0:
            if side == "buy":
                # LONG: require price to drop min_buf% BELOW the trail before triggering
                effective_trail = state.trailing_stop_price * (1.0 - min_buf)
            else:
                # SHORT: require price to rise min_buf% ABOVE the trail before triggering
                effective_trail = state.trailing_stop_price * (1.0 + min_buf)
        else:
            effective_trail = state.trailing_stop_price

        if side == "buy" and last <= effective_trail:
            return "trailing_stop_hit"
        if side == "sell" and last >= effective_trail:
            return "trailing_stop_hit"

    # 7. Take-profit check (full close at tier-aware TP multiplier).
    margin_mult = compute_margin_multiplier(entry, last, side, leverage)
    if margin_mult >= _tp_multiplier(cfg, leverage):
        return "tp1_full"

    return "none"


class RiskManager:
    def __init__(self, cfg: RiskConfig, cooldown_seconds: float, start_balance: float) -> None:
        self.cfg = cfg
        self.cooldown_seconds = cooldown_seconds
        self.start_balance: float = start_balance

    def can_open(self, state: SymbolState) -> bool:
        now = time.monotonic()

        if state.position_qty != 0.0:
            return False
        if now < state.cooldown_until_monotonic:
            return False

        # Per-symbol ban after consecutive losses.
        if now < state.banned_until_monotonic:
            return False

        return True

    def mark_opened(self, state: SymbolState, side: str, qty: float, hold_seconds: float) -> None:
        now = time.monotonic()
        state.side = side
        state.position_qty = qty
        state.entry_ts_monotonic = now
        state.close_due_monotonic = now + hold_seconds

        # Calculate stop-loss price only if not already set by exchange-side order placement.
        if state.entry_price is not None and state.entry_price > 0 and state.stop_loss_price is None:
            leverage = max(state.max_leverage, 1.0)
            if state.max_leverage <= 1.0:
                logging.warning(
                    "mark_opened: max_leverage=%.1f for %s is unusually low — "
                    "leverage fetch may have failed; tier1 SL/TP will apply",
                    state.max_leverage, state.symbol,
                )
            state.stop_loss_price = compute_stop_loss_price(
                state.entry_price, leverage, side, _sl_fraction(self.cfg, leverage),
                maintenance_rate=state.maintenance_rate,
            )

        # Init trailing state.
        state.trailing_active = False
        state.best_pnl_pct = 0.0
        state.trailing_stop_price = None

    def record_trade_pnl(self, state: SymbolState, pnl: float) -> None:
        """Update per-symbol ban based on realized PnL.

        MUST be called while holding engine._lock.
        """
        if pnl < 0:
            state.consecutive_losses += 1

            # Per-symbol ban: 2 consecutive losses -> 1 hour ban.
            if state.consecutive_losses >= self.cfg.symbol_max_consecutive_losses:
                state.banned_until_monotonic = (
                    time.monotonic() + self.cfg.symbol_ban_duration_seconds
                )
                logging.warning(
                    "SYMBOL BAN %s: %d consecutive losses, banned for %.0f seconds",
                    state.symbol,
                    state.consecutive_losses,
                    self.cfg.symbol_ban_duration_seconds,
                )
                state.consecutive_losses = 0
        else:
            state.consecutive_losses = 0

    def apply_tp_cooldown(self, state: SymbolState) -> None:
        """Ban symbol for tp_cooldown_seconds after a take_profit close."""
        cd = self.cfg.tp_cooldown_seconds
        if cd <= 0:
            return
        state.banned_until_monotonic = time.monotonic() + cd
        logging.info(
            "TP COOLDOWN %s: banned for %.0f seconds after take_profit",
            state.symbol, cd,
        )

    def mark_closed(self, state: SymbolState) -> None:
        now = time.monotonic()
        state.position_qty = 0.0
        state.side = None
        state.entry_order_id = None
        state.close_order_id = None
        state.close_due_monotonic = 0.0
        state.close_queued = False
        state.cooldown_until_monotonic = now + self.cooldown_seconds
        state.entry_price = None
        state.entry_quote_locked = 0.0
        state.entry_wall_epoch = 0.0
        state.entry_signal_pct = None
        state.stop_loss_price = None
        state.stop_loss_order_id = None
        # Reset trailing state.
        state.trailing_active = False
        state.best_pnl_pct = 0.0
        state.trailing_stop_price = None
        state.close_reason = ""
        # Reset signal state to prevent stale triggers firing after cooldown.
        state.fair_trigger_latched = False
        state.pending_trigger_time = 0.0
        state.pending_trigger_move = 0.0
        state.pending_trigger_extreme = 0.0
        state.pending_trigger_side = ""
