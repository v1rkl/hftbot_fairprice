from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

# Max samples retained per deque. At ~100 msg/s over a 5s window, 500 is generous.
_MAX_SAMPLES: int = 500


@dataclass(slots=True)
class BookTop:
    bid: float | None = None
    ask: float | None = None
    # Full depth levels: list of [price, qty] -- sorted best->worst.
    bids: list[list[float]] = field(default_factory=list)
    asks: list[list[float]] = field(default_factory=list)

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float | None:
        """Spread as percentage of mid price."""
        mid = self.mid
        if mid is None or mid <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / mid * 100.0

    def depth_quote(self, side: str, levels: int | None = None) -> float:
        """Total quote liquidity (price * qty) on given side of the book.
        side='ask' -> available to buy into, side='bid' -> available to sell into."""
        if side not in ("ask", "bid"):
            raise ValueError(f"depth_quote side must be 'ask' or 'bid', got {side!r}")
        rows = self.asks if side == "ask" else self.bids
        total = 0.0
        for i, level in enumerate(rows):
            if levels is not None and i >= levels:
                break
            total += level[0] * level[1]  # price * qty
        return total


VALID_SIDES: frozenset[str] = frozenset({"buy", "sell"})


@dataclass(slots=True)
class SymbolState:
    symbol: str
    tick_size: float
    max_leverage: float = 1.0
    contract_size: float = 1.0
    maintenance_rate: float = 0.0
    book: BookTop = field(default_factory=BookTop)
    last_price: float | None = None
    fair_price: float | None = None
    entry_price: float | None = None
    entry_quote_locked: float = 0.0
    position_qty: float = 0.0
    cooldown_until_monotonic: float = 0.0
    close_due_monotonic: float = 0.0
    side: Literal["buy", "sell"] | None = None
    entry_order_id: str | None = None
    close_order_id: str | None = None
    entry_ts_monotonic: float = 0.0
    entry_wall_epoch: float = 0.0
    entry_signal_pct: float | None = None
    fair_samples: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES),
    )
    last_samples: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=_MAX_SAMPLES),
    )
    fair_trigger_latched: bool = False
    last_update_monotonic: float = 0.0
    monitoring_start_monotonic: float = 0.0
    pending_trigger_time: float = 0.0
    pending_trigger_move: float = 0.0
    pending_trigger_extreme: float = 0.0
    pending_trigger_side: str = ""
    consecutive_order_errors: int = 0
    consecutive_losses: int = 0
    banned_until_monotonic: float = 0.0
    stop_loss_price: float | None = None
    stop_loss_order_id: str | None = None  # exchange-side SL order ID (live mode)
    has_pending_open: bool = False  # True while entry order is in-flight (Phase 2)
    fair_price_updated_monotonic: float = 0.0
    # Trailing stop / take-profit state.
    trailing_active: bool = False
    best_pnl_pct: float = 0.0
    trailing_stop_price: float | None = None
    close_reason: str = ""
    close_queued: bool = False
    close_fail_count: int = 0  # consecutive place_exit failures; triggers exchange verification
