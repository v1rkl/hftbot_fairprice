"""Signal rejection counters — tracks why signals are skipped or filtered out."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

# Fields that represent throughput (not rejections).
_THROUGHPUT_FIELDS: frozenset[str] = frozenset({
    "signals_confirmed", "entries_attempted", "entries_filled",
})


@dataclass(slots=True)
class SignalRejectionCounters:
    """Cumulative counters for each signal rejection reason.

    Categories:
    - Signal detection: no_price_data, fair_stale, window_warmup, signal_latched
    - Signal confirmation: pullback_cancel, last_move_cancel
    - Liquidity gates: no_bid_ask, spread_too_wide, spread_signal_ratio, depth_insufficient
    - Position gates: max_positions, risk_blocked, circuit_breaker
    """

    # Signal detection phase.
    no_price_data: int = 0
    fair_stale: int = 0
    window_warmup: int = 0
    signal_latched: int = 0

    # Signal confirmation phase.
    pullback_cancel: int = 0
    last_move_cancel: int = 0
    adverse_last_chg: int = 0
    fair_rise_capped: int = 0

    # Liquidity / book quality gates.
    no_bid_ask: int = 0
    spread_too_wide: int = 0
    spread_signal_ratio: int = 0
    depth_insufficient: int = 0

    # Position / risk gates.
    max_positions: int = 0
    risk_blocked: int = 0
    circuit_breaker: int = 0

    # Confirmed signals that passed all checks and attempted entry.
    signals_confirmed: int = 0
    entries_attempted: int = 0
    entries_filled: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def nonzero_dict(self) -> dict[str, int]:
        """Return only counters with value > 0."""
        return {k: v for k, v in asdict(self).items() if v > 0}

    def total_rejected(self) -> int:
        """Total rejections across all reasons (excludes throughput counters)."""
        return sum(
            getattr(self, f.name)
            for f in fields(self)
            if f.name not in _THROUGHPUT_FIELDS
        )

    def summary_line(self) -> str:
        """One-line summary for logging (rejection counters only)."""
        nz = {k: v for k, v in asdict(self).items()
              if v > 0 and k not in _THROUGHPUT_FIELDS}
        if not nz:
            return "rejections: (none)"
        parts = [f"{k}={v}" for k, v in nz.items()]
        return "rejections: " + " ".join(parts)
