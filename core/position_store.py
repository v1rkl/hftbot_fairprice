"""Persistent position state — saves open positions to JSON for crash recovery."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .models import SymbolState

# Fields to persist (all that matter for position management after restart).
_PERSIST_FIELDS: tuple[str, ...] = (
    "symbol",
    "side",
    "position_qty",
    "entry_price",
    "entry_quote_locked",
    "entry_wall_epoch",
    "entry_signal_pct",
    "max_leverage",
    "stop_loss_price",
    "trailing_active",
    "best_pnl_pct",
    "trailing_stop_price",
    "close_reason",
    "consecutive_losses",
)


class PositionStore:
    """Atomic JSON persistence for open positions."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, states: dict[str, SymbolState]) -> bool:
        """Save all open positions atomically (write .tmp then os.replace)."""
        records: list[dict[str, Any]] = []
        for state in states.values():
            if state.position_qty == 0.0:
                continue
            rec: dict[str, Any] = {}
            for f in _PERSIST_FIELDS:
                rec[f] = getattr(state, f)
            # Store wall-clock based hold remaining so we can recalculate monotonic on load.
            rec["_saved_wall_epoch"] = time.time()
            # If close is already queued (sentinel=inf), mark for immediate close on restart.
            rec["_immediate_close"] = state.close_queued
            rec["_close_due_wall_epoch"] = (
                state.entry_wall_epoch + (state.close_due_monotonic - state.entry_ts_monotonic)
                if state.entry_ts_monotonic > 0 and state.close_due_monotonic < 1e15
                else 0.0
            )
            records.append(rec)

        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
            return True
        except (OSError, TypeError, ValueError):
            logging.critical("Failed to persist positions to %s", self._path, exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def load(self) -> list[dict[str, Any]]:
        """Load persisted positions. Returns empty list on any error."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logging.warning("Position store %s: expected list, got %s", self._path, type(data).__name__)
                return []
            return data
        except (json.JSONDecodeError, OSError):
            logging.warning("Position store %s: failed to load", self._path, exc_info=True)
            return []

    def clear(self) -> None:
        """Remove the persisted state file."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass
