from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExitResult:
    filled: float
    exit_price: float


@dataclass(slots=True)
class FillResult:
    """Result from _submit_and_wait_fill — tracks whether cancel succeeded."""
    filled: float
    cancel_failed: bool = False
    order_id: str = ""
    avg_price: float = 0.0  # actual average fill price (0 if unknown)
