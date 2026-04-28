from __future__ import annotations

from .models import SymbolState


def compute_deviation(state: SymbolState) -> float | None:
    if state.last_price is None or state.fair_price is None or state.fair_price <= 0:
        return None
    return (state.last_price - state.fair_price) / state.fair_price
