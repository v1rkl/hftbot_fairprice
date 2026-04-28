from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PaperPosition:
    side: str          # "buy"=LONG, "sell"=SHORT
    qty: float         # position size in CONTRACTS (margin * leverage / entry / contract_size)
    entry_price: float
    quote_locked: float   # margin (USDT locked from balance)
    contract_size: float = 1.0  # tokens per contract; needed for PnL calc


class PaperAccount:
    """Paper-mode balance model with leverage support.

    margin = quote_locked (USDT reserved from balance)
    notional = margin * leverage
    qty_tokens = notional / entry_price
    qty_contracts = qty_tokens / contract_size

    PnL (LONG):  qty_contracts * contract_size * (close - entry)
    PnL (SHORT): qty_contracts * contract_size * (entry - close)
    Return = margin + PnL  (cannot go below 0 — no margin call in paper mode)
    """

    def __init__(self, start_quote_usdt: float) -> None:
        self.quote_balance: float = float(start_quote_usdt)
        self.positions: dict[str, PaperPosition] = {}

    def available_quote(self) -> float:
        return self.quote_balance

    def open_position(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        quote_locked: float,
        contract_size: float = 1.0,
    ) -> None:
        quote_locked = float(quote_locked)
        if quote_locked <= 0:
            return
        if self.quote_balance + 1e-12 < quote_locked:
            raise RuntimeError(f"Insufficient paper balance: have={self.quote_balance} need={quote_locked}")
        self.quote_balance -= quote_locked
        self.positions[symbol] = PaperPosition(
            side=side,
            qty=float(qty),
            entry_price=float(entry_price),
            quote_locked=quote_locked,
            contract_size=float(contract_size),
        )

    def close_position(self, symbol: str, close_price: float) -> float:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0.0
        close_price = float(close_price)
        if close_price <= 0 or pos.entry_price <= 0:
            self.quote_balance += pos.quote_locked
            return pos.qty

        tokens = pos.qty * pos.contract_size
        if pos.side == "buy":
            pnl = tokens * (close_price - pos.entry_price)
        else:
            pnl = tokens * (pos.entry_price - close_price)

        self.quote_balance += max(pos.quote_locked + pnl, 0.0)
        return pos.qty
