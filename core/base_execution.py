from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

import ccxt

from .config import BotConfig
from .execution_types import ExitResult, FillResult
from .interfaces import AbstractExecutionEngine
from .models import VALID_SIDES, SymbolState
from .paper import PaperAccount
from .risk import _sl_fraction, compute_stop_loss_price

# Prevent division by zero for near-zero prices.
_MIN_PRICE_GUARD: float = 1e-12


def _validate_price(value: float, symbol: str, field: str) -> bool:
    """Return True if the price is a finite positive number."""
    if not math.isfinite(value) or value <= 0:
        logging.error("Invalid price for %s.%s: %r", symbol, field, value)
        return False
    return True


def _extract_filled(order: dict[str, Any]) -> float:
    """Extract filled quantity from a ccxt order dict.

    Gate.io futures always returns filled=None in fetch_order responses.
    Fall back to amount - remaining when filled is None or zero.
    """
    filled_raw = order.get("filled")
    if filled_raw is not None:
        val = float(filled_raw)
        if val > 0:
            return val
    # Gate.io fallback: compute from amount - remaining
    amount = order.get("amount")
    remaining = order.get("remaining")
    if amount is not None and remaining is not None:
        return max(float(amount) - float(remaining), 0.0)
    return 0.0


class BaseExecutionEngine(AbstractExecutionEngine):
    """Shared execution logic for all exchanges.

    Subclasses must set:
      - self.exchange  (ccxt async exchange instance)
      - self._extra_params  (dict of exchange-specific API params, e.g. {"settle": "usdt"})
    """

    def __init__(self, cfg: BotConfig, paper_account: PaperAccount | None = None) -> None:
        self.cfg = cfg
        self.paper_account = paper_account
        # Subclasses must set these:
        self.exchange: Any = None
        self._extra_params: dict[str, Any] = {}

    async def initialize(self) -> dict[str, Any]:
        result = await self.exchange.load_markets()
        if not isinstance(result, dict):
            raise RuntimeError(f"load_markets() returned unexpected type: {type(result)}")
        return result  # type: ignore[return-value]

    async def close(self) -> None:
        await self.exchange.close()

    async def fetch_open_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions from exchange via ccxt."""
        if self.cfg.runtime.paper_mode:
            return []
        raw = await self.exchange.fetch_positions(
            params=self._extra_params if self._extra_params else {},
        )
        positions: list[dict[str, Any]] = []
        for p in raw:
            # ccxt normalizes contracts → use 'contracts' or 'contractSize'.
            qty = abs(float(p.get("contracts") or p.get("contractSize") or 0.0))
            if qty <= 0:
                continue
            side_raw = (p.get("side") or "").lower()
            if side_raw == "long":
                side = "buy"
            elif side_raw == "short":
                side = "sell"
            else:
                continue
            leverage = float(p.get("leverage") or 0.0)
            if leverage == 0.0:
                # leverage=0 means cross-margin on Gate.io. The bot only operates
                # in isolated margin — skip to avoid mismatched risk calculations.
                logging.warning(
                    "RECONCILE skipping cross-margin position %s side=%s qty=%.2f "
                    "(leverage=0 indicates cross-margin). Close it manually.",
                    p.get("symbol"), side, qty,
                )
                continue
            positions.append({
                "symbol": p["symbol"],
                "side": side,
                "qty": qty,
                "entry_price": float(p.get("entryPrice") or 0.0),
                "notional": float(p.get("notional") or 0.0),
                "leverage": leverage,
                "unrealized_pnl": float(p.get("unrealizedPnl") or 0.0),
            })
        return positions

    async def fetch_position_qty(self, symbol: str) -> float:
        """Return absolute position size on exchange for symbol (0.0 if no position)."""
        if self.cfg.runtime.paper_mode and self.paper_account is not None:
            return 0.0  # paper mode: trust local state
        positions = await self.fetch_open_positions()
        for p in positions:
            if p.get("symbol") == symbol:
                return float(p.get("qty", 0.0))
        return 0.0

    async def fetch_account_balance(self) -> float:
        """Fetch total USDT balance from exchange."""
        if self.cfg.runtime.paper_mode and self.paper_account is not None:
            return self.paper_account.quote_balance
        balance = await self.exchange.fetch_balance(params=self._extra_params or {})
        usdt = balance.get("USDT")
        if usdt is None:
            raise RuntimeError(
                f"USDT key missing from balance response — got keys: {list(balance.keys())}. "
                "Check exchange.market_type config (should be 'swap' for USDT perpetuals)."
            )
        return float(usdt.get("total", 0.0))

    async def place_entry_limit(self, state: SymbolState, side: str) -> float:
        if side not in VALID_SIDES:
            raise ValueError(f"Invalid order side: {side!r}")

        leverage = max(float(state.max_leverage), 1.0)

        if self.cfg.runtime.paper_mode and self.paper_account is not None:
            return self._paper_entry(state, side, leverage)

        mid = state.book.mid
        if mid is None:
            return 0.0

        # Validate book sanity.
        if (state.book.bid and state.book.ask
                and state.book.bid >= state.book.ask):
            logging.error(
                "Crossed book for %s: bid=%.8f >= ask=%.8f",
                state.symbol, state.book.bid, state.book.ask,
            )
            return 0.0

        shift = state.tick_size * self.cfg.strategy.tick_offset_entry
        px = mid + shift if side == "buy" else mid - shift

        if not _validate_price(px, state.symbol, "entry_price"):
            return 0.0

        # Set isolated margin + leverage before each trade.
        # Subclasses can override _set_isolated_leverage() for exchange-specific behaviour.
        ok = await self._set_isolated_leverage(state, leverage)
        if not ok:
            return 0.0

        notional = self.cfg.strategy.quote_size_usdt * leverage
        qty_tokens = notional / max(px, _MIN_PRICE_GUARD)
        # Gate.io (and some other exchanges) trade in CONTRACTS, not tokens.
        # contractSize is the number of tokens per contract (e.g. 10 for STO, DRIFT, PIPPIN).
        # Dividing token qty by contractSize gives the correct number of contracts to submit.
        market = self.exchange.market(state.symbol)
        contract_size = float(market.get("contractSize") or 1.0)
        qty = qty_tokens / max(contract_size, 1e-12)
        amount = float(self.exchange.amount_to_precision(state.symbol, qty))
        price = float(self.exchange.price_to_precision(state.symbol, px))

        if not _validate_price(amount, state.symbol, "entry_amount"):
            return 0.0

        state.entry_price = price
        result = await self._submit_and_wait_fill(state, side, amount, price, is_close=False)
        filled = result.filled
        if result.cancel_failed:
            if filled <= 0:
                # Cancel failed and nothing filled yet — the order may still be live on the
                # exchange and fill later without any stop-loss protection.  Force an immediate
                # close timer so the engine detects the open position on the next reconcile.
                logging.critical(
                    "Entry cancel FAILED and filled=0 for %s/%s — order may be live with no SL. "
                    "Forcing close timer for safety.",
                    state.symbol, result.order_id,
                )
                state.close_due_monotonic = time.monotonic()
                return 0.0
            logging.warning(
                "Entry cancel failed for %s/%s — order may still be live",
                state.symbol, result.order_id,
            )

        # Use actual average fill price for SL calculation; fall back to limit price.
        actual_price = result.avg_price if result.avg_price > 0 else price
        if filled > 0 and actual_price > 0:
            state.entry_price = actual_price
            await self._place_stop_loss_order(state, side, filled, actual_price, leverage)

        return float(filled)

    async def place_exit(self, state: SymbolState) -> ExitResult:
        if state.position_qty == 0.0:
            return ExitResult(0.0, 0.0)
        if state.side not in VALID_SIDES:
            logging.error(
                "Cannot close position: invalid side %r for %s",
                state.side, state.symbol,
            )
            return ExitResult(0.0, 0.0)

        close_side = "sell" if state.side == "buy" else "buy"

        if self.cfg.runtime.paper_mode and self.paper_account is not None:
            return self._paper_exit(state, close_side)

        # Panic close: stop_loss is urgent. Skip the limit-and-wait path (2.5s
        # exposure window) and go straight to market. This prevents the race
        # where exchange liquidation beats our limit close on fast spikes
        # (see FOLKS/USDT 2026-04-16: liquidated @ 1.246 while limit @ 1.241
        # sat unfilled for 2.5s).
        if state.close_reason in ("stop_loss", "trailing_stop"):
            amount = float(self.exchange.amount_to_precision(state.symbol, state.position_qty))
            mfilled, mpx = await self._market_close(state.symbol, close_side, amount)
            if mfilled > 0 and state.stop_loss_order_id:
                await self._cancel_stop_loss_order(state)
            return ExitResult(mfilled, mpx)

        mid = state.book.mid or state.last_price or state.fair_price
        if mid is None:
            return ExitResult(0.0, 0.0)
        shift = state.tick_size * self.cfg.strategy.tick_offset_close
        # Buyers lean above mid (toward ask) for faster fill; sellers lean below mid.
        px = mid + shift if close_side == "buy" else mid - shift

        amount = float(self.exchange.amount_to_precision(state.symbol, state.position_qty))
        price = float(self.exchange.price_to_precision(state.symbol, px))
        # Place market exit FIRST, leaving the server stop intact. If our exit fails,
        # the server stop still protects the position. Cancelling first creates an
        # unprotected window — TRADOOR/USDT 2026-04-13 was liquidated this way:
        # local watchdog cancelled server stop, fast move liquidated before our exit
        # could fill. The server stop uses mark price (same feed as liquidation),
        # so even racing it is safe — whichever fills first closes the position.
        result = await self._submit_and_wait_fill(state, close_side, amount, price, is_close=True)
        filled = result.filled
        exit_px = float(price)
        remaining = max(amount - filled, 0.0)
        if remaining > 0:
            if result.cancel_failed:
                logging.critical(
                    "EMERGENCY MARKET CLOSE %s: cancel failed for order %s, "
                    "sending reduceOnly market order for %.8f",
                    state.symbol, result.order_id, remaining,
                )
            if self.cfg.strategy.close_with_market_fallback or result.cancel_failed:
                mfilled, mpx = await self._market_close(state.symbol, close_side, remaining)
                filled += mfilled
                if mpx > 0:
                    exit_px = mpx
        if filled > 0 and state.stop_loss_order_id:
            await self._cancel_stop_loss_order(state)
        return ExitResult(filled, exit_px)

    # -- Paper mode helpers --

    def _paper_entry(self, state: SymbolState, side: str, leverage: float) -> float:
        if self.paper_account is None:
            raise RuntimeError("paper_account required in paper mode")
        if side == "buy":
            px = state.book.ask or state.last_price or state.fair_price
        else:
            px = state.book.bid or state.last_price or state.fair_price
        if px is None:
            return 0.0
        # Fixed sizing: use quote_size_usdt as margin (same as live), no compounding.
        margin = min(
            self.cfg.strategy.quote_size_usdt,
            self.paper_account.available_quote(),
        )
        if margin <= 0:
            return 0.0
        notional = margin * leverage
        qty_tokens = notional / max(px, _MIN_PRICE_GUARD)
        # Store qty in CONTRACTS (same as live mode) so pnl_linear_usdt * contract_size is correct.
        market = self.exchange.market(state.symbol)
        contract_size = float(market.get("contractSize") or 1.0)
        qty = qty_tokens / max(contract_size, 1e-12)
        price = float(px)
        state.entry_price = price
        state.entry_quote_locked = float(margin)
        self.paper_account.open_position(
            state.symbol,
            side=side,
            qty=float(qty),
            entry_price=price,
            quote_locked=float(margin),
            contract_size=contract_size,
        )
        logging.info(
            "[PAPER] OPEN %s side=%s margin=%.2f lev=%.0fx notional=%.2f "
            "qty=%.6f (contracts) contract_size=%.4f price=%.8f bid=%.8f ask=%.8f",
            state.symbol, side.upper(), margin, leverage, notional, qty, contract_size,
            price, state.book.bid or 0.0, state.book.ask or 0.0,
        )
        return float(qty)

    def _paper_exit(self, state: SymbolState, close_side: str) -> ExitResult:
        if self.paper_account is None:
            raise RuntimeError("paper_account required in paper mode")
        if close_side == "sell":
            px = state.book.bid or state.last_price or state.fair_price
        else:
            px = state.book.ask or state.last_price or state.fair_price
        if px is None:
            return ExitResult(0.0, 0.0)
        amount = float(state.position_qty)
        price = float(px)
        self.paper_account.close_position(state.symbol, close_price=price)
        return ExitResult(amount, price)

    async def _place_stop_loss_order(
        self, state: SymbolState, side: str, qty: float, entry_price: float, leverage: float,
    ) -> None:
        """Place a stop-loss order on the exchange at a fraction of margin before liquidation."""
        if self.cfg.runtime.paper_mode:
            return
        loss_frac = _sl_fraction(self.cfg.risk, leverage)
        sl_price = compute_stop_loss_price(entry_price, leverage, side, loss_frac,
                                           maintenance_rate=state.maintenance_rate)
        close_side = "sell" if side == "buy" else "buy"
        sl_formatted = float(self.exchange.price_to_precision(state.symbol, sl_price))
        if sl_formatted <= 0:
            logging.error(
                "Computed SL price %.8f <= 0 for %s — forcing immediate close for safety",
                sl_formatted, state.symbol,
            )
            state.close_due_monotonic = time.monotonic()
            return
        # Set software backup before API call so the position is always protected.
        state.stop_loss_price = sl_formatted
        try:
            amount = float(self.exchange.amount_to_precision(state.symbol, qty))
            params = dict(self._extra_params)
            params["stopPrice"] = sl_formatted
            params["reduceOnly"] = True
            order = await self.exchange.create_order(
                symbol=state.symbol,
                type="stop_market",
                side=close_side,
                amount=amount,
                price=None,  # market execution at trigger — non-None becomes stop-limit
                params=params,
            )
            sl_order_id = str(order.get("id", ""))
            if sl_order_id:
                state.stop_loss_order_id = sl_order_id
            logging.info(
                "STOP-LOSS placed %s side=%s stop=%.8f entry=%.8f lev=%.0fx loss_frac=%.2f",
                state.symbol, close_side.upper(), sl_formatted,
                entry_price, leverage, loss_frac,
            )
        except Exception as e:
            logging.error(
                "Failed to place stop-loss for %s: %s — forcing immediate close for safety",
                state.symbol, e,
            )
            # Exchange-side SL failed — force close immediately; software stop also active.
            state.close_due_monotonic = time.monotonic()

    async def _cancel_stop_loss_order(self, state: SymbolState) -> None:
        """Cancel the exchange-side stop-loss order to prevent orphaned orders after close."""
        if not state.stop_loss_order_id:
            return
        fetch_params = self._extra_params if self._extra_params else {}
        try:
            await self.exchange.cancel_order(
                state.stop_loss_order_id,
                state.symbol,
                **({"params": fetch_params} if fetch_params else {}),
            )
            logging.info("STOP-LOSS order cancelled %s id=%s", state.symbol, state.stop_loss_order_id)
        except Exception as e:
            logging.warning(
                "Failed to cancel stop-loss order %s/%s: %s (may already be filled/cancelled)",
                state.symbol, state.stop_loss_order_id, e,
            )
        finally:
            state.stop_loss_order_id = None

    async def _set_isolated_leverage(self, state: SymbolState, leverage: float) -> bool:
        """Set isolated margin mode + leverage before entry. Returns False to abort the trade.

        Default implementation: try set_margin_mode then set_leverage via ccxt unified API.
        Subclasses (e.g. GateioExecutionEngine) can override for exchange-specific behaviour.
        """
        if self.cfg.runtime.paper_mode:
            return True

        # set_margin_mode — some exchanges don't support this; ignore those errors.
        try:
            await self.exchange.set_margin_mode(
                "isolated", state.symbol, **({"params": self._extra_params} if self._extra_params else {}),
            )
        except Exception as e:
            err = str(e).lower()
            harmless = (
                isinstance(e, ccxt.NotSupported)
                or any(kw in err for kw in ("already", "same", "no change", "not support", "not yet"))
            )
            if harmless:
                logging.debug("set_margin_mode skipped for %s: %s", state.symbol, e)
            else:
                logging.error(
                    "set_margin_mode failed for %s: %s — aborting entry",
                    state.symbol, e,
                )
                return False

        try:
            await self.exchange.set_leverage(
                round(leverage), state.symbol, **self._leverage_params(),
            )
        except Exception as e:
            logging.error(
                "set_leverage failed for %s: %s — aborting entry to avoid wrong leverage",
                state.symbol, e,
            )
            return False

        return True

    # -- Live order helpers --

    def _leverage_params(self) -> dict[str, Any]:
        """Extra kwargs for set_leverage. Override in subclass if needed."""
        if self._extra_params:
            return {"params": self._extra_params}
        return {}

    def _order_params(self, is_limit: bool = True) -> dict[str, Any]:
        """Build params dict for create_order."""
        params: dict[str, Any] = dict(self._extra_params)
        if is_limit:
            params["timeInForce"] = "GTC"
        return params

    async def _submit_and_wait_fill(
        self, state: SymbolState, side: str, amount: float, price: float, is_close: bool,
    ) -> FillResult:
        if amount <= 0:
            return FillResult(filled=0.0)
        if self.cfg.runtime.paper_mode:
            return FillResult(filled=amount)

        params = self._order_params(is_limit=True)
        if is_close:
            params["reduceOnly"] = True
        order = await self.exchange.create_order(
            symbol=state.symbol, type="limit", side=side,
            amount=amount, price=price, params=params,
        )
        order_id = str(order["id"])
        if is_close:
            state.close_order_id = order_id
        else:
            state.entry_order_id = order_id
        return await self._wait_fill_with_timeout(state.symbol, order_id, amount)

    async def _wait_fill_with_timeout(
        self, symbol: str, order_id: str, expected: float,
    ) -> FillResult:
        timeout_ms = self.cfg.strategy.fill_timeout_ms
        if timeout_ms < 100:
            logging.warning(
                "fill_timeout_ms=%d is less than one poll interval (100ms) — "
                "will poll exactly once before cancelling",
                timeout_ms,
            )
        loops = max(timeout_ms // 100, 1)
        filled = 0.0
        fetch_params = self._extra_params if self._extra_params else {}
        avg_price: float = 0.0
        for _ in range(loops):
            await asyncio.sleep(0.1)
            try:
                order = await self.exchange.fetch_order(
                    order_id, symbol, **({"params": fetch_params} if fetch_params else {}),
                )
            except ccxt.NetworkError as poll_exc:
                logging.warning("Transient error polling order %s/%s: %s — retrying", symbol, order_id, poll_exc)
                continue
            except ccxt.RequestTimeout as poll_exc:
                logging.warning("Timeout polling order %s/%s: %s — retrying", symbol, order_id, poll_exc)
                continue
            filled = _extract_filled(order)
            if order.get("status") == "closed":
                avg_price = float(order.get("average") or order.get("price") or 0.0)
                return FillResult(filled=filled, order_id=order_id, avg_price=avg_price)

        # Timeout — cancel and re-fetch final fill.
        cancel_failed = False
        try:
            await self.exchange.cancel_order(
                order_id, symbol, **({"params": fetch_params} if fetch_params else {}),
            )
            # Re-fetch to get final fill amount after cancel.
            order = await self.exchange.fetch_order(
                order_id, symbol, **({"params": fetch_params} if fetch_params else {}),
            )
            filled = _extract_filled(order) or filled
            avg_price = float(order.get("average") or order.get("price") or 0.0)
        except Exception as cancel_exc:
            cancel_failed = True
            logging.error(
                "Cancel order failed %s/%s: %s — order may still be live",
                symbol, order_id, cancel_exc,
            )
        return FillResult(filled=filled, cancel_failed=cancel_failed, order_id=order_id, avg_price=avg_price)

    async def _market_close(
        self, symbol: str, side: str, amount: float,
    ) -> tuple[float, float]:
        if self.cfg.runtime.paper_mode:
            return amount, 0.0
        params = self._order_params(is_limit=False)
        params["reduceOnly"] = True
        order = await self.exchange.create_order(
            symbol=symbol, type="market", side=side, amount=amount, params=params,
        )
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or order.get("price") or 0.0)
        return filled, avg
