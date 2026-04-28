from __future__ import annotations

import logging
import os
import time
from typing import Any

import ccxt.async_support as ccxt_async

from core.base_execution import BaseExecutionEngine
from core.config import BotConfig
from core.models import SymbolState
from core.paper import PaperAccount
from core.risk import _sl_fraction, compute_stop_loss_price


class MexcExecutionEngine(BaseExecutionEngine):
    """MEXC contract futures execution engine.

    Differences from Gate.io:
      - ccxt exchange id: mexc
      - margin mode: set via openType=1 (isolated) in set_leverage params
      - stop-loss: placed via create_order(type='stop_market') with triggerPrice + reduceOnly
      - Env vars: MEXC_API_KEY / MEXC_SECRET
    """

    def __init__(self, cfg: BotConfig, paper_account: PaperAccount | None = None) -> None:
        super().__init__(cfg, paper_account)
        self._extra_params = {}
        self.exchange = ccxt_async.mexc({
            "apiKey": os.getenv("MEXC_API_KEY", ""),
            "secret": os.getenv("MEXC_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": cfg.exchange.market_type},
        })

    async def initialize(self) -> dict[str, Any]:
        if not self.cfg.runtime.paper_mode:
            api_key = os.getenv("MEXC_API_KEY", "")
            secret = os.getenv("MEXC_SECRET", "")
            if not api_key or not secret:
                raise RuntimeError(
                    "MEXC_API_KEY and MEXC_SECRET environment variables "
                    "must be set for live trading"
                )
        return await super().initialize()

    async def _set_isolated_leverage(self, state: SymbolState, leverage: float) -> bool:
        """MEXC isolated-margin leverage setter.

        ccxt.mexc does not support a unified set_margin_mode call for contract futures,
        so we instead pass openType=1 (isolated) directly to set_leverage. positionType
        is symmetric: MEXC accepts 1 (long) or 2 (short); passing openType alone without
        positionType sets the default for BOTH sides on isolated contracts.
        """
        if self.cfg.runtime.paper_mode:
            return True

        if leverage <= 0:
            logging.error(
                "_set_isolated_leverage called with leverage=%.4f for %s — aborting entry",
                leverage, state.symbol,
            )
            return False

        try:
            await self.exchange.set_leverage(
                round(leverage),
                state.symbol,
                params={"openType": 1},  # 1 = isolated, 2 = cross
            )
        except Exception as e:  # noqa: BLE001
            logging.error(
                "MEXC set_leverage failed for %s: %s — aborting entry",
                state.symbol, e,
            )
            return False

        return True

    async def _place_stop_loss_order(
        self, state: SymbolState, side: str, qty: float, entry_price: float, leverage: float,
    ) -> None:
        """Place a server-side stop-loss order on MEXC.

        Uses ccxt create_order(type='stop_market') with triggerPrice + reduceOnly.
        If the API rejects the order, falls back to software-only stop (state.stop_loss_price
        is still set, and engine.py:update_trailing_state polls every tick).
        """
        if self.cfg.runtime.paper_mode:
            # Still set state.stop_loss_price so engine-side tick check can arm software stop.
            loss_frac = _sl_fraction(self.cfg.risk, leverage)
            sl_price = compute_stop_loss_price(
                entry_price, leverage, side, loss_frac,
                maintenance_rate=state.maintenance_rate,
            )
            state.stop_loss_price = float(
                self.exchange.price_to_precision(state.symbol, sl_price),
            )
            return

        loss_frac = _sl_fraction(self.cfg.risk, leverage)
        sl_price = compute_stop_loss_price(
            entry_price, leverage, side, loss_frac,
            maintenance_rate=state.maintenance_rate,
        )
        sl_formatted = float(self.exchange.price_to_precision(state.symbol, sl_price))
        if sl_formatted <= 0:
            logging.error(
                "Computed SL price %.8f <= 0 for %s — software stop only",
                sl_formatted, state.symbol,
            )
            return

        # Software backup first, always, so position is protected even if API fails.
        state.stop_loss_price = sl_formatted

        close_side = "sell" if side == "buy" else "buy"
        amount = float(self.exchange.amount_to_precision(state.symbol, qty))
        params: dict[str, Any] = dict(self._extra_params)
        params["reduceOnly"] = True
        params["triggerPrice"] = sl_formatted

        try:
            order = await self.exchange.create_order(
                symbol=state.symbol,
                type="stop_market",
                side=close_side,
                amount=amount,
                price=None,
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
        except Exception as e:  # noqa: BLE001
            logging.warning(
                "MEXC SL placement failed for %s: %s — falling back to software stop "
                "(state.stop_loss_price=%.8f still active)",
                state.symbol, e, sl_formatted,
            )
            state.stop_loss_order_id = None
            # Do NOT force-close here — software stop will trigger in the next engine tick
            # if price actually crosses the level.

    async def _cancel_stop_loss_order(self, state: SymbolState) -> None:
        """Cancel the server-side stop-loss order if one is live."""
        if self.cfg.runtime.paper_mode:
            state.stop_loss_order_id = None
            return
        if not state.stop_loss_order_id:
            return
        try:
            await self.exchange.cancel_order(state.stop_loss_order_id, state.symbol)
            logging.info(
                "STOP-LOSS order cancelled %s id=%s",
                state.symbol, state.stop_loss_order_id,
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if any(k in msg for k in ("closed", "filled", "not found", "already")):
                logging.warning(
                    "MEXC cancel SL %s/%s: %s (already gone, ignoring)",
                    state.symbol, state.stop_loss_order_id, e,
                )
            else:
                logging.error(
                    "MEXC cancel SL failed %s/%s: %s",
                    state.symbol, state.stop_loss_order_id, e,
                )
        finally:
            state.stop_loss_order_id = None
