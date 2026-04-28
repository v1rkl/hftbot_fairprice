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


class GateioExecutionEngine(BaseExecutionEngine):
    """Gate.io futures execution engine.

    Key differences from MEXC:
    - Uses ccxt_async.gateio with settle="usdt"
    - set_leverage requires params={"settle": "usdt"}
    - create_order for futures requires params={"settle": "usdt"}
    - Env vars: GATEIO_API_KEY / GATEIO_SECRET
    """

    def __init__(self, cfg: BotConfig, paper_account: PaperAccount | None = None) -> None:
        super().__init__(cfg, paper_account)
        self._extra_params = {"settle": "usdt"}
        api_key = os.getenv("GATEIO_API_KEY", "")
        secret = os.getenv("GATEIO_SECRET", "")
        if not cfg.runtime.paper_mode and (not api_key or not secret):
            raise RuntimeError(
                "GATEIO_API_KEY and GATEIO_SECRET environment variables "
                "must be set for live trading"
            )
        self.exchange = ccxt_async.gateio({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": cfg.exchange.market_type,
                "defaultSettle": "usdt",
            },
        })

    async def _set_isolated_leverage(self, state: SymbolState, leverage: float) -> bool:
        """Gate.io-specific: set isolated margin via direct futures leverage API.

        Gate.io USDT perpetuals:
          - Sending leverage > 0 switches the contract to isolated mode.
          - Gate.io returns leverage=0 in the response when the contract is in cross margin.
          - cross_leverage_limit in the response is the max leverage *allowed* for cross mode;
            it is always non-zero and does NOT indicate current margin mode.

        Step 1: attempt to disable account-level cross mode (best-effort, non-fatal).
        Step 2: set per-contract isolated leverage and verify response leverage > 0.
        """
        if self.cfg.runtime.paper_mode:
            return True

        if leverage <= 0:
            logging.error(
                "_set_isolated_leverage called with leverage=%.4f for %s — aborting entry",
                leverage, state.symbol,
            )
            return False

        market = self.exchange.market(state.symbol)
        contract = market["id"]  # e.g. "KERNEL_USDT"
        requested_lev = round(leverage)
        lev_str = str(requested_lev)
        try:
            # Step 1: disable global cross mode so per-contract isolated leverage takes effect.
            # Gate.io may return INVALID_REQUEST_BODY when this endpoint does not apply to
            # the account type (sub-accounts, unified accounts) — treat as non-fatal since
            # Step 2 sets per-contract mode directly and its response is the authoritative check.
            try:
                await self.exchange.private_futures_post_settle_positions_cross_mode({
                    "settle": "usdt",
                    "cross_margin": False,
                })
                logging.debug("Cross-mode disabled for USDT futures (or was already off)")
            except ccxt_async.ExchangeError as cm_err:
                msg = str(cm_err).lower()
                if (
                    "no need" in msg
                    or "position exists" in msg
                    or "position is not empty" in msg
                    or "invalid_request_body" in msg  # endpoint not applicable to account type
                    or "invalid request body" in msg
                ):
                    logging.warning(
                        "Cross-mode disable skipped for %s: %s — proceeding, will verify via leverage response",
                        state.symbol, cm_err,
                    )
                else:
                    raise  # unknown exchange error → outer except → abort

            # Step 2: set isolated leverage for this specific contract.
            # Sending leverage > 0 switches contract from cross to isolated mode on Gate.io.
            resp = await self.exchange.private_futures_post_settle_positions_contract_leverage({
                "settle": "usdt",
                "contract": contract,
                "leverage": lev_str,
            })
            # Parse leverage numerically — Gate.io may return int, float, or string.
            try:
                actual_lev_float = float(resp.get("leverage", 0))
            except (ValueError, TypeError):
                logging.error(
                    "Gate.io leverage field unparseable for %s (raw=%r) — aborting entry",
                    state.symbol, resp.get("leverage"),
                )
                return False

            cross_lev = str(resp.get("cross_leverage_limit", "n/a"))

            # leverage=0 in response → cross margin mode is active.
            if actual_lev_float == 0.0:
                logging.error(
                    "Gate.io returned leverage=0 for %s (cross_leverage_limit=%s) — "
                    "position is in CROSS margin, aborting entry",
                    state.symbol, cross_lev,
                )
                return False

            # Confirm exchange applied the leverage we requested (may be capped per contract rules).
            if actual_lev_float != requested_lev:
                logging.error(
                    "Gate.io confirmed leverage=%.0f for %s but requested %s — "
                    "risk parameters would be mismatched, aborting entry",
                    actual_lev_float, state.symbol, lev_str,
                )
                return False

            logging.info(
                "Isolated margin confirmed %s leverage=%.0f cross_leverage_limit=%s",
                state.symbol, actual_lev_float, cross_lev,
            )
            return True
        except Exception as e:
            logging.error(
                "set_isolated_leverage failed for %s: %s — aborting entry",
                state.symbol, e,
            )
            return False

    async def _place_stop_loss_order(
        self, state: SymbolState, side: str, qty: float, entry_price: float, leverage: float,
    ) -> None:
        """Gate.io: place a price-trigger stop-loss using /futures/usdt/price_orders.

        Uses initial.close=true so Gate.io closes the existing position when triggered
        instead of opening a new one (the bug that caused the STO liquidation incident).

        Trigger rules:
          LONG  (side='buy'):  stop is BELOW entry — trigger when price <= stop → rule=2
          SHORT (side='sell'): stop is ABOVE entry — trigger when price >= stop → rule=1

        On API failure: falls back to software-only stop (state.stop_loss_price is still
        set so engine.py's tick loop provides backup protection).
        """
        loss_frac = _sl_fraction(self.cfg.risk, leverage)
        sl_price = compute_stop_loss_price(entry_price, leverage, side, loss_frac,
                                           maintenance_rate=state.maintenance_rate)
        close_side = "sell" if side == "buy" else "buy"

        if self.cfg.runtime.paper_mode:
            sl_formatted = sl_price
        else:
            sl_formatted = float(self.exchange.price_to_precision(state.symbol, sl_price))

        if sl_formatted <= 0:
            logging.error(
                "Computed SL price %.8f <= 0 for %s — aborting SL placement",
                sl_formatted, state.symbol,
            )
            return

        # Always set software backup first — engine.py checks this every tick.
        state.stop_loss_price = sl_formatted
        state.stop_loss_order_id = None

        if self.cfg.runtime.paper_mode:
            logging.info(
                "STOP-LOSS set (paper) %s side=%s stop=%.8f entry=%.8f lev=%.0fx loss_frac=%.2f",
                state.symbol, close_side.upper(), sl_formatted, entry_price, leverage, loss_frac,
            )
            return

        # rule=2: trigger when price <= stop (LONG — price falls to stop)
        # rule=1: trigger when price >= stop (SHORT — price rises to stop)
        rule = 2 if side == "buy" else 1

        market = self.exchange.market(state.symbol)
        contract = market["id"]  # e.g. "STO_USDT"

        try:
            resp = await self.exchange.private_futures_post_settle_price_orders({
                "settle": "usdt",
                "trigger": {
                    "strategy_type": 0,   # price trigger
                    "price_type": 1,      # mark price (same feed used for liquidation)
                    "price": str(sl_formatted),
                    "rule": rule,
                    "expiration": 86400,  # 24h TTL; Gate.io auto-cancels close=true when position closes
                },
                "initial": {
                    "contract": contract,
                    "size": 0,            # 0 + close=true = close entire position
                    "price": "0",         # "0" = market execution at trigger (not stop-limit)
                    "close": True,        # close existing position, never open a new one
                    "tif": "ioc",         # Gate.io REQUIRES tif=ioc for price=0 (market) orders
                },
            })
            order_id = str(resp.get("id", ""))
            if order_id:
                state.stop_loss_order_id = order_id
            else:
                logging.warning(
                    "STOP-LOSS placed for %s but response had no id — cannot cancel later: %r",
                    state.symbol, resp,
                )
            logging.info(
                "STOP-LOSS placed (server) %s side=%s stop=%.8f entry=%.8f "
                "lev=%.0fx loss_frac=%.2f rule=%d order_id=%s",
                state.symbol, close_side.upper(), sl_formatted, entry_price,
                leverage, loss_frac, rule, order_id,
            )
        except Exception as e:
            logging.error(
                "STOP-LOSS server placement failed for %s: %s — "
                "forcing immediate close (software-only stop cannot react fast "
                "enough to microcap volatility; HANA 2026-04-14 was liquidated "
                "this way). Software backup at %.8f still active during close.",
                state.symbol, e, sl_formatted,
            )
            # Exchange-side SL failed — schedule immediate close; software stop
            # remains as fallback if close itself fails.
            state.close_due_monotonic = time.monotonic()

    async def _cancel_stop_loss_order(self, state: SymbolState) -> None:
        """Cancel the server-side stop-loss via /futures/usdt/price_orders/{order_id}.

        Gate.io auto-cancels close=true orders when the position is closed normally,
        so calling this is optional — but it cleans up immediately on normal bot exits.
        """
        if not state.stop_loss_order_id:
            return

        if self.cfg.runtime.paper_mode:
            state.stop_loss_order_id = None
            return

        order_id = state.stop_loss_order_id
        try:
            await self.exchange.private_futures_delete_settle_price_orders_order_id({
                "settle": "usdt",
                "order_id": order_id,
            })
            logging.info(
                "STOP-LOSS cancelled (server) %s order_id=%s",
                state.symbol, order_id,
            )
        except Exception as e:
            logging.warning(
                "Failed to cancel stop-loss order %s/%s: %s "
                "(may already be filled or closed)",
                state.symbol, order_id, e,
            )
        finally:
            state.stop_loss_order_id = None

    async def initialize(self) -> dict[str, Any]:
        # Credentials already validated in __init__ — no re-check needed here.
        result: dict[str, Any] = await self.exchange.load_markets()
        return result
