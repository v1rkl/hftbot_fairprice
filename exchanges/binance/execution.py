from __future__ import annotations

import os
from typing import Any

import ccxt.async_support as ccxt_async

from core.base_execution import BaseExecutionEngine
from core.config import BotConfig
from core.paper import PaperAccount


class BinanceExecutionEngine(BaseExecutionEngine):
    """Binance USDT-M futures execution engine.

    Uses ccxt binanceusdm (USDT-margined perpetuals).
    Paper mode does not require API keys.
    """

    def __init__(self, cfg: BotConfig, paper_account: PaperAccount | None = None) -> None:
        super().__init__(cfg, paper_account)
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_SECRET", "")
        if not cfg.runtime.paper_mode and (not api_key or not secret):
            raise RuntimeError(
                "BINANCE_API_KEY and BINANCE_SECRET environment variables "
                "must be set for live trading"
            )
        self._extra_params = {}
        self.exchange = ccxt_async.binanceusdm({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": cfg.exchange.market_type},
        })

    async def initialize(self) -> dict[str, Any]:
        markets = await self.exchange.load_markets()
        # Binance USDM load_markets() does not populate limits.leverage.max.
        # Inject a conservative default (20x) so the leverage filter works.
        _DEFAULT_MAX_LEVERAGE = 20.0
        for market in markets.values():
            if market.get("swap") or market.get("future"):
                lim = market.setdefault("limits", {})
                lev = lim.setdefault("leverage", {})
                if not lev.get("max"):
                    lev["max"] = _DEFAULT_MAX_LEVERAGE
        return markets
