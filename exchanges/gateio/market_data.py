from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import orjson
import websockets
import websockets.exceptions

from core.config import BotConfig
from core.interfaces import AbstractMarketDataWS, MessageHandler

_GATEIO_WS_URL_DEFAULT = "wss://fx-ws.gateio.ws/v4/ws/usdt"

# Gate.io limits 100 subscriptions per connection (3 channels x N symbols).
# 30 symbols x 3 = 90 subscriptions — safely under the limit.
_MAX_SYMBOLS_PER_SHARD = 30

# Delay between starting each shard to avoid rate limiting on connect.
_SHARD_START_DELAY_S = 0.5

# How often to log overall shard health (seconds).
_HEALTH_LOG_INTERVAL_S = 60.0


class _GateioWSShard:
    """Single Gate.io WebSocket connection handling up to _MAX_SYMBOLS_PER_SHARD symbols."""

    def __init__(
        self,
        cfg: BotConfig,
        symbols: list[str],
        on_message: MessageHandler,
        shard_id: int,
        ws_url: str,
    ) -> None:
        self.cfg = cfg
        self.symbols = symbols
        self.on_message = on_message
        self.shard_id = shard_id
        self._ws_url = ws_url
        self._stop = asyncio.Event()
        self.connected = False
        self._prefix = f"[shard-{shard_id}]"

    async def run_forever(self) -> None:
        backoff_ms = self.cfg.data.reconnect_backoff_ms_min
        while not self._stop.is_set():
            try:
                await self._run_once()
                backoff_ms = self.cfg.data.reconnect_backoff_ms_min
            except websockets.exceptions.ConnectionClosedError as exc:
                self.connected = False
                logging.warning(
                    "Gate.io WS %s connection closed: %s", self._prefix, exc,
                )
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.cfg.data.reconnect_backoff_ms_max)
            except Exception as exc:
                self.connected = False
                logging.exception(
                    "Gate.io WS %s failed, reconnecting: %s", self._prefix, exc,
                )
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.cfg.data.reconnect_backoff_ms_max)

    async def stop(self) -> None:
        self._stop.set()

    async def _run_once(self) -> None:
        async with websockets.connect(
            self._ws_url,
            ping_interval=10,
            ping_timeout=20,
            open_timeout=30,
        ) as ws:
            await self._subscribe(ws)
            self.connected = True
            logging.info(
                "Gate.io WS %s connected (%d symbols)", self._prefix, len(self.symbols),
            )
            async for raw in ws:
                if self._stop.is_set():
                    break
                recv_started = time.perf_counter()
                msg = self._decode(raw)
                if msg is None:
                    continue

                if msg.get("channel") == "futures.ping":
                    await self._handle_ping(ws, msg)
                    continue

                if msg.get("event") not in ("update", "all", None):
                    continue

                self.on_message(msg)
                latency_ms = (time.perf_counter() - recv_started) * 1000.0
                if latency_ms > self.cfg.data.processing_latency_warn_ms:
                    logging.warning(
                        "Slow WS processing %s: %.2f ms", self._prefix, latency_ms,
                    )
        self.connected = False

    async def _handle_ping(self, ws: Any, msg: dict[str, Any]) -> None:
        """Respond to Gate.io application-level ping with a matching pong.

        Gate.io requires the pong to echo the time field from the ping.
        Without it the server may not accept the pong and will disconnect.
        """
        pong_time = msg.get("time") or int(time.time())
        await ws.send(orjson.dumps({"channel": "futures.pong", "time": pong_time}).decode("utf-8"))

    async def _subscribe(self, ws: websockets.ClientConnection) -> None:
        depth = self.cfg.data.orderbook_depth
        instruments = [_to_exchange_symbol(s) for s in self.symbols]
        ts = int(time.time())

        # trades and tickers: one message for all symbols combined.
        await ws.send(orjson.dumps({
            "time": ts, "channel": "futures.trades",
            "event": "subscribe", "payload": instruments,
        }).decode("utf-8"))
        await ws.send(orjson.dumps({
            "time": ts, "channel": "futures.tickers",
            "event": "subscribe", "payload": instruments,
        }).decode("utf-8"))

        # order_book: one message per symbol (requires individual depth/speed params).
        for inst in instruments:
            await ws.send(orjson.dumps({
                "time": ts, "channel": "futures.order_book",
                "event": "subscribe", "payload": [inst, str(depth), "0"],
            }).decode("utf-8"))
            await asyncio.sleep(0.05)

        logging.info(
            "Gate.io WS %s subscriptions sent for %d symbols",
            self._prefix, len(self.symbols),
        )

    @staticmethod
    def _decode(raw: str | bytes) -> dict[str, Any] | None:
        try:
            if isinstance(raw, bytes):
                return orjson.loads(raw)
            return orjson.loads(raw.encode("utf-8"))
        except Exception as exc:
            logging.debug("Gate.io WS decode error: %s", exc)
            return None


def _to_exchange_symbol(symbol: str) -> str:
    base = symbol.split(":")[0]
    return base.replace("/", "_")


class GateioMarketDataWS(AbstractMarketDataWS):
    """
    Gate.io futures WebSocket manager.

    Splits symbols into shards of _MAX_SYMBOLS_PER_SHARD (30) to stay within
    Gate.io's 100-subscriptions-per-connection limit (30 x 3 channels = 90).
    Each shard is an independent WS connection with its own reconnect loop.
    External interface is identical to a single-connection implementation.
    """

    def __init__(self, cfg: BotConfig, symbols: list[str], on_message: MessageHandler) -> None:
        self.cfg = cfg
        self._on_message = on_message
        self._ws_url = cfg.data.ws_url or _GATEIO_WS_URL_DEFAULT
        self._tasks: list[asyncio.Task[None]] = []
        self._health_task: asyncio.Task[None] | None = None

        # Split symbols into shards.
        n = _MAX_SYMBOLS_PER_SHARD
        chunks = [symbols[i: i + n] for i in range(0, len(symbols), n)]
        self._shards = [
            _GateioWSShard(cfg, chunk, on_message, shard_id, self._ws_url)
            for shard_id, chunk in enumerate(chunks)
        ]
        logging.info(
            "Gate.io WS manager: %d symbols -> %d shards (%d per shard)",
            len(symbols), len(self._shards), _MAX_SYMBOLS_PER_SHARD,
        )

    async def run_forever(self) -> None:
        # Start shards with staggered delay to avoid hitting Gate.io rate limits.
        for shard in self._shards:
            task = asyncio.create_task(shard.run_forever())
            self._tasks.append(task)
            await asyncio.sleep(_SHARD_START_DELAY_S)

        self._health_task = asyncio.create_task(self._health_monitor())

        await asyncio.gather(*self._tasks, self._health_task, return_exceptions=True)

    async def stop(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
        await asyncio.gather(*(shard.stop() for shard in self._shards))
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _health_monitor(self) -> None:
        while True:
            await asyncio.sleep(_HEALTH_LOG_INTERVAL_S)
            connected = sum(1 for s in self._shards if s.connected)
            total = len(self._shards)
            if connected < total:
                logging.warning(
                    "Gate.io WS health: %d/%d shards connected", connected, total,
                )
            else:
                logging.info(
                    "Gate.io WS health: %d/%d shards connected", connected, total,
                )
