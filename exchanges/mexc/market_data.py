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

_MEXC_WS_URL_DEFAULT = "wss://contract.mexc.com/edge"

# 3 channels per symbol (fair.price, deal, depth.full) × 30 symbols/shard = 90 subscriptions.
# Empirically stable at this ratio; raise carefully if MEXC enforces a hard cap.
_MAX_SYMBOLS_PER_SHARD = 30

# Delay between starting each shard to avoid rate limiting on connect.
_SHARD_START_DELAY_S = 0.5

# Ping interval for client-initiated keep-alive (MEXC recommends ping every ~30s).
_PING_INTERVAL_S = 15.0

_HEALTH_LOG_INTERVAL_S = 60.0


def _to_exchange_symbol(symbol: str) -> str:
    """BTC/USDT:USDT -> BTC_USDT (MEXC contract format)."""
    base = symbol.split(":")[0]
    return base.replace("/", "_")


class _MexcWSShard:
    """Single MEXC contract WebSocket connection for up to _MAX_SYMBOLS_PER_SHARD symbols."""

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
        self._ping_task: asyncio.Task[None] | None = None

    async def run_forever(self) -> None:
        backoff_ms = self.cfg.data.reconnect_backoff_ms_min
        while not self._stop.is_set():
            try:
                await self._run_once()
                backoff_ms = self.cfg.data.reconnect_backoff_ms_min
            except websockets.exceptions.ConnectionClosedError as exc:
                self.connected = False
                logging.warning("MEXC WS %s connection closed: %s", self._prefix, exc)
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.cfg.data.reconnect_backoff_ms_max)
            except Exception as exc:  # noqa: BLE001
                self.connected = False
                logging.exception("MEXC WS %s failed, reconnecting: %s", self._prefix, exc)
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.cfg.data.reconnect_backoff_ms_max)

    async def stop(self) -> None:
        self._stop.set()
        if self._ping_task is not None:
            self._ping_task.cancel()

    async def _run_once(self) -> None:
        async with websockets.connect(
            self._ws_url,
            ping_interval=None,  # disabled: we run our own _ping_loop
            open_timeout=30,
        ) as ws:
            await self._subscribe(ws)
            self.connected = True
            logging.info(
                "MEXC WS %s connected (%d symbols)", self._prefix, len(self.symbols),
            )

            # Start client-side ping loop (MEXC may close the socket without it).
            self._ping_task = asyncio.create_task(self._ping_loop(ws))

            try:
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    recv_started = time.perf_counter()
                    msg = self._decode(raw)
                    if msg is None:
                        continue

                    channel = str(msg.get("channel") or msg.get("c") or "")
                    if "pong" in channel or channel == "pong":
                        continue
                    if "ping" in channel:
                        # server-initiated ping — reply with a pong.
                        await self._send_pong(ws)
                        continue

                    self.on_message(msg)
                    latency_ms = (time.perf_counter() - recv_started) * 1000.0
                    if latency_ms > self.cfg.data.processing_latency_warn_ms:
                        logging.warning(
                            "Slow WS processing %s: %.2f ms", self._prefix, latency_ms,
                        )
            finally:
                if self._ping_task is not None:
                    self._ping_task.cancel()
                    try:
                        await self._ping_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._ping_task = None
        self.connected = False

    async def _ping_loop(self, ws: Any) -> None:
        """Send periodic pings to keep the MEXC socket alive."""
        try:
            while not self._stop.is_set():
                await asyncio.sleep(_PING_INTERVAL_S)
                try:
                    await ws.send(orjson.dumps({"method": "ping"}).decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    logging.debug("MEXC WS %s ping failed: %s", self._prefix, exc)
                    return
        except asyncio.CancelledError:
            pass

    async def _send_pong(self, ws: Any) -> None:
        try:
            await ws.send(orjson.dumps({"method": "pong"}).decode("utf-8"))
            logging.debug("MEXC WS %s pong sent", self._prefix)
        except Exception:  # noqa: BLE001
            pass

    async def _subscribe(self, ws: Any) -> None:
        depth = self.cfg.data.orderbook_depth
        for sym in self.symbols:
            inst = _to_exchange_symbol(sym)
            # Fair price (mark price stream)
            await ws.send(orjson.dumps({
                "method": "sub.fair.price",
                "param": {"symbol": inst, "gzip": False},
            }).decode("utf-8"))
            # Trade prints
            await ws.send(orjson.dumps({
                "method": "sub.deal",
                "param": {"symbol": inst},
            }).decode("utf-8"))
            # Order book (full snapshot + increments)
            await ws.send(orjson.dumps({
                "method": "sub.depth.full",
                "param": {"symbol": inst, "limit": depth},
            }).decode("utf-8"))
            # Throttle subscriptions to stay under MEXC rate limits.
            await asyncio.sleep(0.05)

        logging.info(
            "MEXC WS %s subscriptions sent for %d symbols (depth=%d)",
            self._prefix, len(self.symbols), depth,
        )

    @staticmethod
    def _decode(raw: str | bytes) -> dict[str, Any] | None:
        try:
            if isinstance(raw, bytes):
                return orjson.loads(raw)
            return orjson.loads(raw.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logging.debug("MEXC WS decode error: %s", exc)
            return None


class MexcMarketDataWS(AbstractMarketDataWS):
    """MEXC contract futures WebSocket manager (sharded).

    Mirrors GateioMarketDataWS: splits symbols into shards, starts each with a
    staggered delay, runs a health-monitor logger.
    """

    def __init__(self, cfg: BotConfig, symbols: list[str], on_message: MessageHandler) -> None:
        self.cfg = cfg
        self._on_message = on_message
        self._ws_url = cfg.data.ws_url or _MEXC_WS_URL_DEFAULT
        self._tasks: list[asyncio.Task[None]] = []
        self._health_task: asyncio.Task[None] | None = None

        n = _MAX_SYMBOLS_PER_SHARD
        chunks = [symbols[i: i + n] for i in range(0, len(symbols), n)]
        self._shards: list[_MexcWSShard] = [
            _MexcWSShard(cfg, chunk, on_message, shard_id, self._ws_url)
            for shard_id, chunk in enumerate(chunks)
        ]
        logging.info(
            "MEXC WS manager: %d symbols -> %d shards (%d per shard)",
            len(symbols), len(self._shards), _MAX_SYMBOLS_PER_SHARD,
        )

    async def run_forever(self) -> None:
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
                    "MEXC WS health: %d/%d shards connected", connected, total,
                )
            else:
                logging.info(
                    "MEXC WS health: %d/%d shards connected", connected, total,
                )
