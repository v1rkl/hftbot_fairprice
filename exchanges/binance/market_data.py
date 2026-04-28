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

_BINANCE_WS_URL = "wss://fstream.binance.com/ws"

# Binance allows up to 200 subscriptions per connection.
# Each symbol = 2 streams (markPrice + bookTicker) → 100 symbols max per shard.
_MAX_SYMBOLS_PER_SHARD = 180  # 1 stream/symbol (markPrice only) × 180 = 180, under 200 limit
_SHARD_START_DELAY_S = 0.5


def _to_exchange_symbol(symbol: str) -> str:
    """BTC/USDT:USDT -> btcusdt (Binance stream format, lowercase)."""
    base = symbol.split(":")[0]          # BTC/USDT
    return base.replace("/", "").lower() # btcusdt


class _BinanceWSShard:
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
                logging.warning("Binance WS %s connection closed: %s", self._prefix, exc)
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.cfg.data.reconnect_backoff_ms_max)
            except Exception as exc:
                self.connected = False
                logging.exception("Binance WS %s failed, reconnecting: %s", self._prefix, exc)
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.cfg.data.reconnect_backoff_ms_max)

    async def stop(self) -> None:
        self._stop.set()

    async def _run_once(self) -> None:
        async with websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=30,
            open_timeout=30,
        ) as ws:
            await self._subscribe(ws)
            self.connected = True
            logging.info("Binance WS %s connected (%d symbols)", self._prefix, len(self.symbols))
            async for raw in ws:
                if self._stop.is_set():
                    break
                recv_started = time.perf_counter()
                msg = self._decode(raw)
                if msg is None:
                    continue
                # Binance wraps combined-stream events: {"stream":"...", "data":{...}}
                if "data" in msg and isinstance(msg["data"], dict):
                    inner = msg["data"]
                    # propagate symbol to top-level for parser compatibility
                    if "s" not in inner and "stream" in msg:
                        stream: str = msg["stream"]
                        sym_raw = stream.split("@")[0].upper()
                        inner["s"] = sym_raw
                    self.on_message(inner)
                else:
                    self.on_message(msg)
                latency_ms = (time.perf_counter() - recv_started) * 1000.0
                if latency_ms > self.cfg.data.processing_latency_warn_ms:
                    logging.warning("Slow WS processing %s: %.2f ms", self._prefix, latency_ms)
        self.connected = False

    async def _subscribe(self, ws: websockets.ClientConnection) -> None:
        streams: list[str] = []
        for symbol in self.symbols:
            inst = _to_exchange_symbol(symbol)
            # markPrice@1s: ~1 msg/sec per symbol — manageable for 700 symbols.
            # bookTicker fires on every bid/ask tick (~100s msgs/sec total) — too much.
            # Paper mode uses fair_price as bid/ask fallback, so bookTicker is optional.
            streams.append(f"{inst}@markPrice@1s")

        # Binance subscription message
        payload = {"method": "SUBSCRIBE", "params": streams, "id": self.shard_id + 1}
        await ws.send(orjson.dumps(payload).decode("utf-8"))
        # Read subscription acknowledgement. Success: {"result":null,"id":N}
        # Failure: {"code":-1,...}. Skip the ack so it doesn't land in the message loop.
        try:
            ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack = self._decode(ack_raw)
            if ack and ack.get("result") is not None:
                logging.error("Binance WS %s subscription error: %s", self._prefix, ack)
        except asyncio.TimeoutError:
            logging.warning("Binance WS %s: no subscription ack within 10s", self._prefix)
        logging.info(
            "Binance WS %s subscriptions sent: %d streams for %d symbols",
            self._prefix, len(streams), len(self.symbols),
        )

    @staticmethod
    def _decode(raw: str | bytes) -> dict[str, Any] | None:
        try:
            if isinstance(raw, bytes):
                return orjson.loads(raw)
            return orjson.loads(raw.encode("utf-8"))
        except Exception as exc:
            logging.debug("Binance WS decode error: %s", exc)
            return None


class BinanceMarketDataWS(AbstractMarketDataWS):
    """Binance USDT-M futures WebSocket manager (sharded)."""

    def __init__(self, cfg: BotConfig, symbols: list[str], on_message: MessageHandler) -> None:
        self.cfg = cfg
        self.symbols = symbols
        self.on_message = on_message
        self._tasks: list[asyncio.Task[None]] = []
        ws_url = cfg.data.ws_url or _BINANCE_WS_URL

        n = _MAX_SYMBOLS_PER_SHARD
        chunks = [symbols[i: i + n] for i in range(0, len(symbols), n)]
        self._shards = [
            _BinanceWSShard(cfg, chunk, on_message, shard_id, ws_url)
            for shard_id, chunk in enumerate(chunks)
        ]
        logging.info(
            "Binance WS manager: %d symbols -> %d shards (%d per shard)",
            len(symbols), len(self._shards), _MAX_SYMBOLS_PER_SHARD,
        )

    async def run_forever(self) -> None:
        for shard in self._shards:
            task = asyncio.create_task(shard.run_forever())
            self._tasks.append(task)
            await asyncio.sleep(_SHARD_START_DELAY_S)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        await asyncio.gather(*(shard.stop() for shard in self._shards))
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
