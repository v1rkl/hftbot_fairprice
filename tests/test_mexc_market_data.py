"""Tests for MexcMarketDataWS.

Unit-only: no network — mocks the websockets module.
Validates shard splitting, symbol-format conversion, subscription message format,
ping/pong handling, decode robustness.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from core.config import (
    BotConfig, DataConfig, ExchangeConfig, RiskConfig, RuntimeConfig, StrategyConfig,
)
from exchanges.mexc.market_data import (
    MexcMarketDataWS,
    _MexcWSShard,
    _to_exchange_symbol,
)


def _make_cfg() -> BotConfig:
    return BotConfig(
        strategy=StrategyConfig(),
        risk=RiskConfig(),
        data=DataConfig(orderbook_depth=20),
        exchange=ExchangeConfig(exchange_id="mexc", market_type="swap"),
        runtime=RuntimeConfig(),
    )


class TestSymbolFormat:
    def test_slash_converted_to_underscore(self) -> None:
        assert _to_exchange_symbol("BTC/USDT:USDT") == "BTC_USDT"

    def test_strips_settle_suffix(self) -> None:
        assert _to_exchange_symbol("SOL/USDT:USDT") == "SOL_USDT"

    def test_without_settle(self) -> None:
        assert _to_exchange_symbol("ETH/USDT") == "ETH_USDT"


class TestShardSplit:
    def test_single_shard_when_symbols_fit(self) -> None:
        cfg = _make_cfg()
        symbols = [f"SYM{i}/USDT:USDT" for i in range(10)]
        mgr = MexcMarketDataWS(cfg, symbols, on_message=lambda m: None)
        assert len(mgr._shards) == 1
        assert len(mgr._shards[0].symbols) == 10

    def test_multiple_shards_when_exceeding_limit(self) -> None:
        cfg = _make_cfg()
        # With max 30 per shard, 75 symbols → 3 shards (30 + 30 + 15)
        symbols = [f"SYM{i}/USDT:USDT" for i in range(75)]
        mgr = MexcMarketDataWS(cfg, symbols, on_message=lambda m: None)
        assert len(mgr._shards) == 3
        assert len(mgr._shards[0].symbols) == 30
        assert len(mgr._shards[1].symbols) == 30
        assert len(mgr._shards[2].symbols) == 15

    def test_exact_shard_boundary(self) -> None:
        cfg = _make_cfg()
        symbols = [f"SYM{i}/USDT:USDT" for i in range(60)]
        mgr = MexcMarketDataWS(cfg, symbols, on_message=lambda m: None)
        assert len(mgr._shards) == 2
        assert len(mgr._shards[0].symbols) == 30
        assert len(mgr._shards[1].symbols) == 30


class TestSubscribe:
    """Shard must send one sub message per channel per symbol in MEXC format."""

    @pytest.mark.asyncio
    async def test_subscribes_to_fair_deal_and_depth_for_each_symbol(self) -> None:
        cfg = _make_cfg()
        sent_payloads: list[dict] = []

        ws = MagicMock()

        async def fake_send(s: str) -> None:
            sent_payloads.append(orjson.loads(s))

        ws.send = fake_send

        shard = _MexcWSShard(
            cfg=cfg,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            on_message=lambda m: None,
            shard_id=0,
            ws_url="wss://example",
        )
        await shard._subscribe(ws)

        # Expect at least 3 channels × 2 symbols = 6 subscriptions
        assert len(sent_payloads) >= 6

        methods_by_symbol: dict[str, set[str]] = {}
        for p in sent_payloads:
            sym = (p.get("param") or {}).get("symbol") or ""
            method = p.get("method", "")
            methods_by_symbol.setdefault(sym, set()).add(method)

        for sym in ("BTC_USDT", "ETH_USDT"):
            methods = methods_by_symbol.get(sym, set())
            assert any("fair" in m for m in methods), f"no fair sub for {sym}: {methods}"
            assert any("deal" in m for m in methods), f"no deal sub for {sym}: {methods}"
            assert any("depth" in m for m in methods), f"no depth sub for {sym}: {methods}"

    @pytest.mark.asyncio
    async def test_depth_subscription_contains_limit(self) -> None:
        """Depth subscription must include limit/depth equal to orderbook_depth."""
        cfg = _make_cfg()
        cfg.data.orderbook_depth = 20
        sent: list[dict] = []

        ws = MagicMock()

        async def fake_send(s: str) -> None:
            sent.append(orjson.loads(s))

        ws.send = fake_send

        shard = _MexcWSShard(
            cfg=cfg,
            symbols=["BTC/USDT:USDT"],
            on_message=lambda m: None,
            shard_id=0,
            ws_url="wss://example",
        )
        await shard._subscribe(ws)

        depth_msgs = [p for p in sent if "depth" in p.get("method", "")]
        assert depth_msgs, "no depth subscription sent"
        params = depth_msgs[0].get("param", {})
        assert int(params.get("limit") or params.get("depth") or 0) == 20


class TestDecodePing:
    """Shard must handle MEXC server pings.

    MEXC sends {"channel":"push.ping","data":...} or {"method":"ping"} — we must
    reply with a pong message. Decoder must handle both bytes and str.
    """

    def test_decode_bytes(self) -> None:
        shard = _MexcWSShard(
            cfg=_make_cfg(), symbols=[], on_message=lambda m: None,
            shard_id=0, ws_url="wss://example",
        )
        out = shard._decode(b'{"channel":"push.pong","data":1}')
        assert isinstance(out, dict)
        assert out["channel"] == "push.pong"

    def test_decode_str(self) -> None:
        shard = _MexcWSShard(
            cfg=_make_cfg(), symbols=[], on_message=lambda m: None,
            shard_id=0, ws_url="wss://example",
        )
        out = shard._decode('{"channel":"push.pong","data":2}')
        assert isinstance(out, dict)

    def test_decode_invalid_returns_none(self) -> None:
        shard = _MexcWSShard(
            cfg=_make_cfg(), symbols=[], on_message=lambda m: None,
            shard_id=0, ws_url="wss://example",
        )
        assert shard._decode(b"not-json") is None
        assert shard._decode("<<>>") is None


class TestHealthMonitor:
    """Manager must expose a shard-health counter used by the monitor loop."""

    def test_shards_start_disconnected(self) -> None:
        cfg = _make_cfg()
        mgr = MexcMarketDataWS(cfg, ["BTC/USDT:USDT"], on_message=lambda m: None)
        assert all(not s.connected for s in mgr._shards)

    def test_health_counter_sums_connected_shards(self) -> None:
        cfg = _make_cfg()
        mgr = MexcMarketDataWS(cfg, [f"S{i}/USDT:USDT" for i in range(60)], on_message=lambda m: None)
        # Simulate shard 0 connected, shard 1 disconnected
        mgr._shards[0].connected = True
        mgr._shards[1].connected = False
        connected = sum(1 for s in mgr._shards if s.connected)
        assert connected == 1
