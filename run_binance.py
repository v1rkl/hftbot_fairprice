from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.config import BotConfig
from core.engine import TradingEngine
from core.run_common import apply_arg_overrides, build_arg_parser, setup_logging
from exchanges.binance.execution import BinanceExecutionEngine
from exchanges.binance.market_data import BinanceMarketDataWS
from exchanges.binance.message_parser import BinanceMessageParser


async def _amain() -> None:
    default_cfg = str(Path(__file__).parent / "config.binance.paper.json")
    parser = build_arg_parser("Binance USDT-M Futures HFT bot -- fair price momentum strategy", default_cfg)
    args = parser.parse_args()

    cfg = BotConfig.load(args.config)
    apply_arg_overrides(cfg, args)
    setup_logging(cfg, args.log_file)

    logging.info(
        "Binance | profile=%s paper=%s threshold=%.2f%% last_max=%.2f%% window=%.1fs hold=%.1fs",
        cfg.runtime.config_name, cfg.runtime.paper_mode,
        cfg.strategy.fair_rise_threshold * 100.0,
        cfg.strategy.last_max_move_pct * 100.0,
        cfg.strategy.fair_move_window_seconds,
        cfg.strategy.hold_seconds,
    )
    if cfg.runtime.paper_mode:
        logging.info("Paper balance: %.2f USDT", cfg.runtime.paper_balance_usdt)

    engine = TradingEngine(
        cfg=cfg,
        market_data_cls=BinanceMarketDataWS,
        execution_cls=BinanceExecutionEngine,
        parser_cls=BinanceMessageParser,
    )
    try:
        await engine.run()
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(_amain())
