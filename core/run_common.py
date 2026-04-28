from __future__ import annotations

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

from .config import BotConfig
from .trade_log import _validate_log_path


def build_arg_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config, help="Path to config JSON")
    p.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    p.add_argument("--paper-balance", type=float, default=None)
    p.add_argument("--run-seconds", type=float, default=None)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--threshold-percent", type=float, default=None)
    p.add_argument("--allowlist", type=str, default=None)
    p.add_argument("--hold-seconds", type=float, default=None)
    p.add_argument("--cooldown-seconds", type=float, default=None)
    p.add_argument("--ws-debug", type=int, default=None)
    return p


def apply_arg_overrides(cfg: BotConfig, args: argparse.Namespace) -> None:
    if args.live:
        if os.getenv("CONFIRM_LIVE_TRADING") != "YES_I_KNOW":
            raise SystemExit(
                "ERROR: To enable live trading, set environment variable "
                "CONFIRM_LIVE_TRADING=YES_I_KNOW"
            )
        cfg.runtime.paper_mode = False
        logging.critical("LIVE TRADING ENABLED -- real orders will be placed")
    if args.paper_balance is not None:
        cfg.runtime.paper_balance_usdt = float(args.paper_balance)
    if args.run_seconds is not None:
        cfg.runtime.max_run_seconds = float(args.run_seconds)
    if args.threshold_percent is not None:
        cfg.strategy.fair_rise_threshold = float(args.threshold_percent) / 100.0
    if args.allowlist is not None:
        cfg.exchange.symbols_allowlist = [s.strip() for s in args.allowlist.split(",") if s.strip()]
    if args.hold_seconds is not None:
        cfg.strategy.hold_seconds = float(args.hold_seconds)
    if args.cooldown_seconds is not None:
        cfg.strategy.cooldown_seconds = float(args.cooldown_seconds)
    if args.ws_debug is not None:
        cfg.runtime.ws_debug_messages = int(args.ws_debug)


def setup_logging(cfg: BotConfig, log_file: str | None) -> None:
    level_name = cfg.runtime.log_level.upper()
    log_level = getattr(logging, level_name, None)
    if log_level is None:
        logging.warning("Unrecognized log_level %r, defaulting to INFO", cfg.runtime.log_level)
        log_level = logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(log_level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)
    if log_file:
        log_path = _validate_log_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)
