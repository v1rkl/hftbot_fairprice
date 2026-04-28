from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")


@dataclass(slots=True)
class StrategyConfig:
    fair_rise_threshold: float = 0.08
    fair_move_window_seconds: float = 5.0
    hold_seconds: float = 60.0
    cooldown_seconds: float = 30.0
    tick_offset_entry: int = 3
    tick_offset_close: int = 6
    quote_size_usdt: float = 30.0
    fill_timeout_ms: int = 2500
    close_with_market_fallback: bool = True
    signal_confirm_seconds: float = 3.0
    signal_max_pullback: float = 0.04
    last_max_move_pct: float = 0.02
    max_open_positions: int = 2
    max_spread_pct: float = 5.0
    spread_to_signal_ratio: float = 0.15
    min_depth_quote_usdt: float = 0.0
    depth_to_notional_ratio: float = 1.0
    entry_follow_timeout_seconds: float = 0.0  # Phase 3: 0 = disabled
    entry_follow_min_pct: float = 0.003        # min price move in signal direction
    reverse_signal: bool = False               # flip buy/sell at entry (momentum -> fade)
    require_last_chg_aligned: bool = False     # reject if last_chg direction opposes signal side
    adverse_last_chg_min_pct: float = 0.003    # min counter-move magnitude to treat as adverse (0.3%)
    max_fair_rise_pct: float = 0.0             # upper cap on |fair_change|; 0 = disabled


@dataclass(slots=True)
class RiskConfig:
    position_size_pct: float = 0.10
    stop_loss_margin_fraction: float = 0.65       # fraction of margin to risk (65%)
    stop_loss_high_lev_margin_fraction: float = 0.90  # for high-leverage (e.g. 125x)
    stop_loss_high_lev_threshold: float = 100.0   # leverage >= this uses high_lev fraction
    symbol_max_consecutive_losses: int = 2
    symbol_ban_duration_seconds: float = 3600.0
    trailing_enabled: bool = True
    trailing_activation_pct: float = 0.015        # tier3 (>= lev_tier2_max)
    trailing_activation_pct_tier1: float = 0.010  # tier1 (< lev_tier1_max)
    trailing_activation_pct_tier2: float = 0.012  # tier2 (>= lev_tier1_max, < lev_tier2_max)
    trailing_lock_pct: float = 0.55               # lock 55% of peak gain; 0.3 caused loss-on-trigger bug (effective trail above entry)
    trailing_min_buffer_pct: float = 0.0          # min distance from current price; 0 = disabled
    tp_cooldown_seconds: float = 0.0             # ban symbol N seconds after take_profit close; 0 = disabled
    tp1_margin_multiplier: float = 4.0            # tier3 (>= lev_tier2_max): exit at 300% margin gain
    tp1_margin_multiplier_tier1: float = 1.6      # tier1 (< lev_tier1_max):  exit at  60% margin gain
    tp1_margin_multiplier_tier2: float = 2.0      # tier2 (>= lev_tier1_max, < lev_tier2_max): exit at 100% margin gain
    lev_tier1_max: float = 30.0                   # leverage < this → tier1
    lev_tier2_max: float = 50.0                   # leverage < this → tier2; >= this → tier3
    stop_loss_margin_fraction_tier1: float = 0.45 # tier1 tighter SL
    stop_loss_margin_fraction_tier2: float = 0.55 # tier2 mid SL


@dataclass(slots=True)
class DataConfig:
    ws_url: str = ""
    orderbook_depth: int = 20
    reconnect_backoff_ms_min: int = 300
    reconnect_backoff_ms_max: int = 5000
    processing_latency_warn_ms: float = 50.0
    max_fair_staleness_seconds: float = 0.0


@dataclass(slots=True)
class ExchangeConfig:
    exchange_id: str = "mexc"
    market_type: str = "swap"
    min_max_leverage: float = 20.0
    symbols_allowlist: list[str] | None = None
    symbols_denylist: list[str] | None = None
    max_maintenance_rate: float = 0.0  # 0 = disabled; e.g. 0.03 skips symbols with mm_rate > 3%


@dataclass(slots=True)
class RuntimeConfig:
    paper_mode: bool = True
    paper_balance_usdt: float = 100.0
    max_run_seconds: float = 0.0
    monitor_interval_seconds: float = 5.0
    ws_debug_messages: int = 0
    minute_snapshot_interval_seconds: float = 60.0
    minute_log_file: str = "logs/minute.log"
    trades_log_file: str = "logs/trades.jsonl"
    log_level: str = "INFO"
    config_name: str = "default"
    journal_dir: str = "logs/journal"
    position_state_file: str = "data/positions.json"
    reconciliation_interval_seconds: float = 60.0


@dataclass(slots=True)
class BotConfig:
    strategy: StrategyConfig
    data: DataConfig
    exchange: ExchangeConfig
    runtime: RuntimeConfig
    risk: RiskConfig

    @staticmethod
    def load(path: str | Path) -> BotConfig:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config {p}: {e}") from e

        cfg = BotConfig(
            strategy=_safe_construct(StrategyConfig, raw.get("strategy", {}), "strategy"),
            data=_safe_construct(DataConfig, raw.get("data", {}), "data"),
            exchange=_safe_construct(ExchangeConfig, raw.get("exchange", {}), "exchange"),
            runtime=_safe_construct(RuntimeConfig, raw.get("runtime", {}), "runtime"),
            risk=_safe_construct(RiskConfig, raw.get("risk", {}), "risk"),
        )
        _validate(cfg)
        return cfg

    def validate_ws_url(self) -> None:
        """Validate that ws_url uses wss:// scheme."""
        url = self.data.ws_url
        if not url:
            if not self.runtime.paper_mode:
                raise ValueError("data.ws_url must be set for live (non-paper) mode")
            return
        parsed = urlparse(url)
        if parsed.scheme != "wss":
            raise ValueError(
                f"WebSocket URL must use wss:// scheme, got: {url!r}"
            )


def _safe_construct(cls: type[T], raw: dict[str, Any], section: str) -> T:
    """Construct a dataclass, rejecting unknown keys with a clear message."""
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{section}' must be a JSON object, got {type(raw).__name__}")
    # type: ignore[arg-type]: mypy cannot prove T is a dataclass at this call site,
    # but all callers (BotConfig.load) always pass dataclass types — safe by construction.
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw.keys()) - known
    if unknown:
        raise ValueError(
            f"Unknown keys in config section '{section}': {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    return cls(**raw)


def _validate(cfg: BotConfig) -> None:
    """Validate config value bounds."""
    s = cfg.strategy
    if s.max_open_positions < 1:
        raise ValueError(f"max_open_positions must be >= 1, got {s.max_open_positions}")
    if s.quote_size_usdt <= 0:
        raise ValueError(f"quote_size_usdt must be positive, got {s.quote_size_usdt}")
    if s.fill_timeout_ms <= 0:
        raise ValueError(f"fill_timeout_ms must be positive, got {s.fill_timeout_ms}")
    if s.hold_seconds <= 0:
        raise ValueError(f"hold_seconds must be positive, got {s.hold_seconds}")
    if s.cooldown_seconds < 0:
        raise ValueError(f"cooldown_seconds must be >= 0, got {s.cooldown_seconds}")
    if s.fair_rise_threshold <= 0:
        raise ValueError(f"fair_rise_threshold must be positive, got {s.fair_rise_threshold}")
    if s.signal_confirm_seconds < 0:
        raise ValueError(f"signal_confirm_seconds must be >= 0, got {s.signal_confirm_seconds}")

    d = cfg.data
    if d.max_fair_staleness_seconds < 0:
        raise ValueError(f"max_fair_staleness_seconds must be >= 0, got {d.max_fair_staleness_seconds}")

    rt = cfg.runtime
    if rt.reconciliation_interval_seconds < 0:
        raise ValueError(f"reconciliation_interval_seconds must be >= 0, got {rt.reconciliation_interval_seconds}")

    r = cfg.risk
    if not (0 < r.position_size_pct <= 1.0):
        raise ValueError(f"risk.position_size_pct must be in (0, 1.0], got {r.position_size_pct}")
    if r.trailing_enabled and not (0 < r.trailing_lock_pct < 1.0):
        raise ValueError(f"trailing_lock_pct must be in (0, 1.0), got {r.trailing_lock_pct}")
    if r.trailing_enabled and not (0 < r.trailing_activation_pct < 1.0):
        raise ValueError(
            f"trailing_activation_pct must be in (0, 1.0) when trailing is enabled, "
            f"got {r.trailing_activation_pct}"
        )
    if r.trailing_enabled and not (0 < r.trailing_activation_pct_tier1 < 1.0):
        raise ValueError(
            f"trailing_activation_pct_tier1 must be in (0, 1.0) when trailing is enabled, "
            f"got {r.trailing_activation_pct_tier1}"
        )
    if r.trailing_enabled and not (0 < r.trailing_activation_pct_tier2 < 1.0):
        raise ValueError(
            f"trailing_activation_pct_tier2 must be in (0, 1.0) when trailing is enabled, "
            f"got {r.trailing_activation_pct_tier2}"
        )
    if r.tp1_margin_multiplier < 1.0:
        raise ValueError(f"tp1_margin_multiplier must be >= 1.0 (exit at profit), got {r.tp1_margin_multiplier}")
    if r.tp1_margin_multiplier_tier1 < 1.0:
        raise ValueError(f"tp1_margin_multiplier_tier1 must be >= 1.0 (exit at profit), got {r.tp1_margin_multiplier_tier1}")
    if r.tp1_margin_multiplier_tier2 < 1.0:
        raise ValueError(f"tp1_margin_multiplier_tier2 must be >= 1.0 (exit at profit), got {r.tp1_margin_multiplier_tier2}")
    if r.lev_tier1_max <= 0:
        raise ValueError(f"lev_tier1_max must be positive, got {r.lev_tier1_max}")
    if r.lev_tier2_max <= r.lev_tier1_max:
        raise ValueError(
            f"lev_tier2_max ({r.lev_tier2_max}) must be greater than lev_tier1_max ({r.lev_tier1_max})"
        )
    if r.stop_loss_high_lev_threshold < r.lev_tier2_max:
        raise ValueError(
            f"stop_loss_high_lev_threshold ({r.stop_loss_high_lev_threshold}) must be >= "
            f"lev_tier2_max ({r.lev_tier2_max}) to avoid overlap"
        )
    if not (0 < r.stop_loss_margin_fraction < 1.0):
        raise ValueError(f"stop_loss_margin_fraction must be in (0, 1.0), got {r.stop_loss_margin_fraction}")
    if not (0 < r.stop_loss_margin_fraction_tier1 < 1.0):
        raise ValueError(f"stop_loss_margin_fraction_tier1 must be in (0, 1.0), got {r.stop_loss_margin_fraction_tier1}")
    if not (0 < r.stop_loss_margin_fraction_tier2 < 1.0):
        raise ValueError(f"stop_loss_margin_fraction_tier2 must be in (0, 1.0), got {r.stop_loss_margin_fraction_tier2}")
    if not (0 < r.stop_loss_high_lev_margin_fraction < 1.0):
        raise ValueError(f"stop_loss_high_lev_margin_fraction must be in (0, 1.0), got {r.stop_loss_high_lev_margin_fraction}")
    if r.stop_loss_high_lev_threshold <= 0:
        raise ValueError(f"stop_loss_high_lev_threshold must be positive, got {r.stop_loss_high_lev_threshold}")
    if r.trailing_min_buffer_pct < 0:
        raise ValueError(f"trailing_min_buffer_pct must be >= 0, got {r.trailing_min_buffer_pct}")

    cfg.validate_ws_url()
