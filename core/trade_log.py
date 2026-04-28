from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _validate_log_path(raw: str | Path, base_dir: Path | None = None) -> Path:
    """Validate and resolve a log file path. Rejects '..' components and null bytes."""
    if "\x00" in str(raw):
        raise ValueError(f"Log file path must not contain null bytes: {raw!r}")
    p = Path(raw)
    if ".." in p.parts:
        raise ValueError(f"Log file path must not contain '..': {raw}")
    if p.is_absolute():
        return p
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()


def _restrict_permissions(path: Path) -> None:
    """Restrict file permissions to owner-only on Unix systems."""
    if hasattr(os, "chmod") and path.exists():
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


@dataclass(slots=True)
class MinuteSnapshot:
    ts_utc: str
    uptime_seconds: float
    symbols_watched: int
    open_positions: int
    paper_balance_usdt: float | None
    trades_closed_session: int
    rejection_counters: dict[str, int] | None = None


class MinuteLogWriter:
    def __init__(self, path: str | Path, interval_seconds: float, start_monotonic: float) -> None:
        self.path = _validate_log_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self._start_mono = start_monotonic
        self._last_write_mono: float = 0.0
        # Eagerly open to avoid TOCTOU race.
        self._fp: TextIO = self.path.open("a", encoding="utf-8")
        _restrict_permissions(self.path)

    def maybe_write(
        self,
        now_mono: float,
        *,
        symbols_watched: int,
        open_positions: int,
        paper_balance_usdt: float | None,
        trades_closed_session: int,
        rejection_counters: dict[str, int] | None = None,
    ) -> None:
        if self._last_write_mono == 0.0:
            self._last_write_mono = now_mono
        if now_mono - self._last_write_mono < self.interval_seconds:
            return
        self._last_write_mono = now_mono
        snap = MinuteSnapshot(
            ts_utc=_utc_iso(time.time()),
            uptime_seconds=now_mono - self._start_mono,
            symbols_watched=symbols_watched,
            open_positions=open_positions,
            paper_balance_usdt=paper_balance_usdt,
            trades_closed_session=trades_closed_session,
            rejection_counters=rejection_counters,
        )
        line = json.dumps(asdict(snap), ensure_ascii=False)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None and not self._fp.closed:
            self._fp.flush()
            self._fp.close()


class TradesLogWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = _validate_log_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Eagerly open to avoid TOCTOU race.
        self._fp: TextIO = self.path.open("a", encoding="utf-8")
        _restrict_permissions(self.path)

    def record_closed_trade(self, record: dict[str, Any]) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None and not self._fp.closed:
            self._fp.flush()
            self._fp.close()


class TradeJournalWriter:
    """Human-readable trade journal: per-session .txt + cumulative all_trades.txt."""

    def __init__(self, journal_dir: str | Path, config_name: str, cfg_snapshot: dict[str, Any]) -> None:
        # Sanitize config_name: strip path separators and '..' to prevent traversal.
        safe_name = config_name.replace("/", "_").replace("\\", "_").replace("..", "_")
        if not safe_name:
            safe_name = "unnamed"
        base = _validate_log_path(journal_dir)
        self._dir = base / safe_name
        self._dir.mkdir(parents=True, exist_ok=True)

        # Per-session file: journal/{config}/session_2026-03-24_14-30-05.txt
        ts = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
        session_path = self._dir / f"session_{ts}.txt"
        self._session_fp: TextIO = session_path.open("w", encoding="utf-8")
        _restrict_permissions(session_path)

        # Cumulative file: journal/{config}/all_trades.txt
        self._all_path = self._dir / "all_trades.txt"
        is_new = not self._all_path.exists() or self._all_path.stat().st_size == 0
        self._all_fp: TextIO = self._all_path.open("a", encoding="utf-8")
        _restrict_permissions(self._all_path)

        self._trade_num: int = 0
        self._session_wins: int = 0
        self._session_losses: int = 0
        self._session_pnl: float = 0.0
        self._start_balance: float = cfg_snapshot.get("start_balance", 0.0)

        # Write header to session file.
        self._write_session_header(config_name, cfg_snapshot)
        # Write separator to cumulative file.
        if is_new:
            self._all_fp.write(f"{'=' * 80}\n")
            self._all_fp.write(f"  ALL TRADES — {config_name}\n")
            self._all_fp.write(f"{'=' * 80}\n\n")
            self._all_fp.flush()
        self._all_fp.write(f"--- Session started {ts} ---\n")
        self._all_fp.flush()

    def _write_session_header(self, config_name: str, snap: dict[str, Any]) -> None:
        fp = self._session_fp
        w = 80
        fp.write(f"{'=' * w}\n")
        fp.write(f"{'HFT BOT TRADE JOURNAL':^{w}}\n")
        fp.write(f"{config_name:^{w}}\n")
        fp.write(f"{'=' * w}\n")
        fp.write(f"Started:  {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        bal = snap.get("start_balance", 0)
        mode = "paper" if snap.get("paper_mode", True) else "LIVE"
        fp.write(f"Balance:  {bal:.2f} USDT ({mode})\n")
        fp.write(f"Exchange: {snap.get('exchange_id', '?')} | Market: {snap.get('market_type', '?')}\n\n")

        strat = snap.get("strategy", {})
        fp.write("Strategy:\n")
        fp.write(f"  fair_rise_threshold: {strat.get('fair_rise_threshold', 0):.3f}"
                 f" ({strat.get('fair_rise_threshold', 0) * 100:.1f}%)\n")
        fp.write(f"  fair_move_window:    {strat.get('fair_move_window_seconds', 0):.1f}s\n")
        fp.write(f"  hold_seconds:        {strat.get('hold_seconds', 0):.1f}s\n")
        fp.write(f"  signal_confirm:      {strat.get('signal_confirm_seconds', 0):.1f}s\n")
        fp.write(f"  max_spread:          {strat.get('max_spread_pct', 0):.1f}%\n\n")

        risk = snap.get("risk", {})
        fp.write("Risk:\n")
        fp.write(f"  position_size:       {risk.get('position_size_pct', 0) * 100:.0f}%\n")
        trailing = "ON" if risk.get("trailing_enabled", False) else "OFF"
        fp.write(f"  trailing:            {trailing}"
                 f" (activate {risk.get('trailing_activation_pct', 0) * 100:.1f}%,"
                 f" lock {risk.get('trailing_lock_pct', 0) * 100:.0f}%)\n")
        fp.write(f"  TP: {risk.get('tp1_margin_multiplier', 0):.1f}x margin"
                 f" -> full close\n")
        fp.write(f"  stop_loss:           {risk.get('stop_loss_before_liq_pct', 0) * 100:.1f}%"
                 f" before liquidation\n")
        fp.write(f"  symbol_ban:          {risk.get('symbol_max_consecutive_losses', 0)} losses"
                 f" -> {risk.get('symbol_ban_duration_seconds', 0) / 3600:.0f}h ban\n")
        fp.write(f"\n{'-' * w}\n\n")
        fp.flush()

    def record_trade(self, record: dict[str, Any]) -> None:
        self._trade_num += 1
        pnl = record.get("pnl_quote_usdt", 0.0)
        if pnl >= 0:
            self._session_wins += 1
        else:
            self._session_losses += 1
        self._session_pnl += pnl

        line = self._format_trade(self._trade_num, record)
        self._session_fp.write(line)
        self._session_fp.flush()
        self._all_fp.write(line)
        self._all_fp.flush()

    @staticmethod
    def _format_trade(num: int, r: dict[str, Any]) -> str:
        symbol = r.get("symbol", "?")
        side = (r.get("side_open", "?")).upper()
        entry_p = r.get("entry_price", 0.0)
        exit_p = r.get("exit_price", 0.0)
        qty = r.get("qty_base", 0.0)
        pnl = r.get("pnl_quote_usdt", 0.0)
        pnl_pct = r.get("pnl_pct_on_notional", 0.0)
        entry_t = r.get("entry_time_utc", "?")
        signal = r.get("fair_rise_at_entry_pct", None)
        reason = r.get("close_reason", "")
        leverage = r.get("leverage", 0)
        hold_s = r.get("hold_seconds", 0.0)
        best_pnl = r.get("best_pnl_pct", None)

        pnl_sign = "+" if pnl >= 0 else ""
        side_word = "LONG" if side == "BUY" else "SHORT"

        # Compact timestamp: just time portion.
        if "T" in str(entry_t):
            display_time = str(entry_t).replace("T", " ").split("+")[0]
        else:
            display_time = str(entry_t)

        lines = f"#{num:<3} {display_time}  {symbol}\n"
        lines += f"    {side_word}  entry={entry_p:.8f}  exit={exit_p:.8f}  qty={qty:.6f}"
        if leverage:
            lines += f"  lev={leverage:.0f}x"
        lines += "\n"
        lines += f"    PnL: {pnl_sign}{pnl:.4f} USDT ({pnl_sign}{pnl_pct:.2f}%)"
        if hold_s > 0:
            lines += f"  hold: {hold_s:.1f}s"
        lines += "\n"

        detail_parts: list[str] = []
        if signal is not None:
            detail_parts.append(f"signal: fair {'+' if signal >= 0 else ''}{signal:.2f}%")
        if reason:
            reason_display = reason.replace("_", " ")
            if best_pnl is not None and "trailing" in reason:
                reason_display += f" (best {best_pnl * 100:+.1f}%)"
            detail_parts.append(f"close: {reason_display}")
        if detail_parts:
            lines += f"    {' | '.join(detail_parts)}\n"
        lines += "\n"
        return lines

    def write_summary(self, final_balance: float | None = None) -> None:
        total = self._session_wins + self._session_losses
        wr = (self._session_wins / total * 100.0) if total > 0 else 0.0
        bal_str = f"{final_balance:.2f} USDT" if final_balance is not None else "N/A"
        pnl_sign = "+" if self._session_pnl >= 0 else ""
        pnl_pct = (self._session_pnl / self._start_balance * 100.0) if self._start_balance > 0 else 0.0

        summary = (
            f"{'-' * 80}\n"
            f"Session Summary:\n"
            f"  Trades: {total}  |  Won: {self._session_wins}  |"
            f"  Lost: {self._session_losses}  |  Win rate: {wr:.1f}%\n"
            f"  Total PnL: {pnl_sign}{self._session_pnl:.4f} USDT"
            f" ({pnl_sign}{pnl_pct:.2f}%)\n"
            f"  Balance: {bal_str}\n"
            f"{'=' * 80}\n"
        )
        self._session_fp.write(summary)
        self._session_fp.flush()
        self._all_fp.write(summary + "\n")
        self._all_fp.flush()

    def close(self) -> None:
        for fp in (self._session_fp, self._all_fp):
            if fp is not None and not fp.closed:
                fp.flush()
                fp.close()


def pnl_linear_usdt(
    *,
    side_open: str,
    qty_base: float,
    entry_price: float,
    exit_price: float,
    contract_size: float = 1.0,
) -> float:
    """USDT PnL for a linear contract.

    qty_base is in contracts (as stored by the bot).  For exchanges that trade
    in contracts (e.g. Gate.io STO contractSize=10, RLS contractSize=100) the
    real token quantity is qty_base * contract_size.
    """
    tokens = qty_base * contract_size
    if side_open == "buy":
        return tokens * (exit_price - entry_price)
    if side_open == "sell":
        return tokens * (entry_price - exit_price)
    return 0.0
