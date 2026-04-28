from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, NamedTuple

from .config import BotConfig
from .interfaces import AbstractExecutionEngine, AbstractMarketDataWS, AbstractMessageParser
from .models import VALID_SIDES, SymbolState
from .paper import PaperAccount
from .position_store import PositionStore
from .rejection_counters import SignalRejectionCounters
from .risk import RiskManager, compute_stop_loss_price, update_trailing_state
from .signal import compute_deviation
from .trade_log import MinuteLogWriter, TradeJournalWriter, TradesLogWriter, pnl_linear_usdt


class _CloseSnapshot(NamedTuple):
    """Frozen snapshot of fields needed to close a position, captured under lock."""
    state: SymbolState
    entry_price: float | None
    side: str
    entry_wall: float
    signal_at_entry: float | None
    close_reason: str
    leverage: float
    best_pnl_pct: float

# Prevent division by zero for near-zero prices.
_MIN_PRICE_GUARD: float = 1e-12
# Reset signal latch when fair_change drops below 30% of threshold.
_SIGNAL_RESET_HYSTERESIS: float = 0.3
# Halt signals for a symbol after this many consecutive order errors.
_MAX_CONSECUTIVE_ERRORS: int = 3
# Cooldown (seconds) after consecutive error limit is hit.
_ERROR_COOLDOWN_SECONDS: float = 300.0
# Max symbols to log in periodic monitor output.
_MONITOR_LOG_MAX_SYMBOLS: int = 5


class TradingEngine:
    def __init__(
        self,
        cfg: BotConfig,
        market_data_cls: type[AbstractMarketDataWS],
        execution_cls: type[AbstractExecutionEngine],
        parser_cls: type[AbstractMessageParser],
    ) -> None:
        self.cfg = cfg
        self.market_data_cls = market_data_cls
        self.execution_cls = execution_cls
        self.parser_cls = parser_cls
        self.parser: AbstractMessageParser = parser_cls()

        self.paper_account: PaperAccount | None = None
        if cfg.runtime.paper_mode:
            self.paper_account = PaperAccount(cfg.runtime.paper_balance_usdt)

        self.exec: AbstractExecutionEngine = execution_cls(cfg, paper_account=self.paper_account)
        # start_balance is set properly in run() after exchange init for live mode.
        start_bal = cfg.runtime.paper_balance_usdt if cfg.runtime.paper_mode else 0.0
        self.risk = RiskManager(cfg.risk, cfg.strategy.cooldown_seconds, start_bal)
        self.states: dict[str, SymbolState] = {}
        self.ws: AbstractMarketDataWS | None = None
        self._lock = asyncio.Lock()
        self._running = False
        self._last_monitor_monotonic: float = 0.0
        self._minute_log: MinuteLogWriter | None = None
        self._trades_log: TradesLogWriter | None = None
        self._journal: TradeJournalWriter | None = None
        self._trades_closed_session: int = 0
        self._run_start_monotonic: float = 0.0
        # Pending-opens counter: incremented before async order call, decremented after.
        self._pending_opens: int = 0
        # Message queue for backpressure instead of unbounded create_task.
        self._msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=5000)
        # Persistent position state for crash recovery.
        psf = (cfg.runtime.position_state_file or "").strip()
        self._position_store: PositionStore | None = PositionStore(psf) if psf else None
        self._positions_dirty: bool = False
        # Periodic reconciliation.
        self._last_reconcile_monotonic: float = 0.0
        # Signal rejection counters.
        self.rejection_counters = SignalRejectionCounters()
        # Dropped WS messages (queue full).
        self._dropped_messages: int = 0

    async def run(self) -> None:
        markets = await self.exec.initialize()

        # In live mode, fetch real account balance for session PnL% reporting.
        if not self.cfg.runtime.paper_mode:
            try:
                live_bal = await self.exec.fetch_account_balance()
                if live_bal > 0:
                    self.risk.start_balance = live_bal
                    logging.info("Live account balance: %.2f USDT", live_bal)
                else:
                    raise RuntimeError(
                        f"Live account balance is zero or negative ({live_bal:.2f} USDT). "
                        "Check API key permissions and that the account is funded."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch live balance: {exc}") from exc

        symbols = self._select_symbols(markets)
        if not symbols:
            raise RuntimeError("No symbols found with max leverage >= configured threshold.")

        for symbol in symbols:
            market = markets[symbol]
            tick = self._tick_size(market)
            lev = self._max_leverage(market) or 1.0
            cs = float(market.get("contractSize") or 1.0)
            mm = float(market.get("info", {}).get("maintenance_rate") or 0.0)
            self.states[symbol] = SymbolState(symbol=symbol, tick_size=tick, max_leverage=lev, contract_size=cs, maintenance_rate=mm)

        thr = self.cfg.exchange.min_max_leverage
        if thr <= 0:
            logging.info("Selected %d symbols (no leverage filter)", len(symbols))
        else:
            logging.info("Selected %d symbols (max leverage >= %.0fx)", len(symbols), thr)

        # Reconcile positions with exchange before starting (critical for live mode).
        try:
            await self._reconcile_startup()
        except Exception:
            await self.exec.close()
            raise

        self.ws = self.market_data_cls(self.cfg, symbols, self._on_ws_message)
        self._running = True
        self._run_start_monotonic = time.monotonic()

        self._init_log_writers()

        ws_task = asyncio.create_task(self.ws.run_forever())
        timer_task = asyncio.create_task(self._timer_loop())
        consumer_task = asyncio.create_task(self._message_consumer())

        start = time.monotonic()
        try:
            while self._running:
                if self.cfg.runtime.max_run_seconds and self.cfg.runtime.max_run_seconds > 0:
                    if time.monotonic() - start >= self.cfg.runtime.max_run_seconds:
                        logging.info("Max run time reached: %.1fs", self.cfg.runtime.max_run_seconds)
                        break
                await asyncio.sleep(0.25)
        finally:
            self._running = False
            if self.ws:
                await self.ws.stop()
            await asyncio.gather(ws_task, timer_task, consumer_task, return_exceptions=True)
            self._close_aux_logs()

    async def shutdown(self) -> None:
        self._running = False
        if self.ws:
            await self.ws.stop()
        # Final persist under lock before shutdown.
        if self._position_store is not None:
            async with self._lock:
                self._position_store.save(self.states)
        await self.exec.close()
        self._close_aux_logs()

    def _close_aux_logs(self) -> None:
        rc = self.rejection_counters
        nz = rc.nonzero_dict()
        if nz:
            logging.info("SESSION REJECTION SUMMARY: %s", rc.summary_line())
            logging.info(
                "Signal funnel: confirmed=%d attempted=%d filled=%d rejected=%d",
                rc.signals_confirmed, rc.entries_attempted,
                rc.entries_filled, rc.total_rejected(),
            )
        if self._dropped_messages > 0:
            logging.warning("SESSION: %d WS messages dropped (queue full)", self._dropped_messages)
        if self._journal is not None:
            bal = self.paper_account.quote_balance if self.paper_account else None
            self._journal.write_summary(final_balance=bal)
            self._journal.close()
            self._journal = None
        if self._minute_log is not None:
            self._minute_log.close()
            self._minute_log = None
        if self._trades_log is not None:
            self._trades_log.close()
            self._trades_log = None

    def _init_log_writers(self) -> None:
        """Initialise minute-snapshot, trades, and journal log writers."""
        mf = (self.cfg.runtime.minute_log_file or "").strip()
        interval = float(self.cfg.runtime.minute_snapshot_interval_seconds or 0.0)
        if mf and interval > 0:
            self._minute_log = MinuteLogWriter(mf, interval, self._run_start_monotonic)
            logging.info("Minute snapshots -> %s every %.0fs", mf, interval)

        tf = (self.cfg.runtime.trades_log_file or "").strip()
        if tf:
            self._trades_log = TradesLogWriter(tf)
            logging.info("Trades log -> %s", tf)

        jd = (self.cfg.runtime.journal_dir or "").strip()
        if jd:
            cfg_snap = {
                "start_balance": self.cfg.runtime.paper_balance_usdt if self.cfg.runtime.paper_mode else 0.0,
                "paper_mode": self.cfg.runtime.paper_mode,
                "exchange_id": self.cfg.exchange.exchange_id,
                "market_type": self.cfg.exchange.market_type,
                "strategy": asdict(self.cfg.strategy),
                "risk": asdict(self.cfg.risk),
            }
            self._journal = TradeJournalWriter(jd, self.cfg.runtime.config_name, cfg_snap)
            logging.info("Trade journal -> %s/%s/", jd, self.cfg.runtime.config_name)

    def _select_symbols(self, markets: dict[str, Any]) -> list[str]:
        min_lev = self.cfg.exchange.min_max_leverage
        max_mm = self.cfg.exchange.max_maintenance_rate
        selected: list[str] = []
        allow = set(self.cfg.exchange.symbols_allowlist or [])
        deny = set(self.cfg.exchange.symbols_denylist or [])

        for symbol, m in markets.items():
            if m.get("type") not in {"swap", "future"}:
                continue
            if allow and symbol not in allow:
                continue
            if symbol in deny:
                continue
            if max_mm > 0:
                mm_rate = float((m.get("info") or {}).get("maintenance_rate") or 0.0)
                if mm_rate > max_mm:
                    logging.warning(
                        "Skipping %s — maintenance_rate=%.4f > max=%.4f",
                        symbol, mm_rate, max_mm,
                    )
                    continue
            if min_lev <= 0:
                selected.append(symbol)
                continue
            max_lev = self._max_leverage(m)
            if max_lev is not None and max_lev >= min_lev:
                selected.append(symbol)
        return selected

    @staticmethod
    def _max_leverage(market: dict[str, Any]) -> float | None:
        lim = (market.get("limits") or {}).get("leverage") or {}
        if lim.get("max") is not None:
            return float(lim["max"])
        info = market.get("info") or {}
        for key in ("maxLeverage", "max_leverage", "leverage"):
            value = info.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _tick_size(market: dict[str, Any]) -> float:
        precision = market.get("precision") or {}
        p = precision.get("price")
        if isinstance(p, (int, float)) and p > 0:
            return float(p)
        limits_price = (market.get("limits") or {}).get("price") or {}
        min_px = limits_price.get("min")
        if isinstance(min_px, (int, float)) and min_px > 0:
            return float(min_px)
        return 0.0001

    async def _timer_loop(self) -> None:
        while self._running:
            await asyncio.sleep(0.05)
            now = time.monotonic()

            # Phase 1: collect due-to-close positions + monitoring (under lock, no I/O).
            to_close: list[_CloseSnapshot] = []
            async with self._lock:
                for state in self.states.values():
                    if state.position_qty == 0.0 or state.close_queued or state.has_pending_open:
                        continue

                    # --- Trailing stop + take-profit logic ---
                    action = update_trailing_state(state, self.cfg.risk)

                    if action == "tp1_full":
                        logging.info(
                            "TP TRIGGERED %s side=%s — closing 100%% at %.1fx margin",
                            state.symbol, (state.side or "").upper(),
                            self.cfg.risk.tp1_margin_multiplier,
                        )
                        state.close_reason = "take_profit"
                        state.close_queued = True
                        to_close.append(self._snapshot_for_close(state))
                        continue
                    elif action == "trailing_stop_hit":
                        logging.warning(
                            "TRAILING STOP HIT %s side=%s last=%.8f trail=%.8f",
                            state.symbol, (state.side or "").upper(),
                            state.last_price or 0.0, state.trailing_stop_price or 0.0,
                        )
                        state.close_reason = "trailing_stop"
                        state.close_queued = True
                        to_close.append(self._snapshot_for_close(state))
                        continue

                    # --- Fixed stop-loss (before trailing activates) ---
                    stop_hit = False
                    if not state.trailing_active:
                        if state.stop_loss_price is not None and state.last_price is not None:
                            if state.side == "buy" and state.last_price <= state.stop_loss_price:
                                stop_hit = True
                            elif state.side == "sell" and state.last_price >= state.stop_loss_price:
                                stop_hit = True
                    if stop_hit:
                        logging.warning(
                            "STOP-LOSS HIT %s side=%s last=%.8f stop=%.8f",
                            state.symbol, (state.side or "").upper(),
                            state.last_price or 0.0, state.stop_loss_price or 0.0,
                        )
                        state.close_reason = "stop_loss"
                        state.close_queued = True
                        to_close.append(self._snapshot_for_close(state))
                    elif self._check_entry_follow_timeout(state, now):
                        logging.warning(
                            "ENTRY FOLLOW TIMEOUT %s side=%s — price did not follow "
                            "signal within %.1fs (min_pct=%.3f%%) closing early",
                            state.symbol, (state.side or "").upper(),
                            self.cfg.strategy.entry_follow_timeout_seconds,
                            self.cfg.strategy.entry_follow_min_pct * 100,
                        )
                        state.close_reason = "entry_follow_timeout"
                        state.close_queued = True
                        to_close.append(self._snapshot_for_close(state))
                    elif now >= state.close_due_monotonic:
                        state.close_reason = "hold_timeout"
                        state.close_queued = True
                        to_close.append(self._snapshot_for_close(state))

                interval = float(self.cfg.runtime.monitor_interval_seconds or 0.0)
                if interval > 0 and (now - self._last_monitor_monotonic) >= interval:
                    self._last_monitor_monotonic = now
                    states = [
                        s for s in self.states.values()
                        if s.fair_price is not None or s.last_price is not None
                    ]
                    for s in states[:_MONITOR_LOG_MAX_SYMBOLS]:
                        dev = compute_deviation(s)
                        logging.info(
                            "MONITOR %s last=%.8f fair=%.8f mid=%.8f "
                            "dev=%.4f%% pos_qty=%.6f",
                            s.symbol,
                            s.last_price or 0.0,
                            s.fair_price or 0.0,
                            (s.book.mid or 0.0),
                            (dev * 100.0 if dev is not None else 0.0),
                            s.position_qty,
                        )

                if self._minute_log is not None:
                    open_n = sum(1 for s in self.states.values() if s.position_qty > 0.0)
                    paper_bal = self.paper_account.quote_balance if self.paper_account else None
                    self._minute_log.maybe_write(
                        now,
                        symbols_watched=len(self.states),
                        open_positions=open_n,
                        paper_balance_usdt=paper_bal,
                        trades_closed_session=self._trades_closed_session,
                        rejection_counters=self.rejection_counters.nonzero_dict() or None,
                    )

                # Check if positions need persisting (actual I/O done outside lock).
                should_persist = self._positions_dirty and self._position_store is not None

            # Persist dirty positions outside the lock (file I/O).
            if should_persist and self._position_store is not None:
                if self._position_store.save(self.states):
                    async with self._lock:
                        self._positions_dirty = False
                # If save failed, dirty stays True — retry next tick.

            # Periodic reconciliation (outside lock — does network I/O).
            reconcile_interval = float(self.cfg.runtime.reconciliation_interval_seconds or 0.0)
            if reconcile_interval > 0 and not self.cfg.runtime.paper_mode:
                if (now - self._last_reconcile_monotonic) >= reconcile_interval:
                    self._last_reconcile_monotonic = now
                    try:
                        await self._periodic_reconcile()
                    except Exception as exc:
                        logging.error("Periodic reconciliation error: %s", exc, exc_info=True)

            # Phase 2: execute closes outside the lock (all network I/O happens here).
            for snap in to_close:
                try:
                    await self._close_position(snap)
                except Exception as exc:
                    logging.error(
                        "Unexpected error closing position for %s: %s",
                        snap.state.symbol, exc, exc_info=True,
                    )


    def _on_ws_message(self, msg: dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            return
        try:
            self._msg_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._dropped_messages += 1
            channel = msg.get("channel") or msg.get("c") or "unknown"
            logging.warning(
                "Message queue full — dropping message (channel=%s, total_dropped=%d)",
                channel, self._dropped_messages,
            )

    async def _message_consumer(self) -> None:
        """Drain messages from the queue sequentially — provides backpressure."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._msg_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await self._handle_message(msg)

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        try:
            state: SymbolState | None = None
            sig_side: str = ""
            sig_move: float = 0.0

            # -- Phase 1: update state, detect signal, gate checks (under lock, no I/O) --
            async with self._lock:
                symbol = self.parser.extract_symbol(msg)
                if not symbol or symbol not in self.states:
                    return
                state = self.states[symbol]
                self.parser.ingest(state, msg)

                result = self._check_fair_rise_signal(state)
                if result is None:
                    return
                sig_side, sig_move = result
                if self.cfg.strategy.reverse_signal:
                    sig_side = "sell" if sig_side == "buy" else "buy"

                reject_reason = self._check_liquidity(state, sig_side, sig_move)
                if reject_reason:
                    logging.info(
                        "SKIP %s side=%s fair_chg=%.4f%%: %s",
                        state.symbol, sig_side.upper(), sig_move * 100.0, reject_reason,
                    )
                    return

                # Atomically check open count including pending opens.
                open_count = sum(1 for s in self.states.values() if s.position_qty != 0.0)
                if open_count + self._pending_opens >= self.cfg.strategy.max_open_positions:
                    self.rejection_counters.max_positions += 1
                    return
                if not self.risk.can_open(state):
                    self.rejection_counters.risk_blocked += 1
                    return

                # Check circuit breaker.
                if state.consecutive_order_errors >= _MAX_CONSECUTIVE_ERRORS:
                    if time.monotonic() < state.cooldown_until_monotonic:
                        self.rejection_counters.circuit_breaker += 1
                        return
                    # Cooldown expired — reset error counter.
                    state.consecutive_order_errors = 0

                self.rejection_counters.entries_attempted += 1
                self._pending_opens += 1
                state.has_pending_open = True

            # -- Phase 2: place order outside the lock (all network I/O happens here) --
            filled: float = 0.0
            order_exc: Exception | None = None
            try:
                filled = await self.exec.place_entry_limit(state, sig_side)
            except Exception as exc:  # noqa: BLE001
                order_exc = exc

            # -- Phase 3: write back results under lock --
            halting = False
            async with self._lock:
                self._pending_opens -= 1
                state.has_pending_open = False
                if order_exc is not None:
                    state.entry_price = None
                    state.entry_quote_locked = 0.0
                    state.consecutive_order_errors += 1
                    if state.consecutive_order_errors >= _MAX_CONSECUTIVE_ERRORS:
                        state.cooldown_until_monotonic = (
                            time.monotonic() + _ERROR_COOLDOWN_SECONDS
                        )
                        halting = True
                elif filled > 0:
                    state.consecutive_order_errors = 0
                    state.entry_wall_epoch = time.time()
                    state.entry_signal_pct = sig_move * 100.0
                    self.risk.mark_opened(state, sig_side, filled, self.cfg.strategy.hold_seconds)
                    self._positions_dirty = True
                    self.rejection_counters.entries_filled += 1

            # Logging outside the lock (avoid holding lock during string formatting).
            if order_exc is not None:
                logging.warning(
                    "Entry failed for %s side=%s: %s", state.symbol, sig_side, order_exc,
                )
                if halting:
                    logging.error(
                        "HALTING signals for %s: %d consecutive order errors, "
                        "cooldown %.0fs",
                        state.symbol,
                        state.consecutive_order_errors,
                        _ERROR_COOLDOWN_SECONDS,
                    )
                return

            if filled > 0:
                if self.paper_account is not None:
                    logging.info(
                        "Paper balance after OPEN: %.2f USDT",
                        self.paper_account.quote_balance,
                    )
                logging.info(
                    "OPEN %s side=%s qty=%.8f fair_chg=%.4f%% "
                    "last=%.8f fair=%.8f",
                    state.symbol, sig_side.upper(), filled, sig_move * 100.0,
                    state.last_price or 0.0, state.fair_price or 0.0,
                )
        except Exception as exc:
            logging.exception("Unexpected error in _handle_message: %s", exc)

    def _check_entry_follow_timeout(self, state: SymbolState, now: float) -> bool:
        """Phase 3: Return True if price has not followed the signal direction
        within entry_follow_timeout_seconds after entry.

        Only fires when:
        - Feature is enabled (timeout > 0)
        - Position is open and trailing has NOT yet activated
        - Timeout window has elapsed
        - Price has NOT moved entry_follow_min_pct in the signal direction
        """
        timeout = self.cfg.strategy.entry_follow_timeout_seconds
        if timeout <= 0:
            return False
        if state.position_qty == 0.0:
            return False
        if state.trailing_active:
            return False
        if state.entry_ts_monotonic <= 0:
            return False
        if (now - state.entry_ts_monotonic) < timeout:
            return False

        entry = state.entry_price
        last = state.last_price
        side = state.side
        if entry is None or last is None or side is None or entry <= 0:
            return False

        min_pct = self.cfg.strategy.entry_follow_min_pct
        if side == "buy":
            pnl_pct = (last - entry) / entry
        else:
            pnl_pct = (entry - last) / entry

        return pnl_pct < min_pct

    @staticmethod
    def _snapshot_for_close(state: SymbolState) -> _CloseSnapshot:
        """Capture a frozen snapshot of close-relevant fields while holding the lock."""
        if state.side not in VALID_SIDES:
            logging.error(
                "CLOSE SKIPPED %s: invalid side=%r, cannot determine PnL direction",
                state.symbol, state.side,
            )
        return _CloseSnapshot(
            state=state,
            entry_price=state.entry_price,
            side=state.side if state.side in VALID_SIDES else "buy",
            entry_wall=state.entry_wall_epoch or time.time(),
            signal_at_entry=state.entry_signal_pct,
            close_reason=state.close_reason or "unknown",
            leverage=state.max_leverage,
            best_pnl_pct=state.best_pnl_pct,
        )

    async def _close_position(self, snap: _CloseSnapshot) -> None:
        state = snap.state

        try:
            ex = await self.exec.place_exit(state)
        except Exception as exc:
            logging.error(
                "place_exit failed for %s: %s — position may still be open on exchange",
                state.symbol, exc, exc_info=True,
            )
            # Read state under lock, then release before any network I/O.
            # Holding the lock during a network call would block the entire pipeline
            # (message consumer, timer loop, other symbol closes) for the round-trip.
            async with self._lock:
                if abs(state.position_qty) < 1e-9:
                    return
                state.close_fail_count += 1
                should_verify = state.close_fail_count >= 3
                if should_verify:
                    logging.warning(
                        "place_exit failed %d times for %s — verifying position on exchange",
                        state.close_fail_count, state.symbol,
                    )

            # Network I/O outside the lock.
            qty_on_exchange: float | None = None
            if should_verify:
                try:
                    qty_on_exchange = await self.exec.fetch_position_qty(state.symbol)
                except Exception as verify_exc:
                    logging.error(
                        "Failed to verify position for %s: %s — will retry close",
                        state.symbol, verify_exc,
                    )

            # Write result back under lock.
            async with self._lock:
                if qty_on_exchange is not None and abs(qty_on_exchange) < 1e-9:
                    logging.warning(
                        "Position %s is GONE on exchange (qty=%.8f) — "
                        "was likely closed externally (liquidation/stop-loss). "
                        "Clearing local state.",
                        state.symbol, qty_on_exchange,
                    )
                    # Record the trade. Use stop_loss_price as exit estimate:
                    # last_price can be wrong if the market recovered after liquidation,
                    # which would record fake profit and bypass risk tracking.
                    # stop_loss_price is the level where we expected the trade to close,
                    # so it's a conservative and directionally-correct approximation.
                    entry_price_f = float(snap.entry_price) if snap.entry_price is not None else 0.0
                    if state.stop_loss_price is not None and state.stop_loss_price > 0:
                        exit_price_f = float(state.stop_loss_price)
                    elif state.last_price is not None:
                        exit_price_f = float(state.last_price)
                        logging.warning(
                            "Position %s GONE: no stop_loss_price set, "
                            "falling back to last_price=%.8f as exit estimate",
                            state.symbol, exit_price_f,
                        )
                    else:
                        exit_price_f = 0.0
                    qty_f = float(state.position_qty)
                    if entry_price_f > 0 and exit_price_f > 0 and qty_f > 0:
                        pnl = pnl_linear_usdt(
                            side_open=snap.side,
                            qty_base=qty_f,
                            entry_price=entry_price_f,
                            exit_price=exit_price_f,
                            contract_size=state.contract_size,
                        )
                        notional = entry_price_f * qty_f * state.contract_size
                        pnl_pct = (pnl / notional * 100.0) if notional > 0 else 0.0
                        close_reason = "external_close"
                        trade_record = {
                            "event": "trade_closed",
                            "symbol": state.symbol,
                            "side_open": snap.side,
                            "entry_time_utc": datetime.fromtimestamp(snap.entry_wall, tz=UTC).isoformat(),
                            "exit_time_utc": datetime.fromtimestamp(time.time(), tz=UTC).isoformat(),
                            "entry_price": entry_price_f,
                            "exit_price": exit_price_f,
                            "qty_base": qty_f,
                            "notional_entry_quote": notional,
                            "pnl_quote_usdt": round(pnl, 8),
                            "pnl_pct_on_notional": round(pnl_pct, 6),
                            "fair_rise_at_entry_pct": (
                                round(snap.signal_at_entry, 6)
                                if snap.signal_at_entry is not None else None
                            ),
                            "close_reason": close_reason,
                            "leverage": snap.leverage,
                            "hold_seconds": round(time.time() - snap.entry_wall, 1),
                            "best_pnl_pct": round(snap.best_pnl_pct, 6),
                            "paper_mode": self.cfg.runtime.paper_mode,
                        }
                        if self._trades_log is not None:
                            self._trades_log.record_closed_trade(trade_record)
                        if self._journal is not None:
                            self._journal.record_trade(trade_record)
                        self.risk.record_trade_pnl(state, pnl)
                    state.close_fail_count = 0
                    self.risk.mark_closed(state)
                    self._positions_dirty = True
                    return
                if should_verify:
                    state.close_fail_count = 0  # reset counter, schedule retry
                state.close_queued = False
                state.close_due_monotonic = time.monotonic()
            return

        if ex.filled > 0:
            exit_wall = time.time()
            entry_price_f = float(snap.entry_price) if snap.entry_price is not None else 0.0
            exit_price_f = float(ex.exit_price)

            if entry_price_f <= 0:
                logging.error(
                    "CLOSE %s: entry_price unknown (None/0), PnL will not be recorded. "
                    "Position cleared from local state.",
                    state.symbol,
                )

            pnl = pnl_linear_usdt(
                side_open=snap.side,
                qty_base=ex.filled,
                entry_price=entry_price_f,
                exit_price=exit_price_f,
                contract_size=state.contract_size,
            )
            notional = entry_price_f * ex.filled * state.contract_size
            pnl_pct = (pnl / notional * 100.0) if notional > _MIN_PRICE_GUARD else 0.0

            logging.info(
                "CLOSE %s qty=%.8f entry=%.8f exit=%.8f pnl=%.4f",
                state.symbol, ex.filled, entry_price_f, exit_price_f, pnl,
            )
            if self.paper_account is not None:
                async with self._lock:
                    logging.info(
                        "Paper balance after CLOSE: %.2f USDT",
                        self.paper_account.quote_balance,
                    )

            hold_secs = exit_wall - snap.entry_wall if snap.entry_wall > 0 else 0.0
            trade_record = {
                "event": "trade_closed",
                "symbol": state.symbol,
                "side_open": snap.side,
                "entry_time_utc": datetime.fromtimestamp(snap.entry_wall, tz=UTC).isoformat(),
                "exit_time_utc": datetime.fromtimestamp(exit_wall, tz=UTC).isoformat(),
                "entry_price": entry_price_f,
                "exit_price": exit_price_f,
                "qty_base": ex.filled,
                "notional_entry_quote": notional,
                "pnl_quote_usdt": round(pnl, 8),
                "pnl_pct_on_notional": round(pnl_pct, 6),
                "fair_rise_at_entry_pct": (
                    round(snap.signal_at_entry, 6) if snap.signal_at_entry is not None else None
                ),
                "close_reason": snap.close_reason,
                "leverage": snap.leverage,
                "hold_seconds": round(hold_secs, 1),
                "best_pnl_pct": round(snap.best_pnl_pct, 6),
                "paper_mode": self.cfg.runtime.paper_mode,
            }

            if self._trades_log is not None and entry_price_f > 0 and exit_price_f > 0:
                self._trades_log.record_closed_trade(trade_record)
            if self._journal is not None and entry_price_f > 0 and exit_price_f > 0:
                self._journal.record_trade(trade_record)

            # Mark closed, record PnL, and increment counter under the lock.
            async with self._lock:
                if entry_price_f > 0 and exit_price_f > 0:
                    self.risk.record_trade_pnl(state, pnl)
                if snap.close_reason == "take_profit":
                    self.risk.apply_tp_cooldown(state)
                self.risk.mark_closed(state)
                self._trades_closed_session += 1
                self._positions_dirty = True
        else:
            # Exit did not fill — log alert and restore sentinel so next tick retries.
            logging.error(
                "EXIT FAILED for %s: filled=0 — position may still be open on exchange. "
                "Will retry on next timer tick.",
                state.symbol,
            )
            async with self._lock:
                if state.position_qty != 0.0:
                    state.close_queued = False
                    state.close_due_monotonic = time.monotonic()

    def _check_liquidity(self, state: SymbolState, side: str, dev: float) -> str | None:
        rc = self.rejection_counters
        cfg = self.cfg.strategy
        book = state.book

        spread = book.spread_pct
        if spread is None:
            rc.no_bid_ask += 1
            return "no bid/ask"
        if spread > cfg.max_spread_pct:
            rc.spread_too_wide += 1
            return f"spread {spread:.2f}% > max {cfg.max_spread_pct:.2f}%"
        abs_dev = abs(dev) * 100.0
        if abs_dev > 0 and spread / abs_dev > cfg.spread_to_signal_ratio:
            rc.spread_signal_ratio += 1
            return f"spread/signal={spread / abs_dev:.2f} > {cfg.spread_to_signal_ratio:.2f}"

        entry_side = "ask" if side == "buy" else "bid"
        depth_usdt = book.depth_quote(entry_side)
        margin = cfg.quote_size_usdt
        leverage = max(float(state.max_leverage), 1.0)
        notional = margin * leverage
        required_depth = (
            cfg.min_depth_quote_usdt if cfg.min_depth_quote_usdt > 0
            else notional * cfg.depth_to_notional_ratio
        )
        if depth_usdt < required_depth:
            rc.depth_insufficient += 1
            return f"depth {entry_side}={depth_usdt:.1f} USDT < required {required_depth:.1f}"

        return None

    def _check_fair_rise_signal(self, state: SymbolState) -> tuple[str, float] | None:
        rc = self.rejection_counters
        fair = state.fair_price
        last = state.last_price
        if fair is None or last is None or fair <= 0 or last <= 0:
            rc.no_price_data += 1
            return None

        now = time.monotonic()

        # Skip signal if fair price is stale (e.g. MEXC microcap coins update slowly).
        max_stale = self.cfg.data.max_fair_staleness_seconds
        if max_stale > 0:
            if state.fair_price_updated_monotonic <= 0:
                logging.debug("STALE fair %s: never received", state.symbol)
                rc.fair_stale += 1
                return None
            age = now - state.fair_price_updated_monotonic
            if age > max_stale:
                logging.debug("STALE fair %s: age=%.1fs > max=%.1fs", state.symbol, age, max_stale)
                rc.fair_stale += 1
                return None
        window = max(float(self.cfg.strategy.fair_move_window_seconds), 1.0)
        threshold = float(self.cfg.strategy.fair_rise_threshold)
        last_max = float(self.cfg.strategy.last_max_move_pct)
        confirm_secs = float(self.cfg.strategy.signal_confirm_seconds)
        max_pullback = float(self.cfg.strategy.signal_max_pullback)

        if state.monitoring_start_monotonic == 0.0:
            state.monitoring_start_monotonic = now
        if now - state.monitoring_start_monotonic < window:
            rc.window_warmup += 1
            return None

        state.fair_samples.append((now, fair))
        state.last_samples.append((now, last))

        cutoff = now - window
        while state.fair_samples and state.fair_samples[0][0] < cutoff:
            state.fair_samples.popleft()
        while state.last_samples and state.last_samples[0][0] < cutoff:
            state.last_samples.popleft()

        if not state.fair_samples or not state.last_samples:
            rc.window_warmup += 1
            return None

        old_fair = state.fair_samples[0][1]
        old_last = state.last_samples[0][1]
        fair_change = (fair - old_fair) / old_fair
        last_change = (last - old_last) / old_last

        # -- Confirmation phase --
        if state.pending_trigger_time > 0:
            return self._confirm_signal(state, fair_change, last_change, now, confirm_secs,
                                        max_pullback, last_max)

        # -- Detection phase --
        return self._detect_signal(state, fair_change, last_change, now, threshold, last_max,
                                   confirm_secs)

    def _reset_signal_state(self, state: SymbolState) -> None:
        state.pending_trigger_time = 0.0
        state.pending_trigger_move = 0.0
        state.pending_trigger_extreme = 0.0
        state.pending_trigger_side = ""
        state.fair_trigger_latched = False

    def _confirm_signal(
        self,
        state: SymbolState,
        fair_change: float,
        last_change: float,
        now: float,
        confirm_secs: float,
        max_pullback: float,
        last_max: float,
    ) -> tuple[str, float] | None:
        side = state.pending_trigger_side
        if side == "buy" and fair_change > state.pending_trigger_extreme:
            state.pending_trigger_extreme = fair_change
        elif side == "sell" and fair_change < state.pending_trigger_extreme:
            state.pending_trigger_extreme = fair_change

        adverse = abs(state.pending_trigger_extreme - fair_change)
        if adverse > max_pullback:
            logging.debug(
                "Signal cancelled %s side=%s fair_chg=%.4f%% adverse=%.4f%%",
                state.symbol, side, fair_change * 100, adverse * 100,
            )
            self._reset_signal_state(state)
            self.rejection_counters.pullback_cancel += 1
            return None

        if now - state.pending_trigger_time >= confirm_secs:
            if last_max > 0.0 and abs(last_change) > last_max:
                logging.info(
                    "Signal cancelled %s: last moved %.4f%% > max %.4f%%",
                    state.symbol, last_change * 100, last_max * 100,
                )
                self._reset_signal_state(state)
                self.rejection_counters.last_move_cancel += 1
                return None

            # Max fair_rise cap: extreme spikes (>5%) are already exhausted.
            max_rise = self.cfg.strategy.max_fair_rise_pct
            if max_rise > 0 and abs(fair_change) > max_rise:
                logging.info(
                    "Signal cancelled %s side=%s: |fair_chg|=%.4f%% > cap %.2f%%",
                    state.symbol, side, abs(fair_change) * 100, max_rise * 100,
                )
                self._reset_signal_state(state)
                self.rejection_counters.fair_rise_capped += 1
                return None

            # Adverse last_chg filter: HANA 2026-04-14 liquidation happened
            # because mark price jumped +4.14% but last traded price was
            # -0.67% — a reversion already in progress. Entering momentum
            # into that reversion is the wrong side.
            if self.cfg.strategy.require_last_chg_aligned:
                min_pct = self.cfg.strategy.adverse_last_chg_min_pct
                adverse = (
                    (side == "buy" and last_change < -min_pct)
                    or (side == "sell" and last_change > min_pct)
                )
                if adverse:
                    logging.info(
                        "Signal cancelled %s side=%s: adverse last_chg %.4f%% "
                        "opposes fair_chg %.4f%% (>%.3f%% threshold → reversion)",
                        state.symbol, side, last_change * 100, fair_change * 100,
                        min_pct * 100,
                    )
                    self._reset_signal_state(state)
                    self.rejection_counters.adverse_last_chg += 1
                    return None

            confirmed_move = state.pending_trigger_move
            confirmed_side = state.pending_trigger_side
            self._reset_signal_state(state)
            state.fair_trigger_latched = True  # Keep latched after confirmation.
            self.rejection_counters.signals_confirmed += 1
            logging.info(
                "Signal CONFIRMED %s side=%s fair_chg=%.4f%% last_chg=%.4f%%",
                state.symbol, confirmed_side.upper(), confirmed_move * 100, last_change * 100,
            )
            return confirmed_side, confirmed_move

        return None

    def _detect_signal(
        self,
        state: SymbolState,
        fair_change: float,
        last_change: float,
        now: float,
        threshold: float,
        last_max: float,
        confirm_secs: float,
    ) -> tuple[str, float] | None:
        if state.fair_trigger_latched:
            if abs(fair_change) < threshold * _SIGNAL_RESET_HYSTERESIS:
                state.fair_trigger_latched = False
            else:
                self.rejection_counters.signal_latched += 1
            return None

        if fair_change >= threshold and (last_max == 0.0 or abs(last_change) <= last_max):
            state.fair_trigger_latched = True
            state.pending_trigger_time = now
            state.pending_trigger_move = fair_change
            state.pending_trigger_extreme = fair_change
            state.pending_trigger_side = "buy"
            logging.info(
                "Signal pending %s side=BUY: fair_chg=+%.4f%% last_chg=%.4f%% "
                "-- confirming %.1fs",
                state.symbol, fair_change * 100, last_change * 100, confirm_secs,
            )
        elif fair_change <= -threshold and (last_max == 0.0 or abs(last_change) <= last_max):
            state.fair_trigger_latched = True
            state.pending_trigger_time = now
            state.pending_trigger_move = fair_change
            state.pending_trigger_extreme = fair_change
            state.pending_trigger_side = "sell"
            logging.info(
                "Signal pending %s side=SELL: fair_chg=%.4f%% last_chg=%.4f%% "
                "-- confirming %.1fs",
                state.symbol, fair_change * 100, last_change * 100, confirm_secs,
            )

        return None

    # -- Reconciliation --

    async def _reconcile_startup(self) -> None:
        """Reconcile local state with exchange positions on startup."""
        if self.cfg.runtime.paper_mode:
            logging.info("Paper mode — skipping startup reconciliation")
            return

        # 1. Load persisted state.
        persisted: list[dict[str, Any]] = []
        if self._position_store is not None:
            persisted = self._position_store.load()
        persisted_by_symbol: dict[str, dict[str, Any]] = {
            p["symbol"]: p for p in persisted if "symbol" in p
        }

        # 2. Fetch live positions from exchange.
        try:
            exchange_positions = await self.exec.fetch_open_positions()
        except Exception as exc:
            logging.critical(
                "STARTUP RECONCILIATION FAILED: cannot fetch positions: %s", exc,
            )
            raise  # Cannot trade without knowing about open positions.

        exchange_by_symbol = {p["symbol"]: p for p in exchange_positions}

        restored = 0
        orphans = 0

        # 3. For each exchange position — restore state.
        for symbol, ex_pos in exchange_by_symbol.items():
            if symbol not in self.states:
                logging.critical(
                    "RECONCILE: exchange has position in %s but symbol not in watchlist — "
                    "MANUAL CLOSE REQUIRED!",
                    symbol,
                )
                orphans += 1
                continue

            state = self.states[symbol]
            local = persisted_by_symbol.get(symbol)

            if local:
                # Restore from persisted state (with corruption protection).
                try:
                    state.side = local.get("side")
                    state.entry_price = local.get("entry_price")
                    state.entry_quote_locked = float(local.get("entry_quote_locked", 0.0))
                    state.entry_wall_epoch = float(local.get("entry_wall_epoch", 0.0))
                    state.entry_signal_pct = local.get("entry_signal_pct")
                    state.max_leverage = float(local.get("max_leverage", state.max_leverage))
                    state.stop_loss_price = local.get("stop_loss_price")
                    state.trailing_active = bool(local.get("trailing_active", False))
                    state.best_pnl_pct = float(local.get("best_pnl_pct", 0.0))
                    state.trailing_stop_price = local.get("trailing_stop_price")
                    state.consecutive_losses = int(local.get("consecutive_losses", 0))
                    # Recalculate monotonic timers from wall-clock.
                    if local.get("_immediate_close"):
                        remaining = 0.0  # Was already queued for close before crash.
                        state.close_queued = False  # Will be re-queued by timer loop.
                    else:
                        close_due_wall = float(local.get("_close_due_wall_epoch", 0.0))
                        remaining = max(0.0, close_due_wall - time.time()) if close_due_wall > 0 else self.cfg.strategy.hold_seconds
                    state.entry_ts_monotonic = time.monotonic()
                    state.close_due_monotonic = time.monotonic() + remaining
                    logging.info(
                        "RECONCILE RESTORED %s: side=%s qty=%.8f entry=%.8f remaining=%.0fs",
                        symbol, state.side, ex_pos["qty"],
                        state.entry_price or 0.0, remaining,
                    )
                except (TypeError, ValueError) as e:
                    logging.error(
                        "RECONCILE: corrupt persisted state for %s: %s — treating as orphan",
                        symbol, e,
                    )
                    local = None  # Fall through to orphan path below.

            if not local:
                # Orphan: position on exchange but no persisted state.
                state.side = ex_pos["side"]
                state.entry_price = ex_pos["entry_price"]
                state.entry_wall_epoch = time.time()
                state.entry_ts_monotonic = time.monotonic()
                state.close_due_monotonic = time.monotonic() + self.cfg.strategy.hold_seconds
                # Compute stop-loss for orphan.
                if state.entry_price and state.entry_price > 0 and state.side:
                    leverage = max(state.max_leverage, 1.0)
                    loss_frac = (
                        self.cfg.risk.stop_loss_high_lev_margin_fraction
                        if leverage >= self.cfg.risk.stop_loss_high_lev_threshold
                        else self.cfg.risk.stop_loss_margin_fraction
                    )
                    state.stop_loss_price = compute_stop_loss_price(
                        state.entry_price, leverage, state.side, loss_frac,
                        maintenance_rate=state.maintenance_rate,
                    )
                logging.warning(
                    "RECONCILE ORPHAN %s: found on exchange but no persisted state. "
                    "side=%s qty=%.8f entry=%.8f — scheduling close in %.0fs",
                    symbol, ex_pos["side"], ex_pos["qty"], ex_pos["entry_price"],
                    self.cfg.strategy.hold_seconds,
                )
                orphans += 1

            # Always trust exchange qty.
            state.position_qty = ex_pos["qty"]
            restored += 1

        # 4. Positions in persisted but NOT on exchange — closed externally.
        for symbol in persisted_by_symbol:
            if symbol not in exchange_by_symbol:
                logging.info(
                    "RECONCILE: persisted position %s not found on exchange — "
                    "assuming closed externally (exchange stop-loss?)",
                    symbol,
                )

        # 5. Clear persisted file (will be re-saved by _timer_loop).
        if self._position_store is not None:
            self._position_store.clear()

        if exchange_positions:
            logging.critical(
                "STARTUP RECONCILIATION: found %d open positions on exchange "
                "(%d restored, %d orphans)",
                len(exchange_positions), restored, orphans,
            )
        else:
            logging.info("STARTUP RECONCILIATION: no open positions found")

    async def _periodic_reconcile(self) -> None:
        """Compare local position state with exchange every N seconds."""
        if self.cfg.runtime.paper_mode:
            return

        try:
            exchange_positions = await self.exec.fetch_open_positions()
        except Exception as exc:
            logging.error("Periodic reconciliation failed: %s", exc)
            return

        exchange_by_symbol = {p["symbol"]: p for p in exchange_positions}

        async with self._lock:
            for symbol, state in self.states.items():
                ex_pos = exchange_by_symbol.get(symbol)
                local_qty = state.position_qty

                if local_qty != 0.0 and ex_pos is None:
                    # Skip if already queued for close (close_due=inf sentinel).
                    if state.close_queued:
                        continue
                    # Local open, exchange closed — stop-loss on exchange fired
                    # or the position was liquidated. Use stop_loss_price as
                    # conservative exit estimate (last_price can be stale or
                    # post-liquidation recovered).
                    logging.critical(
                        "RECONCILE DRIFT %s: local qty=%.8f but exchange shows NO position. "
                        "Resetting local state.",
                        symbol, local_qty,
                    )
                    if state.entry_price and state.side:
                        if state.stop_loss_price and state.stop_loss_price > 0:
                            exit_estimate = float(state.stop_loss_price)
                        elif state.last_price:
                            exit_estimate = float(state.last_price)
                        else:
                            exit_estimate = 0.0
                        entry_f = float(state.entry_price)
                        if exit_estimate > 0 and entry_f > 0:
                            approx_pnl = pnl_linear_usdt(
                                side_open=state.side,
                                qty_base=local_qty,
                                entry_price=entry_f,
                                exit_price=exit_estimate,
                                contract_size=state.contract_size,
                            )
                            notional = entry_f * local_qty * state.contract_size
                            pnl_pct = (approx_pnl / notional * 100.0) if notional > 0 else 0.0
                            trade_record = {
                                "event": "trade_closed",
                                "symbol": symbol,
                                "side_open": state.side,
                                "entry_time_utc": datetime.fromtimestamp(
                                    state.entry_wall_epoch or time.time(), tz=UTC,
                                ).isoformat(),
                                "exit_time_utc": datetime.fromtimestamp(
                                    time.time(), tz=UTC,
                                ).isoformat(),
                                "entry_price": entry_f,
                                "exit_price": exit_estimate,
                                "qty_base": float(local_qty),
                                "notional_entry_quote": notional,
                                "pnl_quote_usdt": round(approx_pnl, 8),
                                "pnl_pct_on_notional": round(pnl_pct, 6),
                                "fair_rise_at_entry_pct": (
                                    round(state.entry_signal_pct, 6)
                                    if state.entry_signal_pct is not None else None
                                ),
                                "close_reason": "reconcile_drift",
                                "leverage": float(state.max_leverage),
                                "hold_seconds": round(
                                    time.time() - (state.entry_wall_epoch or time.time()), 1,
                                ),
                                "best_pnl_pct": round(state.best_pnl_pct, 6),
                                "paper_mode": self.cfg.runtime.paper_mode,
                            }
                            if self._trades_log is not None:
                                self._trades_log.record_closed_trade(trade_record)
                            if self._journal is not None:
                                self._journal.record_trade(trade_record)
                            self.risk.record_trade_pnl(state, approx_pnl)
                    self.risk.mark_closed(state)
                    self._positions_dirty = True

                elif local_qty == 0.0 and ex_pos is not None:
                    # Local closed, exchange open — orphan position.
                    logging.critical(
                        "RECONCILE DRIFT %s: local shows no position but exchange has "
                        "side=%s qty=%.8f. Scheduling immediate close!",
                        symbol, ex_pos["side"], ex_pos["qty"],
                    )
                    state.position_qty = ex_pos["qty"]
                    state.side = ex_pos["side"]
                    state.entry_price = ex_pos["entry_price"]
                    state.entry_wall_epoch = time.time()
                    state.entry_ts_monotonic = time.monotonic()
                    state.close_due_monotonic = time.monotonic()  # Close immediately.
                    state.close_reason = "reconcile_orphan"
                    # Compute stop-loss for orphan.
                    if state.entry_price and state.entry_price > 0 and state.side:
                        leverage = max(state.max_leverage, 1.0)
                        loss_frac = (
                            self.cfg.risk.stop_loss_high_lev_margin_fraction
                            if leverage >= self.cfg.risk.stop_loss_high_lev_threshold
                            else self.cfg.risk.stop_loss_margin_fraction
                        )
                        state.stop_loss_price = compute_stop_loss_price(
                            state.entry_price, leverage, state.side, loss_frac,
                            maintenance_rate=state.maintenance_rate,
                        )
                    self._positions_dirty = True

                elif local_qty != 0.0 and ex_pos is not None:
                    # Both see position — check qty drift.
                    drift = abs(local_qty - ex_pos["qty"])
                    if drift > local_qty * 0.01:  # >1% discrepancy.
                        logging.warning(
                            "RECONCILE QTY DRIFT %s: local=%.8f exchange=%.8f (delta=%.8f). "
                            "Updating to exchange value.",
                            symbol, local_qty, ex_pos["qty"], drift,
                        )
                        state.position_qty = ex_pos["qty"]
                        self._positions_dirty = True

            # Warn about positions in symbols not in watchlist.
            for symbol in exchange_by_symbol:
                if symbol not in self.states:
                    logging.critical(
                        "RECONCILE: exchange has position in %s but symbol not in watchlist! "
                        "Manual close required.",
                        symbol,
                    )
