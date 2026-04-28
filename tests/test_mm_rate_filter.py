"""
TDD tests for max_maintenance_rate symbol filter.

Gate.io maintenance_rate varies dramatically per symbol:
  D/USDT   = 0.04  (4%)  → at 20x, liquidation at 1% move  ← too risky
  STO/DRIFT = 0.025 (2.5%) → at 20x, liquidation at 2.5% move
  BTC/USDT = 0.003 (0.3%) → at 20x, liquidation at 4.7% move

Config: exchange.max_maintenance_rate = 0.0 (disabled, default)
        exchange.max_maintenance_rate = 0.03 → skip D/USDT (4%), allow STO (2.5%)
        exchange.max_maintenance_rate = 0.02 → skip D/USDT and STO/DRIFT/RLS

Behavior:
  - 0.0 (default) → no filter, backwards-compatible
  - > 0 → symbols with maintenance_rate > threshold are skipped at startup
  - Skipped symbols are logged at WARNING level with the reason
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.config import BotConfig, ExchangeConfig


# ---------------------------------------------------------------------------
# ExchangeConfig: field must exist
# ---------------------------------------------------------------------------

class TestExchangeConfigField:
    def test_default_is_zero(self):
        cfg = ExchangeConfig()
        assert cfg.max_maintenance_rate == 0.0

    def test_can_be_set(self):
        cfg = ExchangeConfig(max_maintenance_rate=0.03)
        assert cfg.max_maintenance_rate == 0.03

    def test_loads_from_json(self, tmp_path):
        """Config file with exchange.max_maintenance_rate is parsed."""
        f = tmp_path / "cfg.json"
        f.write_text("""{
            "strategy": {"hold_seconds": 25, "quote_size_usdt": 10,
                         "fair_rise_threshold": 0.035},
            "data": {},
            "exchange": {"exchange_id": "gateio", "min_max_leverage": 20,
                         "max_maintenance_rate": 0.03},
            "runtime": {"paper_mode": true, "paper_balance_usdt": 100},
            "risk": {}
        }""")
        cfg = BotConfig.load(f)
        assert cfg.exchange.max_maintenance_rate == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# _select_symbols: maintenance_rate filter
# ---------------------------------------------------------------------------

def _make_engine(max_mm: float = 0.0, min_lev: float = 0.0):
    """Build a minimal TradingEngine with the given config."""
    from core.config import (
        BotConfig, DataConfig, ExchangeConfig, RiskConfig, RuntimeConfig, StrategyConfig,
    )
    from core.engine import TradingEngine

    cfg = BotConfig(
        strategy=StrategyConfig(hold_seconds=25, quote_size_usdt=10,
                                fair_rise_threshold=0.035),
        data=DataConfig(),
        exchange=ExchangeConfig(exchange_id="gateio", min_max_leverage=min_lev,
                                max_maintenance_rate=max_mm),
        runtime=RuntimeConfig(paper_mode=True, paper_balance_usdt=100),
        risk=RiskConfig(),
    )
    engine = TradingEngine.__new__(TradingEngine)
    engine.cfg = cfg
    return engine


def _market(mm_rate: float, max_lev: float = 50.0, mtype: str = "swap") -> dict:
    return {
        "type": mtype,
        "limits": {"leverage": {"max": max_lev}},
        "info": {"maintenance_rate": str(mm_rate)},
    }


MARKETS = {
    "D/USDT:USDT":     _market(0.04),   # too high (4%)
    "STO/USDT:USDT":   _market(0.025),  # borderline (2.5%)
    "BTC/USDT:USDT":   _market(0.003),  # fine (0.3%)
    "FORM/USDT:USDT":  _market(0.01),   # fine (1%)
}


class TestSelectSymbolsMMFilter:
    """_select_symbols respects max_maintenance_rate."""

    def test_disabled_zero_keeps_all(self):
        engine = _make_engine(max_mm=0.0)
        selected = engine._select_symbols(MARKETS)
        assert set(selected) == set(MARKETS.keys())

    def test_threshold_003_excludes_d_usdt(self):
        engine = _make_engine(max_mm=0.03)
        selected = engine._select_symbols(MARKETS)
        assert "D/USDT:USDT" not in selected
        assert "STO/USDT:USDT" in selected
        assert "BTC/USDT:USDT" in selected
        assert "FORM/USDT:USDT" in selected

    def test_threshold_002_excludes_d_and_sto(self):
        engine = _make_engine(max_mm=0.02)
        selected = engine._select_symbols(MARKETS)
        assert "D/USDT:USDT" not in selected
        assert "STO/USDT:USDT" not in selected
        assert "BTC/USDT:USDT" in selected
        assert "FORM/USDT:USDT" in selected

    def test_threshold_exact_boundary_is_inclusive(self):
        """Symbol with mm_rate == threshold is kept (<=, not <)."""
        engine = _make_engine(max_mm=0.025)
        selected = engine._select_symbols(MARKETS)
        assert "STO/USDT:USDT" in selected  # 0.025 == 0.025 → keep

    def test_threshold_just_below_boundary_excludes(self):
        engine = _make_engine(max_mm=0.0249)
        selected = engine._select_symbols(MARKETS)
        assert "STO/USDT:USDT" not in selected

    def test_missing_maintenance_rate_treated_as_zero(self):
        """Symbols with no maintenance_rate in info are NOT excluded."""
        markets = {"X/USDT:USDT": {"type": "swap",
                                    "limits": {"leverage": {"max": 50}},
                                    "info": {}}}
        engine = _make_engine(max_mm=0.03)
        selected = engine._select_symbols(markets)
        assert "X/USDT:USDT" in selected

    def test_combined_with_leverage_filter(self):
        """Both leverage and mm_rate filters apply simultaneously."""
        markets = {
            "A/USDT:USDT": _market(0.01, max_lev=50),   # pass both
            "B/USDT:USDT": _market(0.05, max_lev=50),   # fail mm
            "C/USDT:USDT": _market(0.01, max_lev=10),   # fail lev
            "D/USDT:USDT": _market(0.05, max_lev=10),   # fail both
        }
        engine = _make_engine(max_mm=0.03, min_lev=20.0)
        selected = engine._select_symbols(markets)
        assert selected == ["A/USDT:USDT"]


class TestSelectSymbolsLogging:
    """Skipped symbols are logged at WARNING."""

    def test_skipped_symbol_is_logged(self, caplog):
        import logging
        engine = _make_engine(max_mm=0.03)
        with caplog.at_level(logging.WARNING):
            engine._select_symbols(MARKETS)
        assert any("D/USDT:USDT" in r.message and "maintenance_rate" in r.message
                   for r in caplog.records)

    def test_accepted_symbol_not_warned(self, caplog):
        import logging
        engine = _make_engine(max_mm=0.03)
        with caplog.at_level(logging.WARNING):
            engine._select_symbols(MARKETS)
        assert not any("BTC/USDT:USDT" in r.message and "maintenance_rate" in r.message
                       for r in caplog.records)
