"""
TDD tests for PnL contract-size bug.

Gate.io (and some other exchanges) quote position sizes in CONTRACTS, not tokens.
For symbols like STO/DRIFT/D/BASED/PIPPIN (contractSize=10) or RLS (contractSize=100),
`ex.filled` / `state.position_qty` is stored in contracts.

pnl_linear_usdt must multiply by contract_size to get the real USDT PnL.

Example:
  STO long, 500 contracts @ 0.1 entry, 0.11 exit, contractSize=10
  Tokens  = 500 * 10 = 5000
  PnL     = 5000 * (0.11 - 0.10) = 50 USDT
  Old buggy formula: 500 * (0.11 - 0.10) = 5 USDT  ← 10x too small
"""
from __future__ import annotations

import pytest

from core.trade_log import pnl_linear_usdt


# ---------------------------------------------------------------------------
# pnl_linear_usdt: contract_size parameter must exist and default to 1.0
# ---------------------------------------------------------------------------

class TestPnlLinearUsdtContractSizeParam:
    """contract_size parameter is present and backward-compatible."""

    def test_default_contract_size_buy_unchanged(self):
        """With default contract_size=1.0, result equals old formula."""
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=100.0,
            entry_price=1.0,
            exit_price=1.05,
        )
        assert pnl == pytest.approx(5.0)

    def test_default_contract_size_sell_unchanged(self):
        pnl = pnl_linear_usdt(
            side_open="sell",
            qty_base=100.0,
            entry_price=1.05,
            exit_price=1.0,
        )
        assert pnl == pytest.approx(5.0)

    def test_explicit_contract_size_one_equals_default(self):
        pnl_default = pnl_linear_usdt(
            side_open="buy",
            qty_base=50.0,
            entry_price=2.0,
            exit_price=2.10,
        )
        pnl_explicit = pnl_linear_usdt(
            side_open="buy",
            qty_base=50.0,
            entry_price=2.0,
            exit_price=2.10,
            contract_size=1.0,
        )
        assert pnl_default == pytest.approx(pnl_explicit)


# ---------------------------------------------------------------------------
# contractSize=10 (STO / DRIFT / D / BASED / PIPPIN)
# ---------------------------------------------------------------------------

class TestPnlContractSizeTen:
    """Symbols with contractSize=10 must return 10x compared to the old formula."""

    def test_sto_long_profit(self):
        """500 contracts @ 0.10 entry → 0.11 exit, contractSize=10 → 50 USDT."""
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=500.0,
            entry_price=0.10,
            exit_price=0.11,
            contract_size=10.0,
        )
        assert pnl == pytest.approx(50.0)

    def test_sto_long_loss(self):
        """500 contracts @ 0.10 entry → 0.09 exit → -50 USDT."""
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=500.0,
            entry_price=0.10,
            exit_price=0.09,
            contract_size=10.0,
        )
        assert pnl == pytest.approx(-50.0)

    def test_sto_short_profit(self):
        """500 contracts SHORT @ 0.10 entry → 0.09 exit → +50 USDT."""
        pnl = pnl_linear_usdt(
            side_open="sell",
            qty_base=500.0,
            entry_price=0.10,
            exit_price=0.09,
            contract_size=10.0,
        )
        assert pnl == pytest.approx(50.0)

    def test_sto_short_loss(self):
        pnl = pnl_linear_usdt(
            side_open="sell",
            qty_base=500.0,
            entry_price=0.09,
            exit_price=0.10,
            contract_size=10.0,
        )
        assert pnl == pytest.approx(-50.0)

    def test_result_is_ten_times_contract_size_one(self):
        """Ratio between contractSize=10 and contractSize=1 must be exactly 10."""
        kwargs = dict(side_open="buy", qty_base=200.0, entry_price=5.0, exit_price=5.5)
        pnl_10 = pnl_linear_usdt(**kwargs, contract_size=10.0)
        pnl_1 = pnl_linear_usdt(**kwargs, contract_size=1.0)
        assert pnl_10 == pytest.approx(pnl_1 * 10.0)


# ---------------------------------------------------------------------------
# contractSize=100 (RLS)
# ---------------------------------------------------------------------------

class TestPnlContractSizeHundred:
    """Symbols with contractSize=100 must return 100x compared to the old formula."""

    def test_rls_long_profit(self):
        """10 contracts @ 0.001 entry → 0.0011 exit, contractSize=100 → 0.10 USDT."""
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=10.0,
            entry_price=0.001,
            exit_price=0.0011,
            contract_size=100.0,
        )
        assert pnl == pytest.approx(0.10, rel=1e-5)

    def test_rls_result_is_hundred_times_contract_size_one(self):
        kwargs = dict(side_open="sell", qty_base=5.0, entry_price=0.002, exit_price=0.0015)
        pnl_100 = pnl_linear_usdt(**kwargs, contract_size=100.0)
        pnl_1 = pnl_linear_usdt(**kwargs, contract_size=1.0)
        assert pnl_100 == pytest.approx(pnl_1 * 100.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPnlEdgeCases:
    def test_unknown_side_returns_zero(self):
        pnl = pnl_linear_usdt(
            side_open="unknown",
            qty_base=100.0,
            entry_price=1.0,
            exit_price=1.1,
            contract_size=10.0,
        )
        assert pnl == 0.0

    def test_zero_qty_returns_zero(self):
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=0.0,
            entry_price=1.0,
            exit_price=1.1,
            contract_size=10.0,
        )
        assert pnl == 0.0

    def test_equal_entry_exit_zero_pnl(self):
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=100.0,
            entry_price=1.0,
            exit_price=1.0,
            contract_size=10.0,
        )
        assert pnl == 0.0

    def test_fractional_contract_size(self):
        """contractSize=0.5 (hypothetical) — should still multiply correctly."""
        pnl = pnl_linear_usdt(
            side_open="buy",
            qty_base=100.0,
            entry_price=1.0,
            exit_price=1.1,
            contract_size=0.5,
        )
        # tokens = 100 * 0.5 = 50; pnl = 50 * 0.1 = 5.0
        assert pnl == pytest.approx(5.0)
