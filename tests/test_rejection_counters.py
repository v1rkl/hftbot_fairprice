"""Tests for SignalRejectionCounters."""
from core.rejection_counters import SignalRejectionCounters


class TestSignalRejectionCounters:
    def test_defaults_are_zero(self) -> None:
        c = SignalRejectionCounters()
        assert c.spread_too_wide == 0
        assert c.signals_confirmed == 0
        assert c.total_rejected() == 0

    def test_increment(self) -> None:
        c = SignalRejectionCounters()
        c.spread_too_wide += 1
        c.spread_too_wide += 1
        c.depth_insufficient += 3
        assert c.spread_too_wide == 2
        assert c.depth_insufficient == 3

    def test_to_dict(self) -> None:
        c = SignalRejectionCounters()
        c.no_price_data = 5
        d = c.to_dict()
        assert d["no_price_data"] == 5
        assert d["spread_too_wide"] == 0
        assert isinstance(d, dict)

    def test_nonzero_dict(self) -> None:
        c = SignalRejectionCounters()
        c.fair_stale = 10
        c.pullback_cancel = 2
        nz = c.nonzero_dict()
        assert nz == {"fair_stale": 10, "pullback_cancel": 2}

    def test_nonzero_dict_empty(self) -> None:
        c = SignalRejectionCounters()
        assert c.nonzero_dict() == {}

    def test_total_rejected_excludes_positive_counters(self) -> None:
        c = SignalRejectionCounters()
        c.spread_too_wide = 5
        c.max_positions = 3
        c.signals_confirmed = 10
        c.entries_attempted = 8
        c.entries_filled = 7
        assert c.total_rejected() == 8  # 5 + 3, not including confirmed/attempted/filled

    def test_summary_line_empty(self) -> None:
        c = SignalRejectionCounters()
        assert c.summary_line() == "rejections: (none)"

    def test_summary_line_with_data(self) -> None:
        c = SignalRejectionCounters()
        c.spread_too_wide = 42
        c.entries_filled = 3
        line = c.summary_line()
        assert "spread_too_wide=42" in line
        # Throughput counters excluded from rejection summary.
        assert "entries_filled" not in line
        assert line.startswith("rejections: ")

    def test_summary_line_only_throughput(self) -> None:
        """When only throughput counters are nonzero, summary shows (none)."""
        c = SignalRejectionCounters()
        c.signals_confirmed = 5
        c.entries_attempted = 4
        c.entries_filled = 3
        assert c.summary_line() == "rejections: (none)"
        assert c.total_rejected() == 0
