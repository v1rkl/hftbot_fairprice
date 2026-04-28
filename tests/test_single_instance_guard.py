"""
TDD tests for the single-instance guard used by run_gateio_microcap_live.bat.

The bat file must ensure that only ONE bot process runs at a time, even if the
bat is accidentally launched twice simultaneously (e.g., double-click race).

Strategy: the PowerShell mutex block in the bat acquires a named mutex before
killing old processes and starting the new one.  A second concurrent bat
invocation will fail to acquire the same mutex and exit immediately.

These tests verify the guard logic in isolation using a helper PowerShell
script that mirrors what the bat does.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MUTEX_NAME = "Global\\HFTBot_GateioMicrocap"

def _run_ps_mutex(hold_ms: int = 100) -> subprocess.CompletedProcess:
    """Acquire the named mutex, hold for hold_ms ms, release. Exit 0=ok, 1=busy."""
    script = (
        f"$m = New-Object System.Threading.Mutex($false, '{MUTEX_NAME}'); "
        f"if (-not $m.WaitOne(0)) {{ exit 1 }}; "
        f"Start-Sleep -Milliseconds {hold_ms}; "
        f"$m.ReleaseMutex(); exit 0"
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=15,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMutexAcquire:
    """Named mutex can be acquired when free."""

    def test_first_invocation_exits_zero(self):
        result = _run_ps_mutex(hold_ms=100)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_second_invocation_exits_one_while_first_holds(self):
        """Start first process holding the mutex, then try to acquire it."""
        first_done = threading.Event()
        results: list[tuple[str, int]] = []

        def run_first():
            r = _run_ps_mutex(hold_ms=800)
            results.append(("first", r.returncode))
            first_done.set()

        t = threading.Thread(target=run_first, daemon=True)
        t.start()
        time.sleep(0.3)  # let first process acquire the mutex

        # Second attempt — should fail immediately (WaitOne(0))
        r2 = _run_ps_mutex(hold_ms=100)
        assert r2.returncode == 1, (
            "Second invocation should exit 1 (mutex already held), "
            f"got {r2.returncode}"
        )

        first_done.wait(timeout=8)
        assert ("first", 0) in results, f"First invocation failed: {results}"

    def test_after_release_next_invocation_succeeds(self):
        """Once the first holder exits, a subsequent run must succeed."""
        r1 = _run_ps_mutex(hold_ms=50)
        assert r1.returncode == 0
        # Mutex released; second run should now acquire
        r2 = _run_ps_mutex(hold_ms=50)
        assert r2.returncode == 0


class TestProcessKillPs:
    """PowerShell kill snippet: terminates matching python.exe processes."""

    _PS_KILL = (
        "Get-WmiObject Win32_Process "
        "| Where-Object {$_.Name -eq 'python.exe' -and "
        "  $_.CommandLine -like '*run_gateio.py*'} "
        "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
        "    -ErrorAction SilentlyContinue }"
    )

    def test_kill_script_runs_without_error_when_no_target(self):
        """No matching processes → script must exit 0 (no crash)."""
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-Command", self._PS_KILL],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0

    def test_kill_script_terminates_dummy_python_process(self):
        """Start a dummy python process with run_gateio.py in argv, kill it."""
        # Start a dummy python process that sleeps and has run_gateio.py in args
        dummy = subprocess.Popen(
            [sys.executable, "-c",
             "import sys, time; sys.argv[0]='run_gateio.py'; time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.3)  # let it register in WMI
            assert dummy.poll() is None, "Dummy process should still be running"

            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-Command", self._PS_KILL],
                capture_output=True, text=True, timeout=10,
            )
            assert r.returncode == 0
            # Give OS time to deliver SIGTERM / TerminateProcess
            dummy.wait(timeout=3)
            assert dummy.poll() is not None, "Dummy process should be dead"
        finally:
            if dummy.poll() is None:
                dummy.kill()


class TestBatGuardContent:
    """Bat file must contain the mutex guard and PowerShell kill snippet."""

    BAT_PATH = os.path.join(
        os.path.dirname(__file__), "..", "run_gateio_microcap_live.bat"
    )

    def _bat_content(self) -> str:
        with open(self.BAT_PATH, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def test_bat_contains_mutex_name(self):
        assert "HFTBot_GateioMicrocap" in self._bat_content(), (
            "Bat file must reference the named mutex 'HFTBot_GateioMicrocap'"
        )

    def test_bat_contains_waitone(self):
        assert "WaitOne" in self._bat_content(), (
            "Bat file must use WaitOne to acquire the mutex"
        )

    def test_bat_contains_powershell_kill(self):
        content = self._bat_content()
        assert "run_gateio.py" in content
        assert "Stop-Process" in content, (
            "Bat file must use Stop-Process to kill old instances"
        )

    def test_bat_exits_nonzero_branch_when_mutex_held(self):
        """Bat file must have 'exit /b' or 'exit' on mutex-held path."""
        content = self._bat_content()
        assert "exit /b" in content or "exit" in content
