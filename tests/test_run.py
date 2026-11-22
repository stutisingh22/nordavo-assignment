"""run.preflight: stop cleanly when the helpdesk is down, never touch tickets."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run  # noqa: E402
from helpdesk import HelpdeskError  # noqa: E402


class FlakyStatus:
    def __init__(self, failures, status=503):
        self.failures, self.code, self.calls = failures, status, 0

    def status(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise HelpdeskError(self.code, "service_unavailable")
        return {"mode": "practice", "open_tickets": 30}


def test_preflight_waits_then_succeeds():
    hd = FlakyStatus(failures=2)
    assert run.preflight(hd, attempts=4, wait_s=0)["open_tickets"] == 30 and hd.calls == 3


def test_preflight_gives_up_with_clear_message():
    with pytest.raises(SystemExit, match="No ticket was touched"):
        run.preflight(FlakyStatus(failures=99), attempts=3, wait_s=0)


def test_preflight_does_not_wait_on_bad_key():
    hd = FlakyStatus(failures=99, status=401)
    with pytest.raises(SystemExit, match="check .env"):
        run.preflight(hd, attempts=4, wait_s=0)
    assert hd.calls == 1
