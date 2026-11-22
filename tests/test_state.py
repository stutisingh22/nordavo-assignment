"""Crash/retry scenarios for state.py against a fake helpdesk (no network)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk import AlreadyHandled, HelpdeskError  # noqa: E402
from state import StateStore, needs_decision, post_once  # noqa: E402

DECISION = {"language": "nl", "category": "question", "priority": "P4", "reason": "F02", "draft_reply": "Hoi"}


class FakeHelpdesk:
    """Mimics the server: one answer per ticket, 409 on the second."""

    def __init__(self):
        self.handled: dict[str, dict] = {}
        self.posts = 0
        self.fail_next: Exception | None = None
        self.crash_after_save = False

    def get_ticket(self, tid):
        return {"id": tid, "handled": self.handled.get(tid)}

    def _post(self, tid, decision):
        self.posts += 1
        if self.fail_next:
            exc, self.fail_next = self.fail_next, None
            raise exc
        if tid in self.handled:
            raise AlreadyHandled(409, "already_handled")
        self.handled[tid] = decision
        if self.crash_after_save:  # server saved it, then our process died
            raise KeyboardInterrupt("simulated crash")
        return {"ok": True}

    reply = handoff = close = _post


@pytest.fixture
def env(tmp_path):
    return FakeHelpdesk(), StateStore(tmp_path)


def test_normal_post_then_rerun_skips(env):
    hd, store = env
    assert needs_decision(hd, store, "T1") == (True, None)
    assert post_once(hd, store, "T1", "reply", DECISION).status == "posted"
    assert needs_decision(hd, store, "T1") == (False, None)
    assert hd.posts == 1


def test_crash_after_server_saved_never_reposts(env):
    hd, store = env
    hd.crash_after_save = True
    with pytest.raises(KeyboardInterrupt):
        post_once(hd, store, "T1", "reply", DECISION)
    # journal says pending, server says handled -> skip, record done
    assert store.latest()["T1"]["event"] == "pending"
    assert needs_decision(hd, store, "T1") == (False, None)
    assert store.latest()["T1"]["event"] == "done"
    assert hd.posts == 1


def test_crash_before_post_resends_same_decision(env):
    hd, store = env
    store.mark_pending("T1", "reply", DECISION)  # crashed right after writing intent
    ok, saved = needs_decision(hd, store, "T1")
    assert ok and saved == {"action": "reply", "decision": DECISION}  # resend exactly, no new LLM call


def test_503_exhausted_marks_failed_and_retry_is_safe(env):
    hd, store = env
    hd.fail_next = HelpdeskError(503, "service_unavailable")
    assert post_once(hd, store, "T1", "reply", DECISION).status == "failed"
    assert needs_decision(hd, store, "T1") == (True, None)
    assert post_once(hd, store, "T1", "reply", DECISION).status == "posted"
    assert len(hd.handled) == 1


def test_409_is_not_retried_and_counts_as_done(env):
    hd, store = env
    hd.handled["T1"] = {"by": "someone else"}
    out = post_once(hd, store, "T1", "handoff", DECISION)
    assert out.status == "already_handled"
    assert hd.posts == 1
    assert store.latest()["T1"]["event"] == "done"


def test_local_done_blocks_even_if_server_reopened(env):
    hd, store = env
    store.mark_done("T1", "reply", "posted")  # stale journal from before a reset
    assert needs_decision(hd, store, "T1") == (False, None)


def test_torn_last_line_is_ignored(env):
    hd, store = env
    store.mark_done("T1", "reply", "posted")
    with open(store.journal, "a") as fh:
        fh.write('{"ticket_id": "T2", "ev')  # power cut mid-write
    assert set(store.latest()) == {"T1"}


def test_second_run_is_refused_while_locked(tmp_path):
    a, b = StateStore(tmp_path), StateStore(tmp_path)
    a.acquire_lock()
    with pytest.raises(SystemExit):
        b.acquire_lock()
    a.release_lock()
    b.acquire_lock()
    b.release_lock()


def test_prefetched_ticket_is_reused_not_fetched_again(env):
    hd, store = env
    calls = []
    real_get = hd.get_ticket
    hd.get_ticket = lambda tid: calls.append(tid) or real_get(tid)
    ticket = hd.get_ticket("T1")
    assert needs_decision(hd, store, "T1", ticket) == (True, None)
    assert calls == ["T1"]  # only the caller's own GET
