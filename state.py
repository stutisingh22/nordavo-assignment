"""Exactly-once posting: no customer ever gets two answers.

Two guards, checked before every POST:
  1. The helpdesk itself (source of truth): GET /tickets/{id} -> "handled" is set => skip.
  2. A local append-only journal, state/decisions.jsonl, fsync'd on every write:
       pending  -> written BEFORE the POST (contains the exact decision we will send)
       done     -> written AFTER the helpdesk confirmed (2xx, or 409 = it already had one)
       failed   -> the POST failed with a non-retryable error; nothing was saved

Crash scenarios:
  - Crash before POST / POST got a 503:  journal says pending, server says open.
    -> Safe to resend. We resend the SAME decision from the journal (no new LLM call).
  - Crash after POST succeeded but before "done" was written: journal says pending,
    server says handled. -> Guard 1 skips it and we record done.
  - Connection lost mid-POST and our retry hits 409: recorded as done.
  - Journal says done but server says open: only happens after /practice/reset with an
    old journal. We do NOT post (fail safe); run.py --reset archives the journal.

A lock file stops two runs from working on the same tickets at the same time.
The journal holds no raw ticket text, and decisions are redacted before they get here.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from helpdesk import ACTIONS, AlreadyHandled, HelpdeskClient, HelpdeskError

STATE_DIR = Path(__file__).resolve().parent / "state"


@dataclass
class Outcome:
    ticket_id: str
    status: str  # posted | already_handled | failed
    detail: str = ""


class StateStore:
    def __init__(self, state_dir: Path = STATE_DIR):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.dir / "decisions.jsonl"
        self._lock_fh = None

    # ---- run lock ----------------------------------------------------------

    def acquire_lock(self) -> None:
        self._lock_fh = open(self.dir / "run.lock", "w")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another agent run is active (state/run.lock). Refusing to start.")

    def release_lock(self) -> None:
        if self._lock_fh:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()
            self._lock_fh = None

    # ---- journal -----------------------------------------------------------

    def _append(self, record: dict) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        with open(self.journal, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def latest(self) -> dict[str, dict]:
        """Last journal record per ticket id. Tolerates a torn final line after a crash."""
        out: dict[str, dict] = {}
        if not self.journal.exists():
            return out
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["ticket_id"]] = rec
        return out

    def mark_pending(self, ticket_id: str, action: str, decision: dict) -> None:
        self._append({"ticket_id": ticket_id, "event": "pending", "action": action, "decision": decision})

    def mark_done(self, ticket_id: str, action: str, how: str) -> None:
        self._append({"ticket_id": ticket_id, "event": "done", "action": action, "how": how})

    def mark_failed(self, ticket_id: str, action: str, error: str) -> None:
        self._append({"ticket_id": ticket_id, "event": "failed", "action": action, "error": error})

    def archive(self) -> Path | None:
        """Move the journal aside (used right after /practice/reset)."""
        if not self.journal.exists():
            return None
        dest = self.dir / f"decisions.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.journal.rename(dest)
        return dest


def needs_decision(hd: HelpdeskClient, store: StateStore, ticket_id: str,
                   ticket: dict | None = None) -> tuple[bool, dict | None]:
    """Should we work on this ticket?

    Returns (yes/no, pending) where pending = {"action", "decision"} if a crash left an unconfirmed
    decision for a ticket the helpdesk still shows as open: resend exactly that, no new LLM call.

    Call this BEFORE spending an LLM call on the ticket. Pass `ticket` if you just fetched it
    (GET /tickets/{id}) to avoid a second request; otherwise it is fetched here.
    """
    rec = store.latest().get(ticket_id)
    if rec and rec["event"] == "done":
        return False, None
    if (ticket or hd.get_ticket(ticket_id)).get("handled"):
        store.mark_done(ticket_id, rec["action"] if rec else "unknown", "server_already_handled")
        return False, None
    if rec and rec["event"] == "pending":
        return True, {"action": rec["action"], "decision": rec["decision"]}
    return True, None


def post_once(hd: HelpdeskClient, store: StateStore, ticket_id: str, action: str, decision: dict) -> Outcome:
    """Post exactly one decision for a ticket. Never raises for 409."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    send: Callable[[str, dict], Any] = {"reply": hd.reply, "handoff": hd.handoff, "close": hd.close}[action]

    store.mark_pending(ticket_id, action, decision)
    try:
        send(ticket_id, decision)
    except AlreadyHandled:
        store.mark_done(ticket_id, action, "409_already_handled")
        return Outcome(ticket_id, "already_handled", "helpdesk returned 409")
    except HelpdeskError as exc:
        # 503s and 429s were already retried inside the client; this is final.
        # A 503 means nothing was saved, so the ticket stays open and is safe to retry next run.
        store.mark_failed(ticket_id, action, f"{exc.status} {exc.error}")
        return Outcome(ticket_id, "failed", f"{exc.status} {exc.error}")
    store.mark_done(ticket_id, action, "posted")
    return Outcome(ticket_id, "posted")
