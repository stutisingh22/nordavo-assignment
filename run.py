"""Run the agent: handle every open ticket once.

    python run.py              handle all open tickets (safe to re-run; continues after a crash)
    python run.py --reset      POST /practice/reset first, then handle all 30 (the final clean run)
    python run.py --dry-run    decide but post nothing
    python run.py --only T005,T007

One JSON line per ticket goes to logs/run_<time>.jsonl (payment details redacted).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from agent import InvalidDecision, decide
from helpdesk import HelpdeskClient, HelpdeskError
from llm import PRICES, Gemini, LLMError
from redact import redact_obj
from state import StateStore, needs_decision, post_once
from verify import make_verifier

LOG_DIR = Path(__file__).resolve().parent / "logs"


def cost_usd(llm: Gemini) -> float:
    p_in, p_out = PRICES.get(llm.model, (0.0, 0.0))
    return llm.usage.input_tokens / 1e6 * p_in + llm.usage.output_tokens / 1e6 * p_out


def preflight(hd: HelpdeskClient, attempts: int = 4, wait_s: int = 30) -> dict:
    """GET /status before touching any ticket (Nordavo FAQ: "wait a minute and try GET /status").

    Each call already retries 503s and network errors; if the helpdesk is still down after
    ~2 minutes we stop with a clear message instead of failing ticket by ticket.
    """
    for i in range(1, attempts + 1):
        try:
            return hd.status()
        except HelpdeskError as exc:
            if exc.status in (400, 401, 404):  # config problem (key, email, URL): waiting will not help
                sys.exit(f"Helpdesk refused the request ({exc.status} {exc.error}): check .env")
            print(f"helpdesk not responding ({exc.status} {exc.error}), attempt {i}/{attempts}"
                  + (f", retrying in {wait_s}s" if i < attempts else ""))
            if i < attempts:
                time.sleep(wait_s)
    sys.exit("Helpdesk still not responding after ~2 minutes. No ticket was touched. "
             "Try again later; if it persists, email Nordavo (they move the deadline if it is on their side).")


def log_line(fh, record: dict) -> None:
    fh.write(json.dumps(redact_obj(record), ensure_ascii=False) + "\n")
    fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="POST /practice/reset first")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not post")
    ap.add_argument("--only", default="", help="comma-separated ticket ids")
    args = ap.parse_args()
    if args.reset and args.dry_run:
        sys.exit("--reset with --dry-run makes no sense")

    hd, store, llm, verifier = HelpdeskClient(), StateStore(), Gemini(), make_verifier()
    store.acquire_lock()
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{run_id}{'_dry' if args.dry_run else ''}.jsonl"
    counts: dict[str, int] = {}

    try:
        st = preflight(hd)
        print(f"helpdesk: mode={st.get('mode')} account={st.get('candidate')} <{os.environ.get('CANDIDATE_EMAIL')}> "
              f"open={st.get('open_tickets')} handled={st.get('handled_tickets')}")
        if args.reset:
            print("reset:", hd.reset())
            if old := store.archive():
                print(f"archived old journal -> {old.name}")

        # Snapshot the open ids first: handling tickets changes the 'open' list we page through.
        ids = [t["id"] for t in hd.iter_tickets(status="open")]
        if args.only:
            wanted = set(args.only.split(","))
            ids = [i for i in ids if i in wanted]
        print(f"run {run_id}: {len(ids)} open tickets, model={llm.model}, thinking={llm.thinking or 'default'}"
              f"{' (DRY RUN)' if args.dry_run else ''}")

        if not ids:
            print("nothing to do: no open tickets")
            return 0
        with open(log_path, "a", encoding="utf-8") as log:
            for tid in ids:
                rec = {"run_id": run_id, "ticket_id": tid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                try:
                    ticket = hd.get_ticket(tid)  # one GET: the 'handled' check and the text
                    todo, pending = needs_decision(hd, store, tid, ticket)
                    if not todo:
                        rec["outcome"] = "skipped_already_handled"
                    elif pending:
                        # Crash recovery: resend the exact decision we were about to send.
                        rec.update(action=pending["action"], resumed=True)
                        rec["outcome"] = "dry_run" if args.dry_run else \
                            post_once(hd, store, tid, pending["action"], pending["decision"]).status
                    else:
                        calls_before, v_before = llm.usage.calls, verifier.usage.calls
                        d = decide(ticket, llm, verifier)
                        tokens = llm.usage.by_call[-1] if llm.usage.calls > calls_before else (0, 0)
                        v_tokens = verifier.usage.by_call[-1] if verifier.usage.calls > v_before else (0, 0)
                        rec.update(
                            subject=ticket.get("subject"), customer_plan=ticket.get("customer_plan"),
                            action=d.action, model_action=d.model_action, overrides=d.overrides,
                            language=d.payload["language"], category=d.payload["category"],
                            priority=d.payload["priority"], confidence=d.payload["confidence"],
                            faq_ids=d.faq_ids, injection_detected=d.injection_detected,
                            payment_details_found=d.payment_details_found,
                            reason=d.payload["reason"], draft_reply=d.payload.get("draft_reply"),
                            reply_en=d.reply_en or None, verified_en=d.verified_en or None,
                            handoff_note=d.payload.get("handoff_note"),
                            tokens_in=tokens[0], tokens_out=tokens[1],
                            verifier_tokens_in=v_tokens[0], verifier_tokens_out=v_tokens[1],
                            model=llm.model, verifier_model=verifier.model if v_tokens[0] else None,
                        )
                        rec["outcome"] = "dry_run" if args.dry_run else \
                            post_once(hd, store, tid, d.action, d.payload).status
                except LLMError as exc:
                    # Leave the ticket open: the next run picks it up. Never post a guess.
                    rec.update(outcome="llm_failed_left_open", error=str(exc)[:200])
                except HelpdeskError as exc:
                    rec.update(outcome="helpdesk_error_left_open", error=f"{exc.status} {exc.error}")
                except InvalidDecision as exc:
                    # The decision would be rejected or is incomplete: never post it, a rerun retries.
                    rec.update(outcome="invalid_decision_left_open", error=str(exc)[:200])

                counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
                log_line(log, rec)
                print(f"  {tid} {rec.get('action', '-'):8} {rec.get('priority', '')} {rec.get('category', ''):16} "
                      f"-> {rec['outcome']} {' '.join(rec.get('overrides', []))}")
    finally:
        store.release_lock()
        hd.close_client()

    print(f"\noutcomes: {counts}")
    for name, m in (("drafter", llm), ("verifier", verifier)):
        u = m.usage
        print(f"{name} {m.model}: {u.calls} calls, {u.input_tokens} in / {u.output_tokens} out tokens, "
              f"~${cost_usd(m):.3f}")
    print(f"total ~${cost_usd(llm) + cost_usd(verifier):.3f}")
    st = hd.stats
    print(f"helpdesk: {st['requests']} requests, {st['503_retried']} x 503 retried, "
          f"{st['429_waited']} x 429 waited, {st['network_error_retried']} network errors retried, "
          f"{st['409_already_handled']} x 409")
    print(f"log: {log_path.relative_to(Path(__file__).resolve().parent)}")
    failed = sum(v for k, v in counts.items() if "left_open" in k or k == "failed")
    if failed:
        print(f"{failed} ticket(s) left open: run again to retry them (already-handled tickets are skipped).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
