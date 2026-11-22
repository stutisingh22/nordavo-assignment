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
import sys
import time
import uuid
from pathlib import Path

from agent import decide
from helpdesk import HelpdeskClient, HelpdeskError
from llm import Gemini, LLMError
from redact import redact_obj
from state import StateStore, needs_decision, post_once
from verify import make_verifier

LOG_DIR = Path(__file__).resolve().parent / "logs"
# List prices (USD per 1M tokens: input, output), used only for the cost line in the summary.
PRICES = {"gemini-3.5-flash": (1.50, 9.00), "gemini-3.5-flash-lite": (0.30, 2.50)}


def cost_usd(llm: Gemini) -> float:
    p_in, p_out = PRICES.get(llm.model, (0.0, 0.0))
    return llm.usage.input_tokens / 1e6 * p_in + llm.usage.output_tokens / 1e6 * p_out


def log_line(fh, record: dict) -> None:
    fh.write(json.dumps(redact_obj(record), ensure_ascii=False) + "\n")
    fh.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="POST /practice/reset first")
    ap.add_argument("--dry-run", action="store_true", help="decide but do not post")
    ap.add_argument("--only", default="", help="comma-separated ticket ids")
    args = ap.parse_args()

    hd, store, llm, verifier = HelpdeskClient(), StateStore(), Gemini(), make_verifier()
    store.acquire_lock()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{run_id}{'_dry' if args.dry_run else ''}.jsonl"
    counts: dict[str, int] = {}

    try:
        if args.reset:
            if args.dry_run:
                sys.exit("--reset with --dry-run makes no sense")
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

        with open(log_path, "a", encoding="utf-8") as log:
            for tid in ids:
                rec = {"run_id": run_id, "ticket_id": tid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                try:
                    todo, pending = needs_decision(hd, store, tid)
                    if not todo:
                        rec["outcome"] = "skipped_already_handled"
                    elif pending:
                        # Crash recovery: resend the exact decision we were about to send.
                        rec.update(action=pending["action"], resumed=True)
                        rec["outcome"] = "dry_run" if args.dry_run else \
                            post_once(hd, store, tid, pending["action"], pending["decision"]).status
                    else:
                        calls_before = llm.usage.calls
                        d = decide(hd.get_ticket(tid), llm, verifier)
                        tokens = llm.usage.by_call[-1] if llm.usage.calls > calls_before else (0, 0)
                        rec.update(
                            action=d.action, model_action=d.model_action, overrides=d.overrides,
                            language=d.payload["language"], category=d.payload["category"],
                            priority=d.payload["priority"], confidence=d.payload["confidence"],
                            faq_ids=d.faq_ids, injection_detected=d.injection_detected,
                            payment_details_found=d.payment_details_found,
                            reason=d.payload["reason"], draft_reply=d.payload.get("draft_reply"),
                            reply_en=d.reply_en or None, verified_en=d.verified_en or None,
                            handoff_note=d.payload.get("handoff_note"),
                            tokens_in=tokens[0], tokens_out=tokens[1],
                        )
                        rec["outcome"] = "dry_run" if args.dry_run else \
                            post_once(hd, store, tid, d.action, d.payload).status
                except LLMError as exc:
                    # Leave the ticket open: the next run picks it up. Never post a guess.
                    rec.update(outcome="llm_failed_left_open", error=str(exc)[:200])
                except HelpdeskError as exc:
                    rec.update(outcome="helpdesk_error_left_open", error=f"{exc.status} {exc.error}")

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
    print(f"log: {log_path.relative_to(Path(__file__).resolve().parent)}")
    failed = sum(v for k, v in counts.items() if "left_open" in k or k == "failed")
    if failed:
        print(f"{failed} ticket(s) left open: run again to retry them (already-handled tickets are skipped).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
