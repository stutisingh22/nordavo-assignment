"""Eval: run the agent on Nordavo's 10 labelled examples and score it against THEIR labels.

Labels come straight from GET /examples (written by Nordavo); the model under test never
writes or sees them. The examples are not in the prompt either, so this is a fair held-out test.

Scores per example: action, category, priority, language, FAQ id (when Nordavo's reason cites one).
Safety (most important): a ticket Nordavo sends to a human must never be replied to or closed.

Usage:  python eval.py --tag before            (results -> eval/results/<time>_<tag>.json)
        python eval.py --tag after --repeat 3  (repeat to see run-to-run variance)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from agent import decide
from helpdesk import HelpdeskClient
from llm import Gemini

OUT_DIR = Path(__file__).resolve().parent / "eval" / "results"
FIELDS = ["action", "category", "priority", "language"]


def score_one(example: dict, decision) -> dict:
    label = example["label"]
    got = {"action": decision.action, **{f: decision.payload[f] for f in ["category", "priority", "language"]}}
    row = {"id": example["id"], "expected": {f: label[f] for f in FIELDS}, "got": got}
    row["match"] = {f: got[f] == label[f] for f in FIELDS}

    # Safety: a human-only ticket that was answered or closed by the agent.
    row["unsafe"] = label["action"] == "handoff" and decision.action != "handoff"
    # Usefulness: an FAQ question that the agent escalated anyway.
    row["over_escalated"] = label["action"] == "reply" and decision.action == "handoff"

    # FAQ grounding: if Nordavo's reason cites an FAQ id, the agent's reply should use it.
    expected_faq = set(re.findall(r"F\d{2}", label.get("reason", "")))
    if label["action"] == "reply" and expected_faq:
        row["faq_ok"] = decision.action == "reply" and bool(expected_faq & set(decision.faq_ids))
    row["overrides"] = decision.overrides
    row["reply_en"] = decision.reply_en
    row["nordavo_reason"] = label.get("reason", "")
    row["agent_reason"] = decision.payload.get("reason", "")
    return row


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    s = {f: sum(r["match"][f] for r in rows) / n for f in FIELDS}
    s["all_four"] = sum(all(r["match"].values()) for r in rows) / n
    s["unsafe"] = sum(r["unsafe"] for r in rows)
    s["over_escalated"] = sum(r["over_escalated"] for r in rows)
    faq = [r["faq_ok"] for r in rows if "faq_ok" in r]
    s["faq_grounding"] = sum(faq) / len(faq) if faq else None
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    hd = HelpdeskClient()
    examples = hd.examples()["examples"]
    hd.close_client()
    llm = Gemini()

    runs = []
    for r in range(args.repeat):
        rows = [score_one(ex, decide(ex, llm)) for ex in examples]
        runs.append({"rows": rows, "summary": summarise(rows)})

    # Print the last run in detail, plus the summary of every run
    print(f"\n{'id':4} {'action':18} {'category':26} {'priority':8} {'lang':6} notes")
    for row in runs[-1]["rows"]:
        cells = []
        for f, w in zip(FIELDS, [18, 26, 8, 6]):
            e, g_ = row["expected"][f], row["got"][f]
            cells.append(f"{e if e == g_ else f'{e}->{g_}':{w}}")
        notes = ("UNSAFE " if row["unsafe"] else "") + ("over-escalated " if row["over_escalated"] else "")
        notes += ("faq-miss " if row.get("faq_ok") is False else "") + (",".join(row["overrides"]))
        print(f"{row['id']:4} " + " ".join(cells) + " " + notes)
    print("\n(expected->got shown only where they differ)")
    for i, run in enumerate(runs, 1):
        s = run["summary"]
        print(f"run {i}: action {s['action']:.0%} | category {s['category']:.0%} | priority {s['priority']:.0%} | "
              f"language {s['language']:.0%} | all four {s['all_four']:.0%} | unsafe {s['unsafe']} | "
              f"over-escalated {s['over_escalated']} | faq grounding "
              f"{'-' if s['faq_grounding'] is None else format(s['faq_grounding'], '.0%')}")
    u = llm.usage
    print(f"tokens: {u.calls} calls, in={u.input_tokens}, out={u.output_tokens}, model={llm.model}, thinking={llm.thinking or 'default'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.tag}.json"
    path.write_text(json.dumps({
        "tag": args.tag, "model": llm.model, "thinking": llm.thinking or "default", "runs": runs,
        "usage": {"calls": u.calls, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens},
    }, ensure_ascii=False, indent=2))
    print(f"saved {path.relative_to(Path(__file__).resolve().parent)}")


if __name__ == "__main__":
    main()
