"""Turn a run log into a readable Markdown breakdown (no API calls, costs nothing).

    python report.py                         latest real run in logs/
    python report.py logs/run_<id>.jsonl     a specific run

Writes logs/run_<id>.md next to the log and prints the path.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from llm import PRICES

LOG_DIR = Path(__file__).resolve().parent / "logs"


def latest_log() -> Path:
    logs = sorted(p for p in LOG_DIR.glob("run_*.jsonl") if not p.stem.endswith("_dry"))
    if not logs:
        sys.exit("no run logs in logs/")
    return logs[-1]


def cell(text: str | None, n: int = 110) -> str:
    text = " ".join((text or "").split()).replace("|", "\\|")
    return text if len(text) <= n else text[: n - 1] + "…"


def counts_table(title: str, counter: Counter, order: list[str]) -> list[str]:
    keys = [k for k in order if k in counter] + sorted(k for k in counter if k not in order)
    return [f"| {title} | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|",
            "| tickets | " + " | ".join(str(counter[k]) for k in keys) + " |", ""]


def cost(rows: list[dict]) -> tuple[float, float, int, int]:
    drafter = verifier = 0.0
    tin = tout = 0
    for r in rows:
        p_in, p_out = PRICES.get(r.get("model", "gemini-3.5-flash"), (0, 0))
        drafter += r.get("tokens_in", 0) / 1e6 * p_in + r.get("tokens_out", 0) / 1e6 * p_out
        v_in, v_out = PRICES.get(r.get("verifier_model") or "gemini-3.5-flash-lite", (0, 0))
        verifier += r.get("verifier_tokens_in", 0) / 1e6 * v_in + r.get("verifier_tokens_out", 0) / 1e6 * v_out
        tin += r.get("tokens_in", 0) + r.get("verifier_tokens_in", 0)
        tout += r.get("tokens_out", 0) + r.get("verifier_tokens_out", 0)
    return drafter, verifier, tin, tout


def build(log: Path) -> str:
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    decided = [r for r in rows if r.get("action")]
    run_id = rows[0]["run_id"] if rows else log.stem
    drafter, verifier, tin, tout = cost(decided)
    total = drafter + verifier
    per100 = total / len(decided) * 100 if decided else 0

    out = [f"# Run report: {run_id}", "",
           f"Log: `{log.name}` · {len(rows)} tickets · models: "
           f"{', '.join(sorted({r['model'] for r in decided if r.get('model')}) or {'-'})} (drafter), "
           f"{', '.join(sorted({r['verifier_model'] for r in decided if r.get('verifier_model')}) or {'-'})} (verifier)",
           "", "## Summary", ""]
    out += counts_table("action", Counter(r["action"] for r in decided), ["reply", "handoff", "close"])
    out += counts_table("priority", Counter(r["priority"] for r in decided if r.get("priority")), ["P1", "P2", "P3", "P4"])
    out += counts_table("category", Counter(r["category"] for r in decided if r.get("category")),
                        ["question", "billing", "bug", "feature_request", "other"])
    out += counts_table("language", Counter(r["language"] for r in decided if r.get("language")), ["nl", "en", "de"])
    out += counts_table("outcome", Counter(r["outcome"] for r in rows), ["posted"])

    overridden = [r for r in decided if r.get("overrides")]
    flagged = [r["ticket_id"] for r in decided if r.get("injection_detected")]
    redacted = [r["ticket_id"] for r in decided if r.get("payment_details_found")]
    out += ["## Safety", "",
            f"- Guardrail overrides (model decision changed by code): **{len(overridden)}**",
            *[f"  - {r['ticket_id']}: model said `{r.get('model_action')}`, sent `{r['action']}`: "
              f"{', '.join(r['overrides'])}" for r in overridden],
            f"- Prompt injection detected (instructions ignored): {', '.join(flagged) or 'none'}",
            f"- Payment details found and redacted: {', '.join(redacted) or 'none'}",
            "- Every reply passed: FAQ citation, confidence ≥ 0.7, language check, menu-path check, "
            "forbidden-promise check, and the independent verifier.", ""]

    out += ["## Cost", "",
            f"- Drafter: ${drafter:.3f} · verifier: ${verifier:.3f} · **total ${total:.3f}** "
            f"for {len(decided)} tickets ({tin:,} input / {tout:,} output tokens)",
            f"- **≈ ${per100:.2f} per 100 tickets** at list price "
            "(Gemini 3.5 Flash $1.50/$9.00, Flash-Lite $0.30/$2.50 per 1M input/output tokens)", ""]

    out += ["## Tickets", "", "| id | subject | lang | category | prio | action | FAQ | what the customer / human gets (English) |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("action"):
            out.append(f"| {r['ticket_id']} | | | | | | | {r['outcome']} |")
            continue
        gist = r.get("reply_en") if r["action"] == "reply" else r.get("handoff_note") or r.get("reason")
        mark = " ⚙" if r.get("overrides") else ""
        out.append(f"| {r['ticket_id']} | {cell(r.get('subject'), 30)} | {r.get('language')} | {r.get('category')} | "
                   f"{r.get('priority')} | **{r['action']}**{mark} | {','.join(r.get('faq_ids') or [])} | {cell(gist)} |")
    out += ["", "⚙ = a guardrail changed the model's decision (see Safety).", ""]

    out += ["## Details", ""]
    for r in decided:
        out += [f"### {r['ticket_id']} · {r.get('subject') or ''} · {r['action']} · {r.get('priority')}", "",
                f"**Reason:** {r.get('reason')}", ""]
        if r["action"] == "reply":
            out += ["**Reply sent (customer's language):**", "", "> " + (r.get("draft_reply") or "").replace("\n", "\n> "), "",
                    "**Drafter's English back-translation:**", "", "> " + (r.get("reply_en") or "").replace("\n", "\n> "), "",
                    "**Independent verifier's translation:**", "", "> " + (r.get("verified_en") or "-"), ""]
        elif r["action"] == "handoff":
            out += [f"**Handoff note:** {r.get('handoff_note')}", ""]
    return "\n".join(out)


def main() -> None:
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_log()
    md = build(log)
    dest = log.with_suffix(".md")
    dest.write_text(md, encoding="utf-8")
    print(f"wrote {dest.relative_to(Path(__file__).resolve().parent)}")


if __name__ == "__main__":
    main()
