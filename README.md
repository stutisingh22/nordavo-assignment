# Roosterly support triage agent

Reads each open ticket, decides language, category, priority and **one** action (reply in the customer's
language, English handoff note, or close), and posts it exactly once. Gemini 3.5 Flash drafts; code checks.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env          # fill in helpdesk key, candidate email/name, GEMINI_API_KEY
python -m pytest -q           # 40 offline tests, no API calls
python eval.py --tag check    # eval on the labelled examples -> eval/results/
python run.py --reset         # final clean run: reset, then all 30 tickets once -> logs/run_<id>.jsonl
python report.py              # readable breakdown incl. English translations -> logs/run_<id>.md
```
`python run.py` (no flag) is safe to repeat: handled tickets are skipped, so after a crash just run it again.
`--dry-run` decides without posting; `python agent.py T005` tries one ticket.

## Decisions

- **Ticket text is data.** It is sent inside `<ticket>` tags, hidden text (zero-width characters, HTML comments)
  is stripped first, and the model flags injection attempts (caught: T007, T030). A flag alone does not force a
  handoff: T030's real question is a plain FAQ one. What injections ask for (refunds, discounts) is blocked in code.
- **The model decides, code can only make it safer.** After the model: always-human rules (refund, cancellation,
  GDPR, security, legal, threat to leave; nl/en/de keywords plus the model's own flags) force a handoff and a
  minimum priority; a reply also needs an FAQ citation, confidence ≥ 0.7, the right language, only menu paths
  and links that exist in the FAQ, and no promises. Any failure turns the reply into a handoff.
- **Dutch without reading Dutch.** The drafter writes an English back-translation, and a *separate*
  model (Flash-Lite) re-translates the reply sentence by sentence and must cite the FAQ entry behind each fact.
  Code, not the checker, fails the reply if any sentence has no source. `report.py` shows both translations.
- **Payment details** (cards, IBAN, CVC) are masked before the model, the logs, the journal and the note (T008).
- **Handoff = no customer message.** The API allows one action per ticket, so we do not also send "a colleague will
  contact you". Feature requests get the F10 roadmap reply and never a claim about whether a feature exists.
- **Edge cases:** an empty body is closed without a model call; tickets over 4000 characters, mixed requests
  where any part needs a human, and languages other than nl/en/de go to a human (language field `en`).

## Reliability

- **503 / network errors:** retried with exponential backoff and full jitter.
- **429:** wait `Retry-After`. We also throttle to 4 req/s (limit 5).
- **409:** never retried; it means the ticket is already handled, recorded as done.
- **No customer gets two answers:** before each POST the agent re-reads the ticket from the server (`handled`
  set → skip) and writes the exact decision to an fsync'd journal (`state/`). After a crash, a pending decision
  is resent unchanged (no new model call); a lock file stops two runs at once. Tested offline in `tests/test_state.py`.
- A `/status` pre-flight shows which account is used and stops cleanly if the helpdesk is down or the key is wrong.
- Model failures, invalid decisions or a helpdesk error leave the ticket **open** for the next run, never a guess.
- **Logs:** one JSON line per ticket in `logs/`, redacted.
- **Keys:** only from env vars (`.env.example`).

## Eval and fix

**eval.py** runs the agent on 11 labelled tickets: Nordavo's 10 examples (labels fetched live from /examples) and T016, which I labelled by hand. The model never sees or writes a label. Each run scores four fields (action, category, priority, language) and two safety counts: unsafe, a ticket that needed a human but got a reply or a close, and over-escalated, an FAQ question needlessly sent to a human.

**Fix: feature requests.** The 10 examples scored 100% from the start, but a dry run on T016 ("do you have
webhooks?") showed the gap: the model either claimed things the FAQ does not say or escalated it. I added T016
as an eval case, a prompt rule (feature request → F10 roadmap reply only) and two code checks (menu paths must
exist in the FAQ; feature-request replies may not claim what exists).

| eval/results/ | action | category | all four | unsafe | over-escalated |
|---|---|---|---|---|---|
| before (`151153_before_fix`, 2 runs) | 91% | 91% | 91% | 0 | 1 |
| after (`151452_after_fix_v2`, 2 runs) | 100% | 100% | 100% | 0 | 0 |
| latest (`185411_unsupported_lang`) | 100% | 100% | 100% | 0 | 0 |

**Final run** (`logs/run_20261122T211354-d9d9b3`): 30 posted, 13 replies, 15 handoffs, 2 closes; P1 for the outage,
the suspicious login and the lawyer (T006, T018, T028). **Cost:** $0.186 for 30 tickets (85k input / 11k output
tokens), **≈ $0.62 per 100 tickets** at list price (Flash $1.50/$9.00, Flash-Lite $0.30/$2.50 per 1M tokens).
Low thinking halved output tokens with equal eval scores. Development spend, mostly repeated evals, was ~$1.60.

## What I would improve

- **Handoff notes:** in the final run 1 of 15 notes (T003) told the human to "process the necessary refund"
  instead of "check the billing record". The decision itself was right (handoff, P2, nothing promised). Fix: a code
  check for granting verbs in notes plus an eval case.
- **A bigger eval.** 11 cases is small and several prompt rules came from them; I would label all 30 and add
  hard cases (mixed languages, polite legal threats) to test the keyword rules.
- **Other models.** All model calls go through one method, `generate_json(system, user, schema)`, so Anthropic or
  OpenAI needs one client class (~60 lines) and a new eval run; the prompt is tuned on Gemini.
- **Cheaper handoffs:** when a keyword rule already forces a human, Flash-Lite could write the note.
- **For the client:** confirm whether handoffs should send an automatic "we've passed this on" message to the customer.
