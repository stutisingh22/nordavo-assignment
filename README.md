# Roosterly support triage agent

Triages every open ticket and posts one decision per ticket, exactly once. Gemini 3.5 Flash drafts; code checks.

## How to run

```bash
git clone https://github.com/stutisingh22/nordavo-assignment.git && cd nordavo-assignment
python3 -m venv .venv && source .venv/bin/activate          # Python 3.10+
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env        # fill in the helpdesk key, your email and name, GEMINI_API_KEY
python -m pytest -q         # 40 offline tests, no API calls
python eval.py --tag check  # eval on the labelled tickets -> eval/results/
python run.py --reset       # final run: reset, then all 30 tickets once -> logs/run_<id>.jsonl
python report.py            # readable report with English translations -> logs/run_<id>.md
```

`python run.py` without `--reset` is safe to repeat after a crash; `--dry-run` posts nothing.

## Decisions

1. **Work in Dutch directly, check in English.** One Gemini call reads the Dutch ticket, writes the reply and an
   English back-translation. A second, independent model (Flash-Lite) translates the reply sentence by sentence and
   must name the FAQ entry behind every fact; code fails the reply if any sentence has none. `report.py` shows both
   translations. Cost: the checker is 8% of the run ($0.014 of $0.186) plus the back-translation tokens.
   Translating first would add a call per ticket and risk losing the exact Dutch menu names.
2. **The model proposes, code can only make it safer.** After the model, code forces a handoff (and a minimum
   priority) for the client's always-human topics (nl/en/de keywords plus the model's own flags) and payment
   details. A reply also needs an FAQ citation, the right language, only FAQ menu paths and links, no promises
   and the checker's pass. Code never turns a handoff into a reply.
3. **Answer what the FAQ covers, escalate the rest.** Escalating everything is punished, so how-tos and feature
   requests get replies (feature requests: the F10 roadmap answer only, never whether a feature exists).

**Assumptions where the brief was open.** *Unsure* = confidence below 0.7 or any failed check: the ticket goes to a human. A handoff sends no message to the customer (the API allows one action per ticket). Requests mixing an FAQ question with an always-human topic go to a human. Languages other than nl/en/de go to a human, with language `en`. An upgrade question is a how-to (F05), not a contract change. Empty tickets are closed without a model call.

**AI tools.** Built with Claude Code as a pair programmer: it wrote most of the code from my decisions; I set the rules, labelled the extra eval case, reviewed every change and read the English translations of every run.

## Reliability

- **503 / network errors:** retried with exponential backoff and full jitter. **429:** wait `Retry-After`; we also throttle to 4 requests/s (limit 5). **409:** never retried, recorded as already handled.
- **Crash:** before each POST the agent re-reads the ticket (`handled` set → skip) and writes the exact decision to an fsync'd journal (`state/`). After a crash, a pending decision is resent unchanged; a lock file stops two runs at once. Any failure leaves the ticket open for the next run, never a guess.
- **Security:** keys only from env vars (`.env` is git-ignored). Ticket text is data: it goes inside `<ticket>`
  tags, hidden text (zero-width characters, HTML comments) is stripped first, and injections are flagged and ignored (T007, T030). Card numbers, IBANs and CVCs are masked before the model, logs, journal and notes (T008).

## Eval and fix

`eval.py` runs the agent on 11 labelled tickets: Nordavo's 10 examples (labels fetched live from `/examples`) and T016, which I labelled by hand. The model never sees or writes a label. It scores action, category, priority and language, plus **unsafe** (needed a human, got a reply or close) and **over-escalated** (FAQ question sent to a human).

**Fix: feature requests.** The 10 examples passed from the start, but a dry run showed T016 ("do you have webhooks?") either invented claims and a menu path or was escalated. I added T016 to the eval, a prompt rule
(F10 roadmap reply only) and two code checks (menu paths must exist in the FAQ; no claims about what exists).

| eval/results/ (2 repeats each) | action | category | all four | unsafe | over-escalated |
| ------------------------------ | ------ | -------- | -------- | ------ | -------------- |
| before: `151153_before_fix`  | 91%    | 91%      | 91%      | 0      | 1              |
| after: `151452_after_fix_v2` | 100%   | 100%     | 100%     | 0      | 0              |

**Final run** (`logs/run_20261122T224913-df9bec`): 30 posted: 13 replies, 15 handoffs, 2 closes.
**Cost per 100 tickets:** drafter 66,154 in × $1.50/1M + 8,068 out × $9.00/1M = $0.172; checker 18,912 in ×
$0.30/1M + 3,385 out × $2.50/1M = $0.014; total $0.186 / 30 × 100 = **$0.62**. Low thinking halved output tokens at
equal eval scores.

## What I would improve

- **Next step:** one of 15 handoff notes (T003) told the human to "process the refund if applicable" instead of only "check the billing record" (the decision itself was right). Add a code check for granting verbs in notes, and grow the eval beyond 11 cases with more human labels.
- **Other models:** the agent only works with Gemini today, but every model call goes through one method, `generate_json(system, user, schema)`. Adding other models means one new client class or library behind that method, then re-running the eval, because the prompt was tuned on Gemini.
- **A rule I would change:** allow an automatic acknowledgement on handoff. Now a customer whose ticket goes to a human hears nothing, because the API allows only one action per ticket.
