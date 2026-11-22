"""verify.py decision logic, with a fake checker model (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify import verify_reply  # noqa: E402


class FakeChecker:
    def __init__(self, out):
        self.out = out

    def generate_json(self, system, user, schema):
        return self.out


def checked(sentences, **kw):
    out = {"sentences": sentences, "promises": [], "answers_question": True, "language_ok": True,
           "reply_language": "nl", **kw}
    return verify_reply(FakeChecker(out), "faq", "<ticket/>", "reply", "nl")


def test_all_facts_cited_passes():
    v = checked([{"text_en": "Hi Anouk,", "kind": "greeting", "support": "none"},
                 {"text_en": "Go to Team → Add employee.", "kind": "fact", "support": "F02"}])
    assert v.passed and v.problems == []


def test_uncited_fact_fails_even_if_checker_is_lenient():
    v = checked([{"text_en": "We do not have webhooks.", "kind": "fact", "support": "none"}])
    assert not v.passed and v.problems[0].startswith("unsupported:")


def test_promise_language_and_relevance_fail():
    assert not checked([], promises=["10% discount"]).passed
    assert not checked([], language_ok=False).passed
    assert not checked([], answers_question=False).passed
    assert not checked([], reply_language="en").passed
