"""Deterministic safety checks applied AFTER the model decides.

The model is good at reading Dutch; it is not a safety mechanism. These rules can only
make a decision more conservative (reply -> handoff, raise priority); never the reverse.
Every override is recorded so it shows up in the log and the eval.
"""

from __future__ import annotations

import re

# --- 1. Client rules that always go to a human ------------------------------------
# Word stems in Dutch, English and German, matched on the (redacted) ticket text.
ALWAYS_HUMAN: dict[str, list[str]] = {
    "refund_or_disputed_charge": [
        r"terugbetal", r"terugstort", r"terug\s?storten", r"geld terug", r"terugkrijgen", r"terug krijgen",
        r"refund", r"money back", r"charged twice", r"double charged", r"dubbel afgeschreven",
        r"twee keer .{0,30}afgeschreven", r"r[uü]ckerstatt", r"erstattung", r"geld zur[uü]ck",
    ],
    "cancellation_or_contract_change": [
        r"opzeg", r"zeggen we op", r"zeg ik op", r"abonnement .{0,20}(stoppen|be[eë]indigen)",
        r"cancel (our|my|the)? ?(subscription|account|contract|plan)", r"terminate", r"k[uü]ndig",
    ],
    "personal_data_request": [
        r"\bavg\b", r"\bgdpr\b", r"\bdsgvo\b", r"persoonsgegevens", r"inzageverzoek", r"inzage in",
        r"gegevens .{0,30}verwijder", r"personal data", r"delete my data", r"personenbezogen",
    ],
    "security_incident": [
        r"gehackt", r"hacked", r"datalek", r"data ?breach", r"vreemde login", r"verdachte login",
        r"suspicious login", r"niet wij", r"wasn'?t us", r"waren wij niet", r"onbekende? .{0,20}ingelogd",
        r"phishing", r"unauthori[sz]ed", r"inbraak",
    ],
    "legal_threat": [
        r"advoca", r"lawyer", r"attorney", r"anwalt", r"rechtszaak", r"juridische stappen",
        r"schadevergoeding", r"aansprakelijk", r"legal action", r"\bsue\b", r"compensation", r"schadenersatz",
    ],
    "threat_to_leave": [
        r"zeggen we op", r"overstappen", r"andere aanbieder", r"concurrent", r"we gaan weg",
        r"switch(ing)? to (another|a different|a competitor)", r"leave roosterly", r"anbieter wechseln",
    ],
}
# Minimum priority per rule (P1 = most urgent). Security/legal are P1 by the brief's definition.
RULE_MIN_PRIORITY = {
    "security_incident": "P1",
    "legal_threat": "P1",
    "refund_or_disputed_charge": "P2",
    "cancellation_or_contract_change": "P2",
    "personal_data_request": "P2",
    "threat_to_leave": "P2",
}


def always_human_hits(text: str) -> list[str]:
    t = text.lower()
    return [rule for rule, pats in ALWAYS_HUMAN.items() if any(re.search(p, t) for p in pats)]


# --- 2. Language check (no Dutch reading required) --------------------------------
STOPWORDS = {
    "nl": "de het een en is ik we wij jullie niet van voor op naar met kan hoe wat waar graag onze ons mijn dat er "
          "je u uw zijn wordt bij ook al nog dan als maar of hebben heeft deze die dit".split(),
    "en": "the and is i we you not of for on to with can how what where please our my that it "
          "your are be this have has from at as or but if will".split(),
    "de": "der die das und ist ich wir nicht von für auf zu mit kann wie was wo bitte unsere mein dass es "
          "auch gibt sie ihr ihre sind wird bei den dem ein eine zum zur oder".split(),
}


def detect_language(text: str) -> tuple[str | None, dict]:
    """Stopword vote. Returns (lang or None if unsure, scores)."""
    words = re.findall(r"[a-zA-ZÀ-ſ']+", text.lower())
    scores = {lang: sum(w in sw for w in words) for lang, sw in STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 >= 3 and s1 >= 2 * s2:
        return best, scores
    return None, scores


# --- 3. Things a reply must never say --------------------------------------------
# A drafted reply may not promise money, credit or human action. Those need a human.
FORBIDDEN_IN_REPLY = [
    r"terugbetal", r"terugstort", r"refund", r"r[uü]ckerstatt", r"korting", r"discount", r"rabatt",
    r"compensat", r"schadevergoeding", r"schadenersatz", r"tegoed", r"gutschrift", r"\bcredit\b",
]
# Only these roosterly.nl hosts exist in the FAQ; any other roosterly link was invented.
ALLOWED_HOSTS = {"app.roosterly.nl", "roadmap.roosterly.nl", "status.roosterly.nl"}
ROOSTERLY_HOST = re.compile(r"\b((?:[a-z0-9-]+\.)*roosterly\.nl)\b", re.IGNORECASE)
OTHER_LINK = re.compile(r"https?://(?![a-z0-9.-]*roosterly\.nl)\S+|www\.\S+", re.IGNORECASE)


# Characters a nl/en/de reply can legitimately contain. Anything else (e.g. "GrüÔe" seen in a
# test run instead of "Grüße") means the model garbled the text: a human should look.
UNEXPECTED_CHAR = re.compile(r"[^\x09\x0a\x0d\x20-\x7eäöüßÄÖÜéèëêïíóòôúûçáàÉË€→‘’“”–—…·]")


def reply_problems(reply: str) -> list[str]:
    problems = []
    low = reply.lower()
    for p in FORBIDDEN_IN_REPLY:
        if re.search(p, low):
            problems.append(f"forbidden_promise:{p}")
    for m in ROOSTERLY_HOST.finditer(reply):
        if m.group(1).lower() not in ALLOWED_HOSTS:
            problems.append(f"unknown_link:{m.group(1)}")
    if OTHER_LINK.search(reply):
        problems.append("external_link")
    if bad := sorted(set(UNEXPECTED_CHAR.findall(reply))):
        problems.append("garbled_characters:" + "".join(bad))
    if "[" in reply and "]" in reply:
        problems.append("placeholder_or_redaction_in_reply")
    return problems


PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]


def raise_priority(current: str, minimum: str) -> str:
    return min(current, minimum, key=PRIORITY_ORDER.index)
