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
        r"storne", r"terugboek", r"crediter", r"creditnota", r"betwist", r"chargeback",
        r"onterecht .{0,30}(afgeschreven|gefactureerd|in rekening|betaald)",
        r"te ?veel .{0,15}(betaald|gefactureerd|afgeschreven|in rekening)", r"overcharged", r"wrongly charged",
        r"zu viel berechnet", r"doppelt abgebucht",
    ],
    "cancellation_or_contract_change": [
        r"opzeg", r"zeggen we op", r"zeg ik op", r"abonnement .{0,20}(stoppen|be[eë]indigen)",
        r"cancel (our|my|the)? ?(subscription|account|contract|plan)", r"terminate", r"k[uü]ndig",
        r"(account|abonnement) .{0,15}(opheffen|verwijderen)", r"downgrad",
        r"contract .{0,20}(wijzig|aanpass|be[eë]indig)", r"close (my|our) account",
        r"end (our|my|the) contract", r"vertrag .{0,20}beenden",
    ],
    "personal_data_request": [
        r"\bavg\b", r"\bgdpr\b", r"\bdsgvo\b", r"persoonsgegevens", r"inzageverzoek", r"inzage in",
        r"gegevens .{0,30}verwijder", r"personal data", r"delete my data", r"personenbezogen",
        r"recht op (vergetelheid|inzage)", r"right to be forgotten", r"datenschutz", r"daten .{0,20}l[oö]schen",
    ],
    "security_incident": [
        r"gehackt", r"hacked", r"datalek", r"data ?breach", r"vreemde login", r"verdachte login",
        r"suspicious login", r"niet wij", r"wasn'?t us", r"waren wij niet", r"onbekende? .{0,20}ingelogd",
        r"phishing", r"unauthori[sz]ed", r"inbraak",
        r"account .{0,20}overgenomen", r"verdachte (activiteit|mail|inlog)", r"beveiligingslek",
        r"kwetsbaarhe", r"vulnerabilit", r"compromis", r"sicherheitsl[uü]cke",
        r"wachtwoord .{0,20}(gelekt|gestolen)", r"someone else .{0,20}(logged|access)",
    ],
    "legal_threat": [
        r"advoca", r"lawyer", r"attorney", r"anwalt", r"rechtszaak", r"juridische stappen",
        r"schadevergoeding", r"aansprakelijk", r"legal action", r"\bsue\b", r"compensation", r"schadenersatz",
        r"rechtsanwalt", r"\bklage\b", r"gerichtlich", r"ingebrekestelling", r"naar de rechter", r"\bjurist",
        r"lawsuit", r"\bcourt\b",
    ],
    "threat_to_leave": [
        r"zeggen we op", r"overstappen", r"andere aanbieder", r"concurrent", r"we gaan weg",
        r"switch(ing)? to (another|a different|a competitor)", r"leave roosterly", r"anbieter wechseln",
        r"stoppen met (roosterly|jullie)", r"we stappen over", r"(overweeg|overwegen) .{0,30}(stoppen|over te stappen)",
        r"looking for (an )?alternative", r"other provider",
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


# --- 4. Menu paths must exist in the FAQ ---------------------------------------------
# Replies give click-paths like "Instellingen → Abonnement → Facturen". An invented path
# (e.g. "Instellingen → Koppelingen" for API access, seen in a dry run) sends customers to a
# screen that may not exist. Every path in a reply must exactly match one in the FAQ.
_SEGMENT = r"[A-ZÄÖÜ][\w-]*(?: [a-zäöüß][\w-]*)?"
MENU_PATH = re.compile(rf"{_SEGMENT}(?:\s*→\s*{_SEGMENT})+")


def _norm_path(p: str) -> str:
    return " → ".join(s.strip() for s in p.split("→"))


def extract_menu_paths(text: str) -> list[str]:
    text = re.sub(r"[*_\"'“”‘’]", "", text).replace("->", "→")
    return [_norm_path(m.group(0)) for m in MENU_PATH.finditer(text)]


def menu_path_problems(reply: str, allowed: set[str]) -> list[str]:
    problems = []
    for path in extract_menu_paths(reply):
        segs = path.split(" → ")
        # The last segment may have swallowed the next word of the sentence ("Abonnement kunt").
        variants = {path, " → ".join(segs[:-1] + [segs[-1].split(" ")[0]])}
        if not variants & allowed:
            problems.append(f"unknown_menu_path:{path}")
    return problems


# --- 5. Claims about what the product has -------------------------------------------
# The FAQ never says a requested feature is missing or planned, so a feature-request reply that
# says so is invented. Callers apply this to feature_request replies only. Checked on the English back-translation (we cannot regex every Dutch phrasing).
EXISTENCE_CLAIM = re.compile(
    r"\b(we (currently )?(do not|don't|does not|doesn't) (yet )?(have|offer|support)|"
    r"(is|are) not (yet |currently )?(available|possible|supported)|not possible|currently (no|not)\b)",
    re.IGNORECASE)


def existence_claim(reply_en: str) -> str | None:
    m = EXISTENCE_CLAIM.search(reply_en or "")
    return m.group(0) if m else None


PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]


def raise_priority(current: str, minimum: str) -> str:
    return min(current, minimum, key=PRIORITY_ORDER.index)
