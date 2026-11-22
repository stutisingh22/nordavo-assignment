import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guardrails as g  # noqa: E402


def test_always_human_rules_fire():
    cases = {
        "Kunnen we die maand terugkrijgen?": "refund_or_disputed_charge",
        "Willen jullie één betaling terugstorten?": "refund_or_disputed_charge",
        "We willen ons abonnement opzeggen.": "cancellation_or_contract_change",
        "Als niemand mij belt, zeggen we op.": "threat_to_leave",
        "inzage in mijn persoonsgegevens (AVG)": "personal_data_request",
        "Iemand heeft ingelogd vanuit Roemenië. Dat waren wij niet.": "security_incident",
        "Onze advocaat neemt contact op over een schadevergoeding.": "legal_threat",
    }
    for text, rule in cases.items():
        assert rule in g.always_human_hits(text), text


def test_plain_faq_questions_do_not_fire():
    for text in [
        "Hoe ruil ik een dienst met een collega?",
        "Hoe voeg ik een nieuwe medewerker toe?",
        "Waar vind ik mijn facturen?",
        "Onze ondernemingsraad vraagt waar onze gegevens worden opgeslagen.",
        "Hoe upgrade ik naar Plus en wanneer gaat de nieuwe prijs in?",
        "Kunt u zeggen waar ik dat op kan vinden?",
        "Can a manager cancel a shift?",
    ]:
        assert g.always_human_hits(text) == [], text


def test_detect_language():
    assert g.detect_language("Hoe kan ik een dienst ruilen met mijn collega? Groet")[0] == "nl"
    assert g.detect_language("When I click Publish I get an error and the rota is not visible")[0] == "en"
    assert g.detect_language("Wie können meine Mitarbeiter untereinander Schichten tauschen? Gibt es das auch")[0] == "de"
    assert g.detect_language("Re:")[0] is None


def test_reply_problems():
    ok = "Hoi Anouk, ga naar Team → Medewerker toevoegen. Goedgekeurde uren zie je op app.roosterly.nl/login."
    assert g.reply_problems(ok) == []
    assert any("forbidden_promise" in p for p in g.reply_problems("Je krijgt 50% korting op de volgende factuur."))
    assert any("unknown_link" in p for p in g.reply_problems("Zie help.roosterly.nl"))
    assert "external_link" in g.reply_problems("Kijk op https://example.com")
    assert "placeholder_or_redaction_in_reply" in g.reply_problems("Hoi [naam],")
    assert g.reply_problems("Viele Grüße, Team Roosterly. Preis: 6 € → Plus") == []
    assert any("garbled_characters" in p for p in g.reply_problems("Viele GrüÔe, Team Roosterly"))


def test_raise_priority_never_lowers():
    assert g.raise_priority("P4", "P2") == "P2"
    assert g.raise_priority("P1", "P2") == "P1"


FAQ_PATHS = {"Instellingen → Abonnement", "Instellingen → Koppelingen → Salaris", "Profiel → Taal",
             "Profile → Language", "Settings → Integrations → Payroll"}


def test_menu_paths_must_exist_in_faq():
    ok = "Ga naar Instellingen → Abonnement kunt u upgraden. Of Profiel → Taal (Profile → Language)."
    assert g.menu_path_problems(ok, FAQ_PATHS) == []
    # Invented / shortened path seen in a dry run (T016)
    bad = "manage API access under Instellingen → Koppelingen (Settings → Integrations)."
    assert len(g.menu_path_problems(bad, FAQ_PATHS)) == 2
    # Model translating menu names itself (German reply)
    assert g.menu_path_problems("unter Profil → Sprache", FAQ_PATHS) == ["unknown_menu_path:Profil → Sprache"]


def test_existence_claim():
    assert g.existence_claim("We currently do not have built-in webhooks for published shifts.")
    assert g.existence_claim("This is not possible yet.")
    assert g.existence_claim("Thank you for your idea! You can vote on roadmap.roosterly.nl.") is None
    assert g.existence_claim("We cannot promise a delivery date for individual requests.") is None


def test_existence_claim_matches_faq_facts_too_so_callers_scope_it():
    # "not supported" is a real FAQ fact for E01 (F12): the agent only applies this check to
    # feature_request replies, so this bug reply is NOT blocked there.
    assert g.existence_claim("Unfortunately iOS 15 is not supported; the app needs iOS 16 or later.")
