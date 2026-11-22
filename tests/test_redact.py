import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redact import redact, redact_obj  # noqa: E402


def test_card_cvc_expiry_masked():
    text = "nieuwe creditcard: 4111 1111 1111 1111, geldig tot 09/28, CVC 123. Groet"
    out, found = redact(text)
    assert found
    assert "4111" not in out and "123" not in out and "09/28" not in out
    assert "[CARD REDACTED]" in out and "Groet" in out


def test_card_with_dashes_and_no_spaces():
    for card in ["4111-1111-1111-1111", "4111111111111111", "3782 822463 10005"]:
        out, found = redact(f"kaart {card} graag")
        assert found and card not in out


def test_iban_masked():
    for iban in ["NL91ABNA0417164300", "NL91 ABNA 0417 1643 00", "DE89 3704 0044 0532 0130 00"]:
        out, found = redact(f"Ons IBAN is {iban}.")
        assert found and "[IBAN REDACTED]" in out and "0417" not in out and "3704" not in out


def test_normal_text_untouched():
    for text in [
        "Een dienst van 22:00 tot 02:00 komt eruit als 8 uur.",
        "Er is voor oktober twee keer €312 afgeschreven.",
        "Wat kost Plus voor 25 medewerkers? Factuur van 12/10.",
        "Our staff at the Lisbon location see clock-in times one hour ahead.",
    ]:
        assert redact(text) == (text, False)


def test_redact_obj_nested():
    obj = {"note": "card 4111 1111 1111 1111", "n": 3, "list": ["NL91ABNA0417164300"]}
    out = redact_obj(obj)
    assert "4111" not in out["note"] and out["n"] == 3 and "ABNA" not in out["list"][0]
