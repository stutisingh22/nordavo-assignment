"""Mask payment details before ticket text goes anywhere (LLM, logs, journal, handoff note).

Client rule: never repeat card numbers or IBANs in a reply or a log. We over-mask on
purpose: hiding a harmless 16-digit number is fine, leaking a card is not.
"""

from __future__ import annotations

import re

# 13-19 digits, optionally grouped by spaces or dashes (Visa, Mastercard, Amex, ...)
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# IBAN: country code, 2 check digits, 11-30 alphanumerics, optionally in groups of 4
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.IGNORECASE)
# CVC / CVV / security code followed by 3-4 digits
CVC = re.compile(r"\b(cvc|cvv|cvc2|cvv2|beveiligingscode|security code)\b\s*:?\s*\d{3,4}", re.IGNORECASE)
# Expiry MM/YY or MM/YYYY (only masked when the text also had a card number)
EXPIRY = re.compile(r"\b(0[1-9]|1[0-2])\s?/\s?(\d{2}|\d{4})\b")


def redact(text: str) -> tuple[str, bool]:
    """Return (masked text, whether anything payment-related was found)."""
    if not text:
        return text, False
    found = False

    def sub(pattern: re.Pattern, repl, s: str) -> str:
        nonlocal found
        new, n = pattern.subn(repl, s)
        found = found or n > 0
        return new

    had_card = bool(CARD.search(text))
    text = sub(IBAN, "[IBAN REDACTED]", text)
    text = sub(CARD, "[CARD REDACTED]", text)
    text = sub(CVC, lambda m: f"{m.group(1)} [REDACTED]", text)
    if had_card:
        text = sub(EXPIRY, "[EXPIRY REDACTED]", text)
    return text, found


def redact_obj(obj):
    """Redact every string inside a dict/list (for log lines and outgoing decisions)."""
    if isinstance(obj, str):
        return redact(obj)[0]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj
