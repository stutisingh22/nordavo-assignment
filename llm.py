"""Minimal Gemini client: one JSON-schema call, retries on overload, token accounting.

Called over plain HTTPS (httpx) instead of an SDK, so every request is visible here.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6


class LLMError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # includes "thinking" tokens, which are billed as output
    by_call: list = field(default_factory=list)

    def add(self, meta: dict) -> None:
        i = meta.get("promptTokenCount", 0)
        o = meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        self.calls += 1
        self.input_tokens += i
        self.output_tokens += o
        self.by_call.append((i, o))


def _retry_delay_s(resp: httpx.Response) -> float | None:
    """Gemini puts the wait time in a RetryInfo detail, e.g. "retryDelay": "42s"."""
    try:
        for det in resp.json()["error"].get("details", []):
            if det.get("@type", "").endswith("RetryInfo"):
                return float(det["retryDelay"].rstrip("s"))
    except (ValueError, KeyError, AttributeError):
        pass
    return None


class Gemini:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout_s: float = 90,
                 thinking: str | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "gemini-3.5-flash")
        key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY (see .env.example)")
        self._http = httpx.Client(headers={"x-goog-api-key": key}, timeout=timeout_s)
        self.usage = Usage()
        # Hidden "thinking" tokens are billed as output. "" = model default; else low/medium/high.
        self.thinking = (thinking if thinking is not None else os.environ.get("LLM_THINKING", "")).strip().lower()

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if self.thinking:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking}
        url = API.format(model=self.model)
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self._http.post(url, json=body)
            except httpx.TransportError as exc:
                last = repr(exc)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    self.usage.add(data.get("usageMetadata", {}))
                    try:
                        text = data["candidates"][0]["content"]["parts"][-1]["text"]
                        return json.loads(text)
                    except (KeyError, IndexError, json.JSONDecodeError) as exc:
                        last = f"unparseable response: {exc}"  # retry once more, then give up
                elif resp.status_code in RETRYABLE:
                    last = f"HTTP {resp.status_code}"
                    delay = _retry_delay_s(resp)
                    if delay is not None and delay > 120:
                        # Quota exhausted (e.g. free-tier daily cap): retrying now cannot help.
                        raise LLMError(f"quota exhausted, retry in {delay:.0f}s: {resp.text[:200]}")
                    if delay is not None:
                        time.sleep(delay)
                        continue
                else:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(random.uniform(0, min(30, 1.0 * 2**attempt)))
        raise LLMError(f"gave up after {MAX_ATTEMPTS} attempts: {last}")
