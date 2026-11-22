"""Client for the Roosterly helpdesk API.

Handles the helpdesk's deliberate failure modes:
  - 503 service_unavailable: nothing was saved -> retry with exponential backoff + jitter.
  - 429 rate_limited: wait Retry-After seconds, then retry. We also throttle ourselves
    below the 5 req/s limit so this should rarely happen.
  - 409 already_handled: never retried. Raised as AlreadyHandled so the caller can
    record it and move on.
  - Network errors / timeouts: retried. For POSTs this is the one ambiguous case (the
    server may have saved the decision before the connection dropped); a retry then
    returns 409, which the caller sees as AlreadyHandled. No customer gets two answers.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import Counter
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv

load_dotenv()

MAX_REQUESTS_PER_SECOND = 4  # API limit is 5; stay under it
MAX_ATTEMPTS = 6
BASE_BACKOFF_S = 0.5
MAX_BACKOFF_S = 10.0


class HelpdeskError(Exception):
    def __init__(self, status: int, error: str, body: Any = None):
        super().__init__(f"{status} {error}: {body}")
        self.status = status
        self.error = error
        self.body = body


class AlreadyHandled(HelpdeskError):
    """409: the ticket already has a decision. Never retry."""


class NotFound(HelpdeskError):
    """404: unknown endpoint or ticket not in the current set."""


class ValidationFailed(HelpdeskError):
    """422: a field is missing or invalid. body['problems'] says what."""


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable {name} (see .env.example)")
    return value


class HelpdeskClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        email: str | None = None,
        name: str | None = None,
        timeout_s: float = 20.0,
    ):
        base_url = (base_url or _env("HELPDESK_BASE_URL")).rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key or _env('HELPDESK_API_KEY')}",
            "X-Candidate-Email": email or _env("CANDIDATE_EMAIL"),
        }
        name = name or os.environ.get("CANDIDATE_NAME", "").strip()
        if name:
            headers["X-Candidate-Name"] = name
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s)
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self._min_interval = 1.0 / MAX_REQUESTS_PER_SECOND
        # Evidence for the run summary: how often the helpdesk's deliberate failures were absorbed.
        self.stats: Counter = Counter()

    # ---- low level -------------------------------------------------------

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request_at + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> float:
        # Full jitter: random value in [0, min(cap, base * 2^attempt)]
        return random.uniform(0, min(MAX_BACKOFF_S, BASE_BACKOFF_S * 2**attempt))

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self._throttle()
            self.stats["requests"] += 1
            try:
                resp = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:  # timeout, connection reset, DNS...
                self.stats["network_error_retried"] += 1
                last_exc = exc
                time.sleep(self._backoff(attempt))
                continue

            if resp.status_code < 400:
                return resp.json() if resp.content else None

            try:
                body = resp.json()
            except ValueError:
                body = {"error": resp.text[:200]}
            error = body.get("error", "unknown") if isinstance(body, dict) else "unknown"

            if resp.status_code == 503:
                self.stats["503_retried"] += 1
                last_exc = HelpdeskError(503, error, body)
                time.sleep(self._backoff(attempt))
                continue
            if resp.status_code == 429:
                self.stats["429_waited"] += 1
                last_exc = HelpdeskError(429, error, body)
                try:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                except ValueError:
                    retry_after = 1.0
                time.sleep(retry_after + random.uniform(0, 0.25))
                continue
            if resp.status_code == 409:
                self.stats["409_already_handled"] += 1
                raise AlreadyHandled(409, error, body)
            if resp.status_code == 404:
                raise NotFound(404, error, body)
            if resp.status_code == 422:
                raise ValidationFailed(422, error, body)
            # 400 / 401 / anything else: not retryable, it's our bug or config
            raise HelpdeskError(resp.status_code, error, body)

        raise HelpdeskError(0, "retries_exhausted", repr(last_exc))

    # ---- endpoints -------------------------------------------------------

    def status(self) -> dict:
        return self._request("GET", "/status")

    def examples(self) -> Any:
        return self._request("GET", "/examples")

    def iter_tickets(self, status: str = "open", page_size: int = 10) -> Iterator[dict]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": page_size, "status": status}
            if cursor:
                params["cursor"] = cursor
            page = self._request("GET", "/tickets", params=params)
            for ticket in page.get("tickets", page.get("data", [])):
                yield ticket
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def get_ticket(self, ticket_id: str) -> dict:
        return self._request("GET", f"/tickets/{ticket_id}")

    def reply(self, ticket_id: str, decision: dict) -> dict:
        return self._request("POST", f"/tickets/{ticket_id}/reply", json=decision)

    def handoff(self, ticket_id: str, decision: dict) -> dict:
        return self._request("POST", f"/tickets/{ticket_id}/handoff", json=decision)

    def close(self, ticket_id: str, decision: dict) -> dict:
        return self._request("POST", f"/tickets/{ticket_id}/close", json=decision)

    def reset(self) -> dict:
        return self._request("POST", "/practice/reset")

    def close_client(self) -> None:
        self._http.close()
