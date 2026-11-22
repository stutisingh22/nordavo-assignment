"""One-off exploration: dump /status, /examples and all tickets to data/ for reading.

Read-only: never posts a decision. Output in data/ is gitignored (customer data).
Usage: python scripts/explore.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk import HelpdeskClient  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"


def dump(name: str, obj) -> None:
    path = OUT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"wrote {path.relative_to(OUT.parent)}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    hd = HelpdeskClient()
    try:
        status = hd.status()
        print("status:", json.dumps(status, ensure_ascii=False))
        dump("status.json", status)
        dump("examples.json", hd.examples())
        tickets = list(hd.iter_tickets(status="all"))
        print(f"tickets: {len(tickets)}")
        dump("tickets.json", tickets)
    finally:
        hd.close_client()


if __name__ == "__main__":
    main()
