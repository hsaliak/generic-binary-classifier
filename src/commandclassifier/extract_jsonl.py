"""Extract only valid JSON objects from an LLM JSON response."""

from __future__ import annotations

import json
import sys


def main() -> None:
    response = json.load(sys.stdin)
    message = response["assistant_message"]
    records = []
    rejected = []
    for line in message.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rejected.append(line)
            continue
        if not isinstance(value, dict):
            raise SystemExit("assistant message contains non-object JSON")
        records.append(value)
    if not records:
        raise SystemExit("assistant response has no JSONL records")
    if rejected:
        print(f"rejected {len(rejected)} non-JSON lines", file=sys.stderr)
    for record in records:
        print(json.dumps(record, separators=(",", ":")))


if __name__ == "__main__":
    main()
