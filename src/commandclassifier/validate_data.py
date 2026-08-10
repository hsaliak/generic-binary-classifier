"""Dataset record validation and deterministic normalization."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LABELS = frozenset({"safe", "unsafe"})
PLATFORMS = frozenset({"linux", "macos"})
SHELLS = frozenset({"bash", "zsh", "sh", "other"})
REQUIRED_FIELDS = frozenset(
    {
        "id",
        "text",
        "label",
        "family",
        "platform",
        "shell",
        "risk_reasons",
        "source",
        "generator",
        "prompt_version",
        "batch_id",
        "context_required",
    }
)


class RecordValidationError(ValueError):
    """A record does not satisfy the classification record contract."""


def normalize_text(text: str) -> str:
    """Return the comparison form while preserving command semantics.

    Unicode normalization and trimming remove generator noise. Interior whitespace
    is intentionally preserved because it can change shell quoting and arguments.
    """
    return unicodedata.normalize("NFKC", text).strip()


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field} must be a non-empty string")
    return value


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one record and return a normalized copy."""
    if not isinstance(record, dict):
        raise RecordValidationError("record must be a JSON object")

    unexpected = set(record) - REQUIRED_FIELDS
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise RecordValidationError(f"missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise RecordValidationError(
            f"unexpected fields: {', '.join(sorted(unexpected))}"
        )

    normalized = dict(record)
    for field in (
        "id",
        "text",
        "label",
        "family",
        "shell",
        "source",
        "generator",
        "prompt_version",
        "batch_id",
    ):
        normalized[field] = _require_string(record, field)

    normalized["text"] = normalize_text(normalized["text"])
    if not normalized["text"]:
        raise RecordValidationError("text must contain non-whitespace characters")
    if normalized["label"] not in LABELS:
        raise RecordValidationError("label must be safe or unsafe")
    if normalized["shell"] not in SHELLS:
        raise RecordValidationError(f"shell must be one of {sorted(SHELLS)}")

    platform = record.get("platform")
    if (
        not isinstance(platform, list)
        or not platform
        or any(item not in PLATFORMS for item in platform)
    ):
        raise RecordValidationError("platform must be a non-empty list of linux/macos")
    normalized["platform"] = sorted(set(platform))

    reasons = record.get("risk_reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in reasons
    ):
        raise RecordValidationError("risk_reasons must be a list of non-empty strings")
    normalized["risk_reasons"] = reasons

    if not isinstance(record.get("context_required"), bool):
        raise RecordValidationError("context_required must be boolean")
    return normalized


@dataclass(frozen=True)
class DatasetSummary:
    records: int
    labels: dict[str, int]
    families: dict[str, int]
    platforms: dict[str, int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read, validate, and reject duplicate record IDs and command text."""
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    texts: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = validate_record(raw)
            except (json.JSONDecodeError, RecordValidationError) as error:
                raise RecordValidationError(f"{path}:{line_number}: {error}") from error
            if record["id"] in ids:
                raise RecordValidationError(
                    f"{path}:{line_number}: duplicate id {record['id']}"
                )
            if record["text"] in texts:
                message = f"{path}:{line_number}: duplicate normalized text"
                raise RecordValidationError(f"{message} {record['text']!r}")
            ids.add(record["id"])
            texts.add(record["text"])
            records.append(record)
    if not records:
        raise RecordValidationError(f"{path}: no records")
    return records


def summarize(records: Iterable[dict[str, Any]]) -> DatasetSummary:
    records = list(records)
    labels = Counter(record["label"] for record in records)
    families = Counter(record["family"] for record in records)
    platforms = Counter(
        platform for record in records for platform in record["platform"]
    )
    return DatasetSummary(
        records=len(records),
        labels=dict(sorted(labels.items())),
        families=dict(sorted(families.items())),
        platforms=dict(sorted(platforms.items())),
    )


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write canonical JSONL in deterministic key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate command-classifier JSONL data."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="Optional normalized JSONL destination."
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.output:
        write_jsonl(records, args.output)
    print(json.dumps(summarize(records).__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
