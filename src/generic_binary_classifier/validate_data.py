"""Manifest-driven dataset record validation and deterministic normalization."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class RecordValidationError(ValueError):
    """A record does not satisfy its task record contract."""


@dataclass(frozen=True)
class RecordContract:
    """Record shape derived from a task manifest and its JSON Schema file."""

    labels: frozenset[str]
    input_fields: tuple[str, ...]
    required_string_fields: frozenset[str]
    string_enum_fields: dict[str, frozenset[str]]
    array_enum_fields: dict[str, frozenset[str]]
    array_string_fields: frozenset[str]
    boolean_fields: frozenset[str]

    @property
    def required_fields(self) -> frozenset[str]:
        return (
            self.required_string_fields
            | frozenset(self.string_enum_fields)
            | frozenset(self.array_enum_fields)
            | self.array_string_fields
            | self.boolean_fields
        )

    @property
    def allowed_fields(self) -> frozenset[str]:
        return self.required_fields


def load_record_contract(manifest: Path) -> RecordContract:
    """Derive the record contract from the manifest labels and record schema."""
    try:
        raw: dict[str, Any] = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        labels = frozenset(raw["labels"]["values"])
        input_fields = tuple(raw["input"]["fields"])
        schema = json.loads(
            Path(raw["data"]["record_schema"]).read_text(encoding="utf-8")
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as e:
        raise RecordValidationError(f"cannot load record contract for {manifest}: {e}")

    plain_strings: set[str] = set()
    string_enums: dict[str, frozenset[str]] = {}
    array_enums: dict[str, frozenset[str]] = {}
    array_strings: set[str] = set()
    booleans: set[str] = set()

    for name, spec in schema.get("properties", {}).items():
        kind = spec.get("type")
        if kind == "string":
            allowed = spec.get("enum") or []
            if allowed:
                string_enums[name] = frozenset(allowed)
            else:
                plain_strings.add(name)
        elif kind == "boolean":
            booleans.add(name)
        elif kind == "array":
            items = spec.get("items", {})
            item_enums = items.get("enum") if isinstance(items, dict) else None
            if item_enums:
                array_enums[name] = frozenset(item_enums)
            else:
                array_strings.add(name)
    return RecordContract(
        labels=labels,
        input_fields=input_fields,
        required_string_fields=frozenset(plain_strings),
        string_enum_fields=string_enums,
        array_enum_fields=array_enums,
        array_string_fields=frozenset(array_strings),
        boolean_fields=frozenset(booleans),
    )


def normalize_text(text: str) -> str:
    """Return the comparison form while preserving meaningful interior structure.

    Unicode normalization and trimming remove generator noise. Interior whitespace
    is intentionally preserved because it can change prompt and command semantics.
    """
    return unicodedata.normalize("NFKC", text).strip()


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field} must be a non-empty string")
    return normalize_text(value)


def validate_record(record: dict[str, Any], contract: RecordContract) -> dict[str, Any]:
    """Validate one record against the task contract and return a normalized copy."""
    if not isinstance(record, dict):
        raise RecordValidationError("record must be a JSON object")

    missing = contract.required_fields - set(record)
    unexpected = set(record) - contract.allowed_fields
    if missing:
        raise RecordValidationError(f"missing fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise RecordValidationError(
            f"unexpected fields: {', '.join(sorted(unexpected))}"
        )

    normalized = dict(record)
    for field in contract.required_string_fields:
        normalized[field] = _require_string(record, field)

    for field, allowed in contract.string_enum_fields.items():
        value = _require_string(record, field)
        if value not in allowed:
            raise RecordValidationError(f"{field} must be one of {sorted(allowed)}")
        normalized[field] = value

    label = record.get("label")
    if not isinstance(label, str) or label not in contract.labels:
        raise RecordValidationError(f"label must be one of {sorted(contract.labels)}")
    normalized["label"] = label

    for field, allowed in contract.array_enum_fields.items():
        values = record.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(item not in allowed for item in values)
        ):
            raise RecordValidationError(
                f"{field} must be a non-empty list of {sorted(allowed)}"
            )
        normalized[field] = sorted(set(values))

    for field in contract.array_string_fields:
        values = record.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise RecordValidationError(f"{field} must be a list of non-empty strings")
        normalized[field] = values

    for field in contract.boolean_fields:
        if not isinstance(record.get(field), bool):
            raise RecordValidationError(f"{field} must be boolean")
    return normalized


def read_jsonl(path: Path, contract: RecordContract) -> list[dict[str, Any]]:
    """Read, validate, and reject duplicate record IDs and canonical inputs."""
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    texts: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = validate_record(raw, contract)
            except (json.JSONDecodeError, RecordValidationError) as error:
                raise RecordValidationError(f"{path}:{line_number}: {error}") from error
            if record["id"] in ids:
                raise RecordValidationError(
                    f"{path}:{line_number}: duplicate id {record['id']}"
                )
            text = record.get("text")
            if isinstance(text, str):
                if text in texts:
                    message = f"{path}:{line_number}: duplicate normalized text"
                    raise RecordValidationError(f"{message} {text!r}")
                texts.add(text)
            ids.add(record["id"])
            records.append(record)
    if not records:
        raise RecordValidationError(f"{path}: no records")
    return records


@dataclass(frozen=True)
class DatasetSummary:
    records: int
    labels: dict[str, int]
    families: dict[str, int]
    platforms: dict[str, int]


def summarize(
    records: Iterable[dict[str, Any]], contract: RecordContract
) -> DatasetSummary:
    records = list(records)
    labels = Counter(record["label"] for record in records)
    families = Counter(record["family"] for record in records)
    platforms = Counter(
        platform
        for record in records
        if "platform" in record
        for platform in record["platform"]
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
        description="Validate task JSONL data against its record contract."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="Optional normalized JSONL destination."
    )
    args = parser.parse_args()
    contract = load_record_contract(args.manifest)
    records = read_jsonl(args.input, contract)
    if args.output:
        write_jsonl(records, args.output)
    print(json.dumps(summarize(records, contract).__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
