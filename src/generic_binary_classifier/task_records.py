"""Task-configured record loading and canonical model-input serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generic_binary_classifier.task_definition import (
    TaskDefinition,
    TaskDefinitionError,
    serialize_inputs,
)
from generic_binary_classifier.validate_data import (
    RecordValidationError,
    normalize_text,
)


def record_inputs(record: dict[str, Any], task: TaskDefinition) -> dict[str, str]:
    """Read configured inputs from an explicit mapping or legacy top-level fields."""
    source = record.get("inputs", record)
    if not isinstance(source, dict):
        raise RecordValidationError("inputs must be an object")
    try:
        inputs = {field: source[field] for field in task.input_fields}
        return {field: normalize_text(value) for field, value in inputs.items()}
    except (KeyError, TypeError, TaskDefinitionError) as error:
        raise RecordValidationError(f"invalid task inputs: {error}") from error


def serialize_record(record: dict[str, Any], task: TaskDefinition) -> str:
    """Return the one normalized text value consumed by the vectorizers."""
    return serialize_inputs(record_inputs(record, task), task)


def read_task_jsonl(path: Path, task: TaskDefinition) -> list[dict[str, Any]]:
    """Read generic records and reject duplicate canonical inputs or invalid labels."""
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    inputs: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RecordValidationError("record must be an object")
                record_id = record["id"]
                label = record["label"]
                if not isinstance(record_id, str) or not record_id:
                    raise RecordValidationError("id must be non-empty text")
                if label not in task.labels:
                    raise RecordValidationError(
                        "label is not configured by task definition"
                    )
                canonical = serialize_record(record, task)
            except (json.JSONDecodeError, KeyError, RecordValidationError) as error:
                raise RecordValidationError(f"{path}:{line_number}: {error}") from error
            if record_id in ids:
                raise RecordValidationError(
                    f"{path}:{line_number}: duplicate id {record_id}"
                )
            if canonical in inputs:
                raise RecordValidationError(
                    f"{path}:{line_number}: duplicate normalized task input"
                )
            normalized = dict(record)
            normalized["_serialized_input"] = canonical
            records.append(normalized)
            ids.add(record_id)
            inputs.add(canonical)
    if not records:
        raise RecordValidationError(f"{path}: no records")
    return records
