from pathlib import Path

import pytest

from generic_binary_classifier.merge_data import merge_batches
from generic_binary_classifier.validate_data import (
    RecordValidationError,
    load_record_contract,
    write_jsonl,
)

CONTRACT = load_record_contract(Path("tasks/command-safety-v1.yaml"))


def record(record_id: str, text: str, label: str):
    return {
        "id": record_id,
        "text": text,
        "label": label,
        "family": "test",
        "platform": ["linux"],
        "shell": "bash",
        "risk_reasons": [],
        "source": "test",
        "generator": "test",
        "prompt_version": "v1",
        "batch_id": "original",
        "context_required": False,
    }


def test_merge_deduplicates_same_label_and_rebases_identifiers(tmp_path: Path):
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write_jsonl([record("one", "pwd", "safe")], first)
    write_jsonl([record("two", " pwd ", "safe")], second)

    merged = merge_batches([first, second], CONTRACT)

    assert len(merged) == 1
    assert merged[0]["id"] == "one:one"
    assert merged[0]["batch_id"] == "one"


def test_merge_rejects_conflicting_duplicate_labels(tmp_path: Path):
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write_jsonl([record("one", "pwd", "safe")], first)
    write_jsonl([record("two", "pwd", "unsafe")], second)

    with pytest.raises(RecordValidationError, match="conflicting labels"):
        merge_batches([first, second], CONTRACT)
