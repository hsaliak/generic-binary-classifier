import json
from pathlib import Path

import pytest

from generic_binary_classifier.validate_data import (
    RecordValidationError,
    load_record_contract,
    normalize_text,
    read_jsonl,
    validate_record,
)

CONTRACT = load_record_contract(Path("tasks/command-safety-v1.yaml"))


def valid_record(**overrides):
    record = {
        "id": "record-1",
        "text": "  ls -la  ",
        "label": "safe",
        "family": "read_only_navigation",
        "platform": ["linux", "macos"],
        "shell": "bash",
        "risk_reasons": [],
        "source": "synthetic",
        "generator": "test",
        "prompt_version": "v1",
        "batch_id": "batch-1",
        "context_required": False,
    }
    record.update(overrides)
    return record


def test_contract_derives_command_safety_record_shape():
    assert CONTRACT.labels == frozenset({"safe", "unsafe"})
    assert "risk_reasons" in CONTRACT.array_string_fields
    assert "shell" in CONTRACT.string_enum_fields
    assert "platform" in CONTRACT.array_enum_fields
    assert "context_required" in CONTRACT.boolean_fields


def test_validate_record_normalizes_text_and_platforms():
    actual = validate_record(
        valid_record(platform=["macos", "linux", "linux"]), CONTRACT
    )

    assert actual["text"] == "ls -la"
    assert actual["platform"] == ["linux", "macos"]


def test_normalize_text_preserves_meaningful_internal_whitespace():
    assert normalize_text("  printf 'a  b'  ") == "printf 'a  b'"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"label": "unsure"}, "label must be one of"),
        ({"platform": ["windows"]}, "platform must be a non-empty list"),
        ({"context_required": "false"}, "context_required must be boolean"),
        ({"risk_reasons": ["", "deletes_files"]}, "risk_reasons must be a list"),
        ({"unexpected_field": "x"}, "unexpected fields"),
    ],
)
def test_validate_record_rejects_invalid_contract(overrides, message):
    with pytest.raises(RecordValidationError, match=message):
        validate_record(valid_record(**overrides), CONTRACT)


def test_read_jsonl_rejects_normalized_duplicates(tmp_path: Path):
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(valid_record(id="one", text=" pwd ")),
                json.dumps(valid_record(id="two", text="pwd")),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RecordValidationError, match="duplicate normalized text"):
        read_jsonl(dataset, CONTRACT)
