import json
from pathlib import Path

import pytest

from commandclassifier.validate_data import (
    RecordValidationError,
    normalize_text,
    read_jsonl,
    validate_record,
)


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


def test_validate_record_normalizes_text_and_platforms():
    actual = validate_record(valid_record(platform=["macos", "linux", "linux"]))

    assert actual["text"] == "ls -la"
    assert actual["platform"] == ["linux", "macos"]


def test_normalize_text_preserves_meaningful_internal_whitespace():
    assert normalize_text("  printf 'a  b'  ") == "printf 'a  b'"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"label": "unsure"}, "label must be safe or unsafe"),
        ({"platform": ["windows"]}, "platform must be a non-empty list"),
        ({"context_required": "false"}, "context_required must be boolean"),
        ({"risk_reasons": ["", "deletes_files"]}, "risk_reasons must be a list"),
    ],
)
def test_validate_record_rejects_invalid_contract(overrides, message):
    with pytest.raises(RecordValidationError, match=message):
        validate_record(valid_record(**overrides))


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
        read_jsonl(dataset)
