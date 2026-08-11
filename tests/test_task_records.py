import json
from pathlib import Path

import pytest

from commandclassifier.task_definition import load_task_definition
from commandclassifier.task_records import read_task_jsonl, serialize_record
from commandclassifier.validate_data import RecordValidationError


def task():
    return load_task_definition(Path("tasks/command-safety-v1.yaml"))


def test_command_safety_top_level_text_serializes_through_task_contract():
    assert serialize_record({"text": " git status "}, task()) == "git status"


def test_task_reader_supports_explicit_multiple_inputs(tmp_path: Path):
    manifest = tmp_path / "task.yaml"
    manifest.write_text(
        """task_id: example
task_version: v1
input: {fields: [context, request]}
labels: {values: ['no', 'yes'], positive_class: 'yes'}
data: {split_group_fields: [family]}
decision_policy:
  positive_probability_threshold: 0.5
  review_probability_range: [0.2, 0.9]
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "one",
                "label": "yes",
                "inputs": {"context": "Repo", "request": "Add test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_task_jsonl(dataset, load_task_definition(manifest))

    assert records[0]["_serialized_input"] == "<CONTEXT>\nRepo\n\n<REQUEST>\nAdd test"


def test_task_reader_rejects_duplicate_serialized_inputs(tmp_path: Path):
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": record_id, "label": "safe", "text": " git status "})
            for record_id in ("one", "two")
        ),
        encoding="utf-8",
    )

    with pytest.raises(RecordValidationError, match="duplicate normalized task input"):
        read_task_jsonl(dataset, task())
