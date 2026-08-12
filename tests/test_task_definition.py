from pathlib import Path

import pytest

from generic_binary_classifier.task_definition import (
    TaskDefinitionError,
    load_task_definition,
    serialize_inputs,
)


def test_load_command_safety_task_definition():
    task = load_task_definition(Path("tasks/command-safety-v1.yaml"))

    assert task.model_version == "command-safety-v1"
    assert task.labels == ("safe", "unsafe")
    assert task.positive_class == "unsafe"


def test_prompt_complexity_task_definition_has_binary_v3_contract():
    task = load_task_definition(Path("tasks/prompt-complexity-v1.yaml"))

    assert task.model_version == "prompt-complexity-v1"
    assert task.labels == ("low_complexity", "high_complexity")
    assert task.positive_class == "high_complexity"
    assert task.input_fields == ("text",)


def test_fixture_multifield_task_has_binary_v3_inputs():
    task = load_task_definition(Path("tests/fixtures/generic-multifield/task.yaml"))

    assert task.labels == ("allow", "block")
    assert task.positive_class == "block"
    assert (
        serialize_inputs(
            {"user_context": "Repository", "assistant_text": "Classify request"}, task
        )
        == "<USER_CONTEXT>\nRepository\n\n<ASSISTANT_TEXT>\nClassify request"
    )


def test_serialize_multiple_fields_is_stable(tmp_path: Path):
    path = tmp_path / "task.yaml"
    path.write_text(
        """task_id: example
task_version: v1
input: {fields: [context, task]}
labels: {values: [no, yes], positive_class: yes}
data: {split_group_fields: [family]}
decision_policy:
  positive_probability_threshold: 0.5
  review_probability_range: [0.2, 0.9]
""",
        encoding="utf-8",
    )
    task = load_task_definition(path)

    assert serialize_inputs({"task": " Work ", "context": " Repo "}, task) == (
        "<CONTEXT>\nRepo\n\n<TASK>\nWork"
    )
    with pytest.raises(TaskDefinitionError, match="input fields"):
        serialize_inputs({"context": "Repo"}, task)
