"""Validated configuration for a generic calibrated binary classification task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TaskDefinitionError(ValueError):
    """A task definition cannot be used to train or infer."""


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    task_version: str
    input_fields: tuple[str, ...]
    labels: tuple[str, str]
    positive_class: str
    split_group_fields: tuple[str, ...]
    decision_threshold: float
    review_probability_range: tuple[float, float]

    @property
    def model_version(self) -> str:
        return f"{self.task_id}-{self.task_version}"


def load_task_definition(path: Path) -> TaskDefinition:
    """Load the binary-task fields required by the generic framework."""
    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        inputs = raw["input"]["fields"]
        labels = raw["labels"]["values"]
        positive = raw["labels"]["positive_class"]
        policy = raw["decision_policy"]
        definition = TaskDefinition(
            task_id=raw["task_id"],
            task_version=raw["task_version"],
            input_fields=tuple(inputs),
            labels=tuple(labels),
            positive_class=positive,
            split_group_fields=tuple(raw["data"]["split_group_fields"]),
            decision_threshold=float(policy["positive_probability_threshold"]),
            review_probability_range=tuple(
                map(float, policy["review_probability_range"])
            ),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise TaskDefinitionError(f"invalid task definition {path}: {error}") from error
    if not definition.task_id or not definition.task_version:
        raise TaskDefinitionError("task_id and task_version must be non-empty")
    if not definition.input_fields or any(
        not field for field in definition.input_fields
    ):
        raise TaskDefinitionError("input.fields must contain non-empty names")
    if len(set(definition.input_fields)) != len(definition.input_fields):
        raise TaskDefinitionError("input.fields must be unique")
    if len(definition.labels) != 2 or len(set(definition.labels)) != 2:
        raise TaskDefinitionError(
            "labels.values must contain exactly two unique labels"
        )
    if definition.positive_class not in definition.labels:
        raise TaskDefinitionError("labels.positive_class must be a configured label")
    if not definition.split_group_fields:
        raise TaskDefinitionError("data.split_group_fields must not be empty")
    low, high = definition.review_probability_range
    if not 0 <= low <= high <= 1 or not 0 <= definition.decision_threshold <= 1:
        raise TaskDefinitionError("decision policy probabilities must be in [0, 1]")
    return definition


def serialize_inputs(inputs: dict[str, str], task: TaskDefinition) -> str:
    """Create the stable model input for a task-defined field set."""
    if set(inputs) != set(task.input_fields):
        raise TaskDefinitionError("input fields do not match task definition")
    values = []
    for field in task.input_fields:
        value = inputs[field]
        if not isinstance(value, str) or not value.strip():
            raise TaskDefinitionError(f"input {field!r} must be non-empty text")
        values.append(value.strip())
    if len(values) == 1:
        return values[0]
    return "\n\n".join(
        f"<{field.upper()}>\n{value}"
        for field, value in zip(task.input_fields, values, strict=True)
    )
