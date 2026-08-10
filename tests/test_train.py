import hashlib
import json
from pathlib import Path

import pytest

from commandclassifier.train import (
    grouped_splits,
    groups_for,
    train_and_report,
    verify_evaluation_manifest,
)


def record(index: int, label: str, group: str) -> dict[str, object]:
    return {
        "id": f"record-{index}",
        "text": f"{'danger' if label == 'unsafe' else 'inspect'} command {index}",
        "label": label,
        "family": group,
        "platform": ["linux" if index % 2 else "macos"],
        "shell": "bash",
        "risk_reasons": [],
        "source": "test",
        "generator": "test",
        "prompt_version": "v1",
        "batch_id": f"batch-{group}",
        "context_required": False,
    }


def config() -> dict[str, object]:
    return {
        "split_group_fields": ["family", "batch_id"],
        "model": {"random_seed": 7},
        "evaluation": {"outer_folds": 3, "inner_folds": 2},
        "decision_policy": {
            "unsafe_probability_threshold": 0.5,
            "review_probability_range": [0.2, 0.9],
        },
    }


def test_evaluation_manifest_must_be_reviewed_and_hash_bound(tmp_path: Path):
    evaluation = tmp_path / "evaluation.jsonl"
    evaluation.write_text("record\n", encoding="utf-8")
    manifest = {
        "dataset": evaluation.name,
        "dataset_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "review_status": "reviewed",
    }
    evaluation.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert verify_evaluation_manifest(evaluation).name == "evaluation.manifest.json"

    manifest["review_status"] = "pending"
    evaluation.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not reviewed"):
        verify_evaluation_manifest(evaluation)


def test_grouped_splits_never_leak_composite_groups():
    # Build a balanced set: each of six groups has two safe and two unsafe records.
    records = [
        record(group * 4 + offset, label, f"group-{group}")
        for group in range(6)
        for offset, label in enumerate(("safe", "unsafe", "safe", "unsafe"))
    ]
    groups = groups_for(records, ["family", "batch_id"])
    splits = grouped_splits(
        [item["label"] for item in records], groups, folds=3, seed=7
    )

    for train_indexes, test_indexes in splits:
        train_groups = {groups[index] for index in train_indexes}
        test_groups = {groups[index] for index in test_indexes}
        assert train_groups.isdisjoint(test_groups)


def test_nested_training_reports_calibration_and_policy_metrics():
    training = [
        record(group * 4 + offset, label, f"group-{group}")
        for group in range(6)
        for offset, label in enumerate(("safe", "unsafe", "safe", "unsafe"))
    ]
    evaluation = [
        record(100, "safe", "evaluation-safe"),
        record(101, "unsafe", "evaluation-unsafe"),
    ]

    model, report = train_and_report(training, evaluation, config())

    assert list(model.classes_) == ["safe", "unsafe"]
    assert report["development_nested_cv"]["threshold_table"]
    assert 0 <= report["development_nested_cv"]["brier_score"] <= 1
    assert report["locked_evaluation"]["breakdown"]["platform"]
    assert report["policy"]["unsafe_probability_threshold"] == 0.5
