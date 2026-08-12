import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_multifield_fixture_trains_and_exports_v3(tmp_path: Path):
    fixture = Path("tests/fixtures/generic-multifield")
    artifact_dir = tmp_path / "artifact"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "generic_binary_classifier.train",
            "--manifest",
            str(fixture / "task.yaml"),
            "--input",
            str(fixture / "development.jsonl"),
            "--evaluation",
            str(fixture / "evaluation.jsonl"),
            "--artifact-dir",
            str(artifact_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"format_version": 3' in (artifact_dir / "model.json").read_text(
        encoding="utf-8"
    )
    inputs = json.dumps(
        {"user_context": "review theta 1", "assistant_text": "allow inspect theta 1"}
    )
    python_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "generic_binary_classifier.cli",
            "--artifact-dir",
            str(artifact_dir),
            "--input-json",
            inputs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    go_result = subprocess.run(
        [
            "go",
            "run",
            "./cmd/generic-binclass",
            "--model",
            str(artifact_dir / "model.json"),
            "--input-json",
            inputs,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd="go",
    )
    python_output = json.loads(python_result.stdout)
    go_output = json.loads(go_result.stdout)
    assert python_output["label"] == go_output["label"]
    assert python_output["positive_probability"] == pytest.approx(
        go_output["positive_probability"]
    )
