import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_positive_first_label_order_python_go_parity(tmp_path: Path):
    """Regression: positive class sorting first must flip the calibration score
    in Go exactly as Python does (see positive_scores in model.py)."""
    fixture = Path("tests/fixtures/generic-positive-first")
    artifact_dir = tmp_path / "artifact"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "commandclassifier.train",
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
    bundle = json.loads((artifact_dir / "model.json").read_text(encoding="utf-8"))
    assert bundle["labels"] == ["high_complexity", "low_complexity"]
    assert bundle["positive_class"] == "high_complexity"

    prompts = ["plan theta complex multi step", "inspect theta simple fast"]
    for prompt in prompts:
        python_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "commandclassifier.cli",
                "--artifact-dir",
                str(artifact_dir),
                "--text",
                prompt,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        go_result = subprocess.run(
            [
                "go",
                "run",
                "./cmd/command-classify",
                "--model",
                str(artifact_dir / "model.json"),
                "--text",
                prompt,
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
        assert python_output["label"] == (
            "high_complexity" if "complex" in prompt else "low_complexity"
        )
