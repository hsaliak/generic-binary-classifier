import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "prompt",
    [
        "continue",
        "proceed",
        "what is the time?",
        "write a function remove_duplicates(items)",
        "PLAN a MIGRATION across 3 services",
        "a  b   c   spaces matter",
        "parse csv then transform then load",
        "say yes/no; check boxes 1,2,3",
    ],
)
def test_python_go_vectorizer_parity_on_varied_ascii(tmp_path: Path, prompt: str):
    """Python and Go must agree on label and probability for varied ASCII prompts."""
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
        go_output["positive_probability"], abs=1e-3
    )
