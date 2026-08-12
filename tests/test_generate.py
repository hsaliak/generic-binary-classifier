from pathlib import Path

import pytest

from commandclassifier.generate import GenerationError, generation_command


def write_task(tmp_path: Path, backend: str) -> Path:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Emit records.", encoding="utf-8")
    manifest = tmp_path / "task.yaml"
    manifest.write_text(
        f"""task_id: fixture
task_version: v1
input: {{fields: [text]}}
labels: {{values: [allow, block], positive_class: block}}
data:
  prompt: {prompt}
  raw_directory: {tmp_path / "raw"}
  split_group_fields: [family]
generation: {{backend: {backend}, model: test-model}}
model: {{random_seed: 1}}
evaluation: {{outer_folds: 2, inner_folds: 2}}
decision_policy:
  positive_probability_threshold: 0.5
  review_probability_range: [0.2, 0.9]
""",
        encoding="utf-8",
    )
    return manifest


def test_claude_command_uses_noninteractive_print_mode(tmp_path: Path):
    manifest = write_task(tmp_path, "claude")
    command, request, output = generation_command(
        manifest, "claude", "002", "boundaries"
    )

    assert command == ["claude", "-p", request]
    assert "Batch ID: 002" in request
    assert output.name == "claude-batch-002.raw"


def test_std_slop_command_uses_manifest_model(tmp_path: Path):
    manifest = write_task(tmp_path, "std_slop")

    command, request, output = generation_command(
        manifest, "std_slop", "001", "boundaries"
    )

    assert command == [
        "std_slop",
        "--model",
        "test-model",
        "--prompt",
        request,
        "--output",
        "json",
    ]
    assert "Batch ID: 001" in request
    assert output.name == "std_slop-batch-001.raw"


def test_std_slop_model_flag_overrides_manifest_model(tmp_path: Path):
    manifest = write_task(tmp_path, "std_slop")

    command, _, _ = generation_command(
        manifest, "std_slop", "001", "boundaries", model="deepseek-v4-flash-latest:high"
    )

    assert command[2] == "deepseek-v4-flash-latest:high"


def test_std_slop_rejects_empty_resolved_model(tmp_path: Path):
    manifest = write_task(tmp_path, "std_slop")
    text = manifest.read_text(encoding="utf-8").replace("model: test-model", "model: ")
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(GenerationError, match="std_slop model must be non-empty"):
        generation_command(manifest, "std_slop", "001", "boundaries")


def test_generation_rejects_backend_not_configured_by_task(tmp_path: Path):
    manifest = write_task(tmp_path, "std_slop")

    with pytest.raises(GenerationError, match="does not match"):
        generation_command(manifest, "claude", "002", "boundaries")
