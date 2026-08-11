"""Generate raw candidate records through a task-configured LLM backend."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from commandclassifier.task_definition import load_task_definition


class GenerationError(ValueError):
    """A generation request cannot be constructed from its task definition."""


def generation_command(
    manifest: Path, backend: str, batch: str, focus: str
) -> tuple[list[str], str, Path]:
    """Build a non-executing backend command and raw-output destination."""
    load_task_definition(manifest)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    configured_backend = raw["generation"].get("backend")
    if backend not in {"std_slop", "claude"}:
        raise GenerationError("backend must be std_slop or claude")
    if backend != configured_backend:
        raise GenerationError("backend does not match task generation.backend")
    prompt_path = Path(raw["data"]["prompt"])
    if not prompt_path.is_file():
        raise GenerationError(f"task prompt does not exist: {prompt_path}")
    if not batch.strip() or not focus.strip():
        raise GenerationError("batch and focus must be non-empty")
    request = (
        f"{prompt_path.read_text(encoding='utf-8').strip()}\n\n"
        f"Batch ID: {batch}. Focus: {focus.strip()}. "
        "Generate distinct records only and emit the configured JSONL record shape."
    )
    output = Path(raw["data"]["raw_directory"]) / f"{backend}-batch-{batch}.raw"
    if backend == "std_slop":
        model = raw["generation"].get("model")
        if not isinstance(model, str) or not model:
            raise GenerationError("std_slop generation.model must be non-empty")
        return (
            ["std_slop", "--model", model, "--prompt", request, "--output", "json"],
            request,
            output,
        )
    return ["claude", "-p", request], request, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unreviewed task candidates.")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--focus", required=True)
    args = parser.parse_args()
    try:
        command, _, output = generation_command(
            args.task, args.backend, args.batch, args.focus
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        output.write_text(completed.stdout, encoding="utf-8")
    except (GenerationError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"generation failed: {error}") from error
    print(json.dumps({"backend": args.backend, "raw_output": str(output)}))


if __name__ == "__main__":
    main()
