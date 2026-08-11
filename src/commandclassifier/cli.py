"""Non-executing command safety inference CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

REQUIRED_BUNDLE_KEYS = frozenset(
    {
        "format_version",
        "model_version",
        "task",
        "input",
        "labels",
        "positive_class",
        "positive_probability_field",
        "input_sha256",
        "artifact_sha256",
        "normalization",
        "word",
        "char",
        "calibration",
        "decision_policy",
    }
)


def load_artifact(artifact_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Verify the portable contract before loading the Python inference model."""
    bundle_path = artifact_dir / "model.json"
    digest_path = artifact_dir / "model.json.sha256"
    try:
        expected = digest_path.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, IndexError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid artifact contract: {error}") from error
    if actual != expected:
        raise ValueError("artifact contract hash mismatch")
    if bundle.get("format_version") != 3 or not REQUIRED_BUNDLE_KEYS <= bundle.keys():
        raise ValueError("unsupported or incomplete artifact contract")
    fields = bundle["input"].get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(field, str) and field for field in fields)
        or len(set(fields)) != len(fields)
        or len(bundle["labels"]) != 2
        or bundle["positive_class"] not in bundle["labels"]
    ):
        raise ValueError("artifact class or input contract is invalid")
    model_path = artifact_dir / "model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != bundle[
        "artifact_sha256"
    ].get("model.joblib"):
        raise ValueError("Python model hash mismatch")
    return joblib.load(model_path), bundle


def classify(model: Any, text: str, bundle: dict[str, Any]) -> dict[str, object]:
    """Classify serialized task input without executing any supplied text."""
    probabilities = model.predict_proba([text])[0]
    scores = dict(zip(model.classes_, map(float, probabilities), strict=True))
    positive_class = bundle["positive_class"]
    positive_probability = scores[positive_class]
    threshold = float(bundle["decision_policy"]["positive_probability_threshold"])
    review_low, review_high = map(
        float, bundle["decision_policy"]["review_probability_range"]
    )
    negative_class = next(
        label for label in bundle["labels"] if label != positive_class
    )
    return {
        "label": positive_class
        if positive_probability >= threshold
        else negative_class,
        bundle["positive_probability_field"]: positive_probability,
        "confidence": positive_probability,
        "review_recommended": review_low <= positive_probability <= review_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify command text without executing it."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--text")
    parser.add_argument("--input-json")
    args = parser.parse_args()
    if bool(args.text) == bool(args.input_json):
        parser.error("provide exactly one of --text or --input-json")
    try:
        model, bundle = load_artifact(args.artifact_dir)
    except ValueError as error:
        parser.error(str(error))
    fields = bundle["input"]["fields"]
    if args.text is not None:
        if len(fields) != 1 or not args.text.strip():
            parser.error("--text requires a one-field artifact and non-empty text")
        text = args.text.strip()
    else:
        try:
            inputs = json.loads(args.input_json)
        except json.JSONDecodeError as error:
            parser.error(f"invalid --input-json: {error.msg}")
        if not isinstance(inputs, dict) or set(inputs) != set(fields):
            parser.error("--input-json fields do not match artifact")
        if not all(
            isinstance(value, str) and value.strip() for value in inputs.values()
        ):
            parser.error("--input-json values must be non-empty text")
        text = (
            "\\n\\n".join(
                f"<{field.upper()}>\\n{inputs[field].strip()}" for field in fields
            )
            if len(fields) > 1
            else inputs[fields[0]].strip()
        )
    result = classify(model, text, bundle)
    result["model_version"] = bundle["model_version"]
    result["task"] = bundle["task"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
