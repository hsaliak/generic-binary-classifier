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
        "labels",
        "positive_class",
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
    if bundle.get("format_version") != 2 or not REQUIRED_BUNDLE_KEYS <= bundle.keys():
        raise ValueError("unsupported or incomplete artifact contract")
    if bundle["labels"] != ["safe", "unsafe"] or bundle["positive_class"] != "unsafe":
        raise ValueError("artifact class contract is invalid")
    model_path = artifact_dir / "model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != bundle[
        "artifact_sha256"
    ].get("model.joblib"):
        raise ValueError("Python model hash mismatch")
    return joblib.load(model_path), bundle


def classify(
    model: Any, text: str, policy: dict[str, Any] | None = None
) -> dict[str, object]:
    """Classify text using a loaded model without executing the text."""
    policy = policy or {
        "unsafe_probability_threshold": 0.5,
        "review_probability_range": [0.2, 0.9],
    }
    probabilities = model.predict_proba([text])[0]
    scores = dict(zip(model.classes_, map(float, probabilities), strict=True))
    unsafe_probability = scores["unsafe"]
    threshold = float(policy["unsafe_probability_threshold"])
    review_low, review_high = map(float, policy["review_probability_range"])
    return {
        "label": "unsafe" if unsafe_probability >= threshold else "safe",
        "unsafe_probability": unsafe_probability,
        "confidence": unsafe_probability,
        "review_recommended": review_low <= unsafe_probability <= review_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify command text without executing it."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    try:
        model, bundle = load_artifact(args.artifact_dir)
    except ValueError as error:
        parser.error(str(error))
    result = classify(model, args.text, bundle["decision_policy"])
    result["model_version"] = bundle["model_version"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
