"""Export a calibrated sklearn text model in a verified portable JSON contract."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import sklearn

from commandclassifier.task_definition import TaskDefinition

FORMAT_VERSION = 3


def calibration_bundle(model: Any) -> dict[str, Any]:
    """Serialize the selected calibration mapping without Python object state."""
    calibrator = model.calibration_model
    if model.calibration_method == "sigmoid":
        return {
            "method": "sigmoid",
            "coefficient": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
        }
    return {
        "method": "isotonic",
        "x_thresholds": calibrator.X_thresholds_.tolist(),
        "y_thresholds": calibrator.y_thresholds_.tolist(),
    }


def export_model(
    model: Any,
    destination: Path,
    input_hashes: dict[str, str],
    decision_policy: dict[str, Any],
    task: TaskDefinition,
    model_sha256: str,
) -> None:
    """Write exact feature, probability, provenance, and policy metadata."""
    features = model.base_model.named_steps["features"].transformer_list
    word = dict(features)["word"]
    char = dict(features)["char"]
    classifier = model.base_model.named_steps["model"]
    bundle = {
        "format_version": FORMAT_VERSION,
        "model_version": task.model_version,
        "task": {"id": task.task_id, "version": task.task_version},
        "input": {
            "fields": list(task.input_fields),
            "serialization": {
                "format": "tagged-v1" if len(task.input_fields) > 1 else "text-v1",
                "tag_case": "upper",
                "separator": "\\n\\n",
            },
        },
        "input_sha256": input_hashes,
        "artifact_sha256": {"model.joblib": model_sha256},
        "labels": list(model.classes_),
        "positive_class": task.positive_class,
        "positive_probability_field": "positive_probability",
        "normalization": {"unicode": "NFC", "strip_outer_whitespace": True},
        "word": {
            "analyzer": "word",
            "ngram_range": list(word.ngram_range),
            "lowercase": word.lowercase,
            "token_pattern": word.token_pattern,
            "sublinear_tf": word.sublinear_tf,
            "norm": word.norm,
            "vocabulary": word.vocabulary_,
            "idf": word.idf_.tolist(),
        },
        "char": {
            "analyzer": "char_wb",
            "ngram_range": list(char.ngram_range),
            "lowercase": char.lowercase,
            "sublinear_tf": char.sublinear_tf,
            "norm": char.norm,
            "vocabulary": char.vocabulary_,
            "idf": char.idf_.tolist(),
        },
        "coefficients": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "calibration": calibration_bundle(model),
        "decision_policy": decision_policy,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }
    destination.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
