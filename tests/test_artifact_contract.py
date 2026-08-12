import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from generic_binary_classifier.cli import classify, load_artifact


class FakeModel:
    classes_ = np.array(["safe", "unsafe"])

    def predict_proba(self, texts):
        return np.array([[0.25, 0.75] for _ in texts])


def bundle() -> dict[str, object]:
    return {
        "format_version": 3,
        "model_version": "command-safety-v1",
        "task": {"id": "command-safety", "version": "v1"},
        "input": {"fields": ["text"], "serialization": {"format": "text-v1"}},
        "labels": ["safe", "unsafe"],
        "positive_class": "unsafe",
        "positive_probability_field": "positive_probability",
        "input_sha256": {"development": "0" * 64},
        "artifact_sha256": {},
        "normalization": {"unicode": "NFC"},
        "word": {},
        "char": {},
        "coefficients": [[0.5]],
        "intercept": [0.0],
        "calibration": {"method": "sigmoid"},
        "decision_policy": {
            "positive_probability_threshold": 0.8,
            "review_probability_range": [0.7, 0.9],
        },
    }


def write_artifact(directory: Path) -> None:
    model_path = directory / "model.joblib"
    joblib.dump(FakeModel(), model_path)
    material = bundle()
    material["artifact_sha256"] = {
        "model.joblib": hashlib.sha256(model_path.read_bytes()).hexdigest()
    }
    encoded = json.dumps(material, sort_keys=True).encode()
    (directory / "model.json").write_bytes(encoded)
    (directory / "model.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  model.json\n", encoding="utf-8"
    )


def test_load_artifact_verifies_contract_and_classify_uses_artifact_policy(
    tmp_path: Path,
):
    write_artifact(tmp_path)

    model, contract = load_artifact(tmp_path)
    result = classify(model, "rm -rf demo", contract)

    assert result["label"] == "safe"
    assert result["confidence"] == 0.75
    assert result["review_recommended"] is True


def test_load_artifact_rejects_tampered_python_model(tmp_path: Path):
    write_artifact(tmp_path)
    with (tmp_path / "model.joblib").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="Python model hash mismatch"):
        load_artifact(tmp_path)


def test_load_artifact_rejects_tampered_contract(tmp_path: Path):
    write_artifact(tmp_path)
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_artifact(tmp_path)


def test_load_artifact_rejects_v2_contract(tmp_path: Path):
    write_artifact(tmp_path)
    material = bundle() | {"format_version": 2}
    encoded = json.dumps(material, sort_keys=True).encode()
    (tmp_path / "model.json").write_bytes(encoded)
    (tmp_path / "model.json.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  model.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported"):
        load_artifact(tmp_path)
