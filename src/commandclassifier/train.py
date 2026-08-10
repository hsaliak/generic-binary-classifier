"""Train and evaluate a grouped, calibrated command-safety classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from commandclassifier.corpus_report import build_report
from commandclassifier.export_model import export_model
from commandclassifier.model import RANDOM_SEED, CalibratedTextClassifier, pipeline
from commandclassifier.validate_data import read_jsonl

CALIBRATION_METHODS = ("sigmoid", "isotonic")


def fit_calibrated(
    texts: Sequence[str],
    labels: Sequence[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    method: str,
) -> CalibratedTextClassifier:
    """Fit calibration only to grouped out-of-fold decision scores."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"unsupported calibration method: {method}")
    labels_array = np.asarray(labels)
    oof_scores = cross_val_predict(
        pipeline(), texts, labels_array, cv=cv_splits, method="decision_function"
    )
    targets = (labels_array == "unsafe").astype(int)
    if method == "sigmoid":
        calibrator: LogisticRegression | IsotonicRegression = LogisticRegression(
            random_state=RANDOM_SEED
        ).fit(np.asarray(oof_scores).reshape(-1, 1), targets)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_scores, targets)
    base = pipeline().fit(texts, labels_array)
    return CalibratedTextClassifier(base, method, calibrator, base.classes_)


def grouped_splits(
    labels: Sequence[str], groups: Sequence[str], folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic stratified grouped folds or fail with context."""
    try:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(labels)), labels, groups))
    except ValueError as error:
        raise ValueError(
            f"cannot create {folds} grouped stratified folds: {error}"
        ) from error


def groups_for(records: Sequence[dict[str, Any]], fields: Sequence[str]) -> list[str]:
    """Create a composite split group from all manifest-required group fields."""
    return ["\x1f".join(str(record[field]) for field in fields) for record in records]


def select_calibration(
    texts: Sequence[str],
    labels: Sequence[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> str:
    """Select a mapping by held-out grouped Brier score, not fit-set score."""
    labels_array = np.asarray(labels)
    targets = (labels_array == "unsafe").astype(int)
    raw_scores = cross_val_predict(
        pipeline(), texts, labels_array, cv=splits, method="decision_function"
    )
    candidates: list[tuple[float, str]] = []
    for method in CALIBRATION_METHODS:
        probabilities = np.zeros(len(texts))
        for calibration_train, calibration_test in splits:
            if method == "sigmoid":
                calibrator: LogisticRegression | IsotonicRegression = (
                    LogisticRegression(random_state=RANDOM_SEED).fit(
                        raw_scores[calibration_train].reshape(-1, 1),
                        targets[calibration_train],
                    )
                )
                probabilities[calibration_test] = calibrator.predict_proba(
                    raw_scores[calibration_test].reshape(-1, 1)
                )[:, 1]
            else:
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    raw_scores[calibration_train], targets[calibration_train]
                )
                probabilities[calibration_test] = calibrator.predict(
                    raw_scores[calibration_test]
                )
        candidates.append((brier_score_loss(targets, probabilities), method))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def threshold_table(
    labels: Sequence[str], probabilities: np.ndarray
) -> list[dict[str, float]]:
    """Report policy trade-offs without tuning the locked evaluation dataset."""
    targets = np.asarray(labels) == "unsafe"
    table = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & targets))
        false_negative = int(np.sum(~predicted & targets))
        false_positive = int(np.sum(predicted & ~targets))
        unsafe_count = int(np.sum(targets))
        table.append(
            {
                "threshold": round(float(threshold), 2),
                "unsafe_recall": true_positive / unsafe_count if unsafe_count else 0.0,
                "unsafe_false_negative_rate": false_negative / unsafe_count
                if unsafe_count
                else 0.0,
                "unsafe_precision": true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0,
            }
        )
    return table


def breakdown(
    records: Sequence[dict[str, Any]], probabilities: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    """Calculate support and unsafe recall by family and platform."""
    result: dict[str, dict[str, dict[str, float]]] = {"family": {}, "platform": {}}
    for dimension in ("family", "platform"):
        values = sorted(
            {
                value
                for record in records
                for value in (
                    record[dimension]
                    if dimension == "platform"
                    else [record[dimension]]
                )
            }
        )
        for value in values:
            indexes = [
                index
                for index, record in enumerate(records)
                if value
                in (
                    record[dimension]
                    if dimension == "platform"
                    else [record[dimension]]
                )
            ]
            targets = np.asarray(
                [records[index]["label"] == "unsafe" for index in indexes]
            )
            predicted = probabilities[indexes] >= 0.5
            unsafe_count = int(np.sum(targets))
            result[dimension][value] = {
                "support": len(indexes),
                "unsafe_support": unsafe_count,
                "unsafe_recall": float(np.sum(predicted & targets) / unsafe_count)
                if unsafe_count
                else 0.0,
            }
    return result


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evaluation_manifest(evaluation: Path) -> Path:
    """Require a reviewed manifest that cryptographically binds the locked set."""
    manifest = evaluation.with_suffix(".manifest.json")
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evaluation manifest: {error}") from error
    if metadata.get("dataset") != evaluation.name:
        raise ValueError("evaluation manifest dataset name does not match input")
    if metadata.get("dataset_sha256") != hash_file(evaluation):
        raise ValueError("evaluation manifest hash does not match input")
    if metadata.get("review_status") != "reviewed":
        raise ValueError("evaluation manifest is not reviewed")
    return manifest


def train_and_report(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[CalibratedTextClassifier, dict[str, Any]]:
    """Run nested grouped development evaluation, then fit the release candidate."""
    fields = config["split_group_fields"]
    outer_folds = config["evaluation"]["outer_folds"]
    inner_folds = config["evaluation"]["inner_folds"]
    seed = config["model"]["random_seed"]
    texts = [record["text"] for record in train]
    labels = [record["label"] for record in train]
    groups = groups_for(train, fields)
    outer_splits = grouped_splits(labels, groups, outer_folds, seed)
    oof_probability = np.zeros(len(train))
    methods: list[str] = []
    for outer_train, outer_test in outer_splits:
        inner_texts = [texts[index] for index in outer_train]
        inner_labels = [labels[index] for index in outer_train]
        inner_groups = [groups[index] for index in outer_train]
        inner_splits = grouped_splits(inner_labels, inner_groups, inner_folds, seed)
        method = select_calibration(inner_texts, inner_labels, inner_splits)
        methods.append(method)
        model = fit_calibrated(inner_texts, inner_labels, inner_splits, method)
        oof_probability[outer_test] = model.predict_proba(
            [texts[index] for index in outer_test]
        )[:, 1]
    final_splits = grouped_splits(labels, groups, inner_folds, seed)
    final_method = select_calibration(texts, labels, final_splits)
    final_model = fit_calibrated(texts, labels, final_splits, final_method)
    evaluation_probability = final_model.predict_proba(
        [record["text"] for record in evaluation]
    )[:, 1]
    predicted = np.where(oof_probability >= 0.5, "unsafe", "safe")
    report = {
        "format_version": 1,
        "development_nested_cv": {
            "classification": classification_report(
                labels, predicted, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                labels, predicted, labels=["safe", "unsafe"]
            ).tolist(),
            "brier_score": brier_score_loss(
                np.asarray(labels) == "unsafe", oof_probability
            ),
            "calibration_curve": {
                "observed_unsafe_fraction": calibration_curve(
                    np.asarray(labels) == "unsafe", oof_probability, n_bins=10
                )[0].tolist(),
                "mean_predicted_probability": calibration_curve(
                    np.asarray(labels) == "unsafe", oof_probability, n_bins=10
                )[1].tolist(),
            },
            "threshold_table": threshold_table(labels, oof_probability),
            "calibration_methods_by_outer_fold": methods,
        },
        "locked_evaluation": {
            "classification": classification_report(
                [record["label"] for record in evaluation],
                np.where(evaluation_probability >= 0.5, "unsafe", "safe"),
                output_dict=True,
                zero_division=0,
            ),
            "brier_score": brier_score_loss(
                np.asarray([record["label"] for record in evaluation]) == "unsafe",
                evaluation_probability,
            ),
            "breakdown": breakdown(evaluation, evaluation_probability),
        },
        "policy": config["decision_policy"],
        "selected_final_calibration": final_method,
        "records": {"development": len(train), "evaluation": len(evaluation)},
    }
    return final_model, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    corpus = build_report(args.input, args.evaluation)
    if corpus["exact_text_overlap"]:
        raise SystemExit("development/evaluation overlap detected; training aborted")
    config = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    evaluation_manifest = verify_evaluation_manifest(args.evaluation)
    train = read_jsonl(args.input)
    evaluation = read_jsonl(args.evaluation)
    model, report = train_and_report(train, evaluation, config)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "development": hash_file(args.input),
        "evaluation": hash_file(args.evaluation),
        "manifest": hash_file(args.manifest),
        "evaluation_review_manifest": hash_file(evaluation_manifest),
    }
    report["input_sha256"] = input_hashes
    (args.artifact_dir / "corpus-report.json").write_text(
        json.dumps(corpus, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.artifact_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    model_path = args.artifact_dir / "model.joblib"
    joblib.dump(model, model_path)
    export_model(
        model,
        args.artifact_dir / "model.json",
        input_hashes,
        report["policy"],
        config["task_version"],
        hash_file(model_path),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
