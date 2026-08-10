"""Reusable calibrated text-model types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_SEED = 20250221


def pipeline() -> Pipeline:
    """Return the uncalibrated text feature and linear-classifier pipeline."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
                ),
            ),
        ]
    )


def unsafe_scores(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """Return the decision score for the explicitly named unsafe class."""
    if list(model.classes_) != ["safe", "unsafe"]:
        raise ValueError("model classes must be exactly ['safe', 'unsafe']")
    return np.asarray(model.decision_function(texts), dtype=float)


@dataclass
class CalibratedTextClassifier:
    """A TF-IDF model with a calibration mapping learned from grouped OOF scores."""

    base_model: Pipeline
    calibration_method: str
    calibration_model: LogisticRegression | IsotonicRegression
    classes_: np.ndarray

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        scores = unsafe_scores(self.base_model, texts)
        if self.calibration_method == "sigmoid":
            unsafe = self.calibration_model.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            unsafe = self.calibration_model.predict(scores)
        unsafe = np.clip(unsafe, 0.0, 1.0)
        return np.column_stack((1.0 - unsafe, unsafe))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.where(self.predict_proba(texts)[:, 1] >= 0.5, "unsafe", "safe")
