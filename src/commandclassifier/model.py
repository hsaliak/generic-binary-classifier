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


def positive_scores(
    model: Pipeline, texts: Sequence[str], positive_class: str
) -> np.ndarray:
    """Return decision scores oriented toward the configured positive class."""
    classes = list(model.classes_)
    if len(classes) != 2 or positive_class not in classes:
        raise ValueError("model classes must contain the configured positive class")
    scores = np.asarray(model.decision_function(texts), dtype=float)
    # sklearn binary decision scores are oriented to classes_[1].
    return scores if classes[1] == positive_class else -scores


@dataclass
class CalibratedTextClassifier:
    """A TF-IDF model with a calibration mapping learned from grouped OOF scores."""

    base_model: Pipeline
    calibration_method: str
    calibration_model: LogisticRegression | IsotonicRegression
    classes_: np.ndarray
    positive_class: str

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        scores = positive_scores(self.base_model, texts, self.positive_class)
        if self.calibration_method == "sigmoid":
            positive = self.calibration_model.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            positive = self.calibration_model.predict(scores)
        positive = np.clip(positive, 0.0, 1.0)
        negative = 1.0 - positive
        return (
            np.column_stack((negative, positive))
            if self.classes_[1] == self.positive_class
            else np.column_stack((positive, negative))
        )

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[np.argmax(probabilities, axis=1)]
