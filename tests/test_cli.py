import numpy as np

from commandclassifier.cli import classify

BUNDLE = {
    "labels": ["safe", "unsafe"],
    "positive_class": "unsafe",
    "positive_probability_field": "positive_probability",
    "decision_policy": {
        "positive_probability_threshold": 0.5,
        "review_probability_range": [0.2, 0.9],
    },
}


class FakeModel:
    classes_ = np.array(["safe", "unsafe"])

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, texts):
        assert len(texts) == 1
        return np.array([self.probabilities])


def test_classify_returns_safe_for_low_unsafe_probability():
    result = classify(FakeModel([0.95, 0.05]), "pwd", BUNDLE)

    assert result == {
        "label": "safe",
        "positive_probability": 0.05,
        "confidence": 0.05,
        "review_recommended": False,
    }


def test_classify_recommends_review_for_ambiguous_probability():
    result = classify(FakeModel([0.45, 0.55]), 'rm "$TARGET"', BUNDLE)

    assert result["label"] == "unsafe"
    assert result["review_recommended"] is True
