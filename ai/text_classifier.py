"""Classical NLP classification for bounded InvestWise dialogue questions.

This module provides probabilistic text classification only.

It does not make financial decisions and it does not decide whether a
prediction is safe to accept. Conformal prediction and dialogue policy
are handled by separate modules.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TextPrediction:
    """Probabilistic output from a bounded text classifier."""

    predicted_class: str
    probabilities: dict[str, float]


class BoundedTextClassifier:
    """TF IDF and logistic regression classifier for bounded answer sets."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> "BoundedTextClassifier":
        """Train the classifier on labelled free text answers."""

        if len(texts) != len(labels):
            raise ValueError("texts and labels must contain the same number of items.")

        if len(texts) == 0:
            raise ValueError("training data cannot be empty.")

        if len(set(labels)) < 2:
            raise ValueError("training data must contain at least two classes.")

        self._pipeline.fit(texts, labels)
        self._is_fitted = True

        return self

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the fitted class labels."""

        self._require_fitted()

        classifier = self._pipeline.named_steps["classifier"]
        return tuple(str(label) for label in classifier.classes_)

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return probabilities for every class."""

        self._require_fitted()

        probabilities = self._pipeline.predict_proba([text])[0]

        return {
            label: float(probability)
            for label, probability in zip(self.classes, probabilities)
        }

    def predict(self, text: str) -> TextPrediction:
        """Return the ordinary argmax prediction and class probabilities.

        The argmax is diagnostic only. Downstream code must not treat it
        as safe when conformal prediction returns multiple plausible classes.
        """

        probabilities = self.predict_proba(text)

        predicted_class = max(
            probabilities,
            key=probabilities.get,
        )

        return TextPrediction(
            predicted_class=predicted_class,
            probabilities=probabilities,
        )

    def predict_probability_matrix(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Return the probability matrix used during conformal calibration."""

        self._require_fitted()

        if len(texts) == 0:
            raise ValueError("texts cannot be empty.")

        return self._pipeline.predict_proba(texts)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("classifier must be fitted before prediction.")
