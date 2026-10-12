"""Split conformal prediction for bounded InvestWise classifiers.

The conformal layer quantifies model uncertainty.

It does not interpret financial safety, customer contradictions, or
vulnerability. Those responsibilities remain with deterministic policy
and financial governance layers.
"""

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np

from ai.text_classifier import BoundedTextClassifier


@dataclass(frozen=True)
class ConformalPrediction:
    """Prediction set produced by a calibrated conformal predictor."""

    prediction_set: frozenset[str]
    probabilities: dict[str, float]
    threshold: float
    alpha: float

    @property
    def is_singleton(self) -> bool:
        """Whether exactly one class remains plausible."""

        return len(self.prediction_set) == 1

    @property
    def is_empty(self) -> bool:
        """Whether no class satisfies the conformal threshold."""

        return len(self.prediction_set) == 0

    @property
    def is_model_uncertain(self) -> bool:
        """Whether the model cannot safely reduce the result to one class."""

        return len(self.prediction_set) != 1


class SplitConformalClassifier:
    """Split conformal wrapper around a fitted probabilistic classifier.

    Nonconformity score:

        1 - P(true_class | x)

    Calibration data must be separate from classifier training data.
    """

    def __init__(
        self,
        classifier: BoundedTextClassifier,
        alpha: float = 0.10,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        self.classifier = classifier
        self.alpha = alpha
        self._threshold: float | None = None

    @property
    def threshold(self) -> float:
        """Return the calibrated nonconformity threshold."""

        if self._threshold is None:
            raise RuntimeError(
                "conformal predictor must be calibrated before prediction."
            )

        return self._threshold

    def calibrate(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> "SplitConformalClassifier":
        """Calibrate using data not used to train the base classifier."""

        if len(texts) != len(labels):
            raise ValueError(
                "calibration texts and labels must contain the same number of items."
            )

        if len(texts) == 0:
            raise ValueError("calibration data cannot be empty.")

        classes = self.classifier.classes
        class_to_index = {
            label: index
            for index, label in enumerate(classes)
        }

        unknown_labels = set(labels) - set(classes)

        if unknown_labels:
            raise ValueError(
                f"calibration contains unknown labels: {sorted(unknown_labels)}"
            )

        probability_matrix = self.classifier.predict_probability_matrix(texts)

        scores = np.asarray(
            [
                1.0 - probability_matrix[row_index, class_to_index[label]]
                for row_index, label in enumerate(labels)
            ],
            dtype=float,
        )

        n = len(scores)

        # Finite sample split conformal quantile.
        rank = ceil((n + 1) * (1 - self.alpha))
        rank = min(rank, n)

        sorted_scores = np.sort(scores)

        self._threshold = float(sorted_scores[rank - 1])

        return self

    def predict(self, text: str) -> ConformalPrediction:
        """Generate a conformal prediction set for one free text answer."""

        threshold = self.threshold
        probabilities = self.classifier.predict_proba(text)

        prediction_set = frozenset(
            label
            for label, probability in probabilities.items()
            if (1.0 - probability) <= threshold
        )

        return ConformalPrediction(
            prediction_set=prediction_set,
            probabilities=probabilities,
            threshold=threshold,
            alpha=self.alpha,
        )
