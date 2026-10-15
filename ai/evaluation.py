"""Evaluation utilities for bounded InvestWise classifiers.

Conventional classification metrics and conformal selective prediction
metrics are deliberately reported separately.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ai.conformal import SplitConformalClassifier
from ai.text_classifier import BoundedTextClassifier


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    labels: tuple[str, ...]
    confusion_matrix: np.ndarray


@dataclass(frozen=True)
class ConformalMetrics:
    empirical_coverage: float
    average_set_size: float
    singleton_rate: float
    multi_class_rate: float
    empty_set_rate: float
    abstention_rate: float
    accepted_prediction_accuracy: float | None


def evaluate_classifier(
    classifier: BoundedTextClassifier,
    texts: list[str],
    labels: list[str],
) -> ClassificationMetrics:
    """Evaluate ordinary argmax classification."""

    predictions = [
        classifier.predict(text).predicted_class
        for text in texts
    ]

    ordered_labels = classifier.classes

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=ordered_labels,
    )

    return ClassificationMetrics(
        accuracy=float(
            accuracy_score(labels, predictions)
        ),
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
        labels=ordered_labels,
        confusion_matrix=matrix,
    )


def evaluate_conformal(
    predictor: SplitConformalClassifier,
    texts: list[str],
    labels: list[str],
) -> ConformalMetrics:
    """Evaluate prediction-set behaviour on unseen labelled examples."""

    results = [
        predictor.predict(text)
        for text in texts
    ]

    set_sizes = [
        len(result.prediction_set)
        for result in results
    ]

    covered = [
        true_label in result.prediction_set
        for true_label, result in zip(labels, results)
    ]

    singleton_indices = [
        index
        for index, size in enumerate(set_sizes)
        if size == 1
    ]

    singleton_correct = []

    for index in singleton_indices:
        predicted = next(
            iter(results[index].prediction_set)
        )

        singleton_correct.append(
            predicted == labels[index]
        )

    accepted_accuracy = (
        float(np.mean(singleton_correct))
        if singleton_correct
        else None
    )

    n = len(results)

    singleton_rate = sum(
        size == 1
        for size in set_sizes
    ) / n

    multi_class_rate = sum(
        size > 1
        for size in set_sizes
    ) / n

    empty_set_rate = sum(
        size == 0
        for size in set_sizes
    ) / n

    # InvestWise accepts only singleton prediction sets.
    abstention_rate = 1.0 - singleton_rate

    return ConformalMetrics(
        empirical_coverage=float(np.mean(covered)),
        average_set_size=float(np.mean(set_sizes)),
        singleton_rate=float(singleton_rate),
        multi_class_rate=float(multi_class_rate),
        empty_set_rate=float(empty_set_rate),
        abstention_rate=float(abstention_rate),
        accepted_prediction_accuracy=accepted_accuracy,
    )
