"""Small classical model comparison for InvestWise semantic intent.

This module exists only to compare bounded semantic classifiers.

Models:
Logistic Regression
Calibrated Linear SVM
Multinomial Naive Bayes

Financial decisions remain outside this module.
"""

from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    pipeline: Pipeline


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def build_candidate_models(
    random_state: int = 42,
) -> tuple[ModelSpecification, ...]:
    """Return the deliberately small semantic model shortlist."""

    logistic = Pipeline(
        [
            ("tfidf", _vectorizer()),
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

    svm = Pipeline(
        [
            ("tfidf", _vectorizer()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                    cv=3,
                ),
            ),
        ]
    )

    naive_bayes = Pipeline(
        [
            ("tfidf", _vectorizer()),
            (
                "classifier",
                MultinomialNB(),
            ),
        ]
    )

    return (
        ModelSpecification(
            name="logistic_regression",
            pipeline=logistic,
        ),
        ModelSpecification(
            name="calibrated_linear_svm",
            pipeline=svm,
        ),
        ModelSpecification(
            name="multinomial_naive_bayes",
            pipeline=naive_bayes,
        ),
    )
