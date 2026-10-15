"""Question-scoped TF-IDF, logistic regression and conformal interpretation.

The models propose fixed options; they never confirm or save an answer. Model
uncertainty is an internal abstention status, not a fourth semantic class.
Only train.csv and calibration.csv are read here. Locked test data is reserved
for independent evaluation, and the settings below are fixed before that run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import threading
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Callable, Mapping

from ai.conformal import SplitConformalClassifier
from ai.question_spec import BusinessSpec
from ai.text_classifier import BoundedTextClassifier


CONFORMAL_ALPHA = 0.10
INTENT_LABELS = frozenset({"clarity", "undecided", "confusion"})
MIN_LEXICAL_COVERAGE = 0.10


class FiniteSampleConformalClassifier(SplitConformalClassifier):
    """Use the conservative full set when the required rank exceeds n.

    Scores are bounded by one, so threshold=1 includes every class. Clipping
    the rank to the largest observed score would lose the finite-sample
    coverage construction for a calibration set that is too small.
    """

    def calibrate(self, texts, labels) -> "FiniteSampleConformalClassifier":
        super().calibrate(texts, labels)
        if ceil((len(texts) + 1) * (1 - self.alpha)) > len(texts):
            self._threshold = 1.0
        return self


@dataclass(frozen=True)
class LexicalSupport:
    active_features: int
    feature_coverage: float

    @property
    def sufficient(self) -> bool:
        # One-word replies such as "none" and "shares" remain admissible.
        return self.active_features > 0 and self.feature_coverage >= MIN_LEXICAL_COVERAGE


def lexical_support(classifier: BoundedTextClassifier, text: str) -> LexicalSupport:
    """Check vocabulary overlap; this is not a conformal or safety guarantee."""

    vectorizer = classifier._pipeline.named_steps["tfidf"]
    active = int(vectorizer.transform([text]).getnnz())
    possible = len(set(vectorizer.build_analyzer()(text)))
    return LexicalSupport(active, active / possible if possible else 0.0)


@dataclass(frozen=True)
class QuestionPredictors:
    intent: SplitConformalClassifier
    options: SplitConformalClassifier
    intent_support: Callable[[str], LexicalSupport] | None = None
    option_support: Callable[[str], LexicalSupport] | None = None


@dataclass(frozen=True)
class QuestionMLResolution:
    candidate_option_id: str | None = None
    ordinary_bucket: str | None = None
    model_uncertain: bool = False
    semantic_undecided: bool = False
    reason: str | None = None
    source: str = "all_question_ml"
    intent_set: frozenset[str] = frozenset()
    option_set: frozenset[str] = frozenset()
    intent_probabilities: dict[str, float] = field(default_factory=dict)
    option_probabilities: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


class AllQuestionMLResolver:
    """Two independent singleton gates, followed by customer confirmation."""

    def __init__(
        self,
        predictors: Mapping[str, QuestionPredictors],
        allowed_options: Mapping[str, frozenset[str]],
        *,
        source_hash: str = "",
    ) -> None:
        self.predictors = dict(predictors)
        self.allowed_options = dict(allowed_options)
        self.source_hash = source_hash

    def resolve(self, question_id: str, text: str) -> QuestionMLResolution:
        pair = self.predictors.get(question_id)
        trace: dict[str, object] = {
            "question_id": question_id,
            "source_hash": self.source_hash,
            "lexical_guard_is_coverage_guarantee": False,
        }
        result_fields: dict[str, object] = {"diagnostics": trace}

        def abstain(reason: str) -> QuestionMLResolution:
            return QuestionMLResolution(model_uncertain=True, reason=reason, **result_fields)

        if pair is None or question_id not in self.allowed_options:
            return abstain("question_model_not_configured")
        if not text.strip():
            return abstain("empty_reply")
        if pair.intent_support is not None:
            support = pair.intent_support(text)
            trace.update(intent_active_features=support.active_features,
                         intent_lexical_coverage=support.feature_coverage)
            if not support.sufficient:
                return abstain("intent_insufficient_lexical_support")

        intent = pair.intent.predict(text)
        result_fields.update(intent_set=intent.prediction_set,
                             intent_probabilities=dict(intent.probabilities))
        trace.update(intent_threshold=intent.threshold, intent_alpha=intent.alpha)
        if intent.is_empty:
            return abstain("empty_intent_prediction_set")
        if not intent.is_singleton:
            return abstain("intent_model_uncertainty")
        label = next(iter(intent.prediction_set))
        if label == "undecided":
            return QuestionMLResolution(ordinary_bucket="Undecided", semantic_undecided=True,
                                        reason="semantic_undecided", **result_fields)
        if label == "confusion":
            return QuestionMLResolution(ordinary_bucket="Confusion", reason="semantic_confusion",
                                        **result_fields)
        if label != "clarity":
            return abstain("invalid_intent_label")

        if pair.option_support is not None:
            support = pair.option_support(text)
            trace.update(option_active_features=support.active_features,
                         option_lexical_coverage=support.feature_coverage)
            if not support.sufficient:
                return abstain("option_insufficient_lexical_support")
        options = pair.options.predict(text)
        result_fields.update(option_set=options.prediction_set,
                             option_probabilities=dict(options.probabilities))
        trace.update(option_threshold=options.threshold, option_alpha=options.alpha)
        if options.is_empty:
            return abstain("empty_option_prediction_set")
        if not options.is_singleton:
            return abstain("option_model_uncertainty")
        option_id = next(iter(options.prediction_set))
        if option_id not in self.allowed_options[question_id]:
            return abstain("invalid_option_label")
        return QuestionMLResolution(candidate_option_id=option_id, ordinary_bucket="Clarity",
                                    reason="option_identified_requires_confirmation", **result_fields)


_CACHE: dict[tuple[object, ...], AllQuestionMLResolver] = {}
_CACHE_LOCK = threading.Lock()
_REQUIRED_COLUMNS = {"id", "question_id", "text", "bucket", "option_id", "source", "group_id", "split"}


def _read_rows(payload: bytes, split: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if not _REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
        raise ValueError(f"{split}.csv must contain {_REQUIRED_COLUMNS}")
    rows = []
    for raw in reader:
        row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
        if row["split"] != split:
            raise ValueError(f"{split}.csv contains a row assigned to another split")
        row["bucket"] = row["bucket"].lower()
        if not row["text"] or row["bucket"] not in INTENT_LABELS:
            raise ValueError(f"{split}.csv contains an empty reply or invalid bucket")
        if not row["id"] or not row["group_id"] or not row["source"]:
            raise ValueError(f"{split}.csv requires provenance, unique IDs and group IDs")
        if row["option_id"] and row["bucket"] != "clarity":
            raise ValueError("only Clarity examples may have a financial option")
        rows.append(row)
    if not rows:
        raise ValueError(f"{split}.csv cannot be empty")
    return rows


def _fit_predictor(train: list[dict[str, str]], calibration: list[dict[str, str]], target: str):
    classifier = BoundedTextClassifier().fit([r["text"] for r in train], [r[target] for r in train])
    predictor = FiniteSampleConformalClassifier(classifier, alpha=CONFORMAL_ALPHA).calibrate(
        [r["text"] for r in calibration], [r[target] for r in calibration])
    return predictor, lambda text: lexical_support(classifier, text)


def build_all_question_ml_resolver(spec: BusinessSpec, data_directory: str | Path) -> AllQuestionMLResolver:
    """Train once per source hash; never read or select settings from test.csv."""

    directory = Path(data_directory)
    train_bytes = (directory / "train.csv").read_bytes()
    calibration_bytes = (directory / "calibration.csv").read_bytes()
    manifest = tuple((qid, tuple(o.id for o in spec.question(qid).options)) for qid in spec.question_order)
    train_hash = hashlib.sha256(train_bytes).hexdigest()
    calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
    key = (train_hash, calibration_hash, spec.version, manifest, CONFORMAL_ALPHA, MIN_LEXICAL_COVERAGE)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        train = _read_rows(train_bytes, "train")
        calibration = _read_rows(calibration_bytes, "calibration")
        normalise = lambda r: (r["question_id"], " ".join(r["text"].lower().split()))
        if {normalise(r) for r in train} & {normalise(r) for r in calibration}:
            raise ValueError("training and calibration replies overlap")
        if {r["group_id"] for r in train} & {r["group_id"] for r in calibration}:
            raise ValueError("training and calibration paraphrase groups overlap")
        ids = [r["id"] for r in train + calibration]
        if len(ids) != len(set(ids)):
            raise ValueError("training and calibration IDs must be unique")
        known_questions = set(spec.question_order)
        if any(r["question_id"] not in known_questions for r in train + calibration):
            raise ValueError("dataset contains an unknown question ID")

        predictors = {}
        allowed = {}
        for qid, option_ids in manifest:
            allowed[qid] = frozenset(option_ids)
            qtrain = [r for r in train if r["question_id"] == qid]
            qcalibration = [r for r in calibration if r["question_id"] == qid]
            if {r["bucket"] for r in qtrain} != INTENT_LABELS or {r["bucket"] for r in qcalibration} != INTENT_LABELS:
                raise ValueError(f"{qid} requires all three intent classes in train and calibration")
            option_train = [r for r in qtrain if r["option_id"]]
            option_calibration = [r for r in qcalibration if r["option_id"]]
            if {r["option_id"] for r in option_train} != allowed[qid] or {r["option_id"] for r in option_calibration} != allowed[qid]:
                raise ValueError(f"{qid} requires every fixed option in train and calibration")
            intent, intent_support = _fit_predictor(qtrain, qcalibration, "bucket")
            options, option_support = _fit_predictor(option_train, option_calibration, "option_id")
            predictors[qid] = QuestionPredictors(intent, options, intent_support, option_support)

        source_hash = hashlib.sha256((train_hash + calibration_hash + repr(manifest)).encode()).hexdigest()
        resolver = AllQuestionMLResolver(predictors, allowed, source_hash=source_hash)
        _CACHE[key] = resolver
        return resolver
