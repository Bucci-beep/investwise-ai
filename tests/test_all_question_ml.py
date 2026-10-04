"""Acceptance gates independent of model accuracy or authored demo wording."""

import csv

import numpy as np
import pytest

from ai.all_question_ml import (
    AllQuestionMLResolver,
    FiniteSampleConformalClassifier,
    LexicalSupport,
    QuestionPredictors,
    build_all_question_ml_resolver,
)
from ai.conformal import ConformalPrediction
from ai.question_spec import BusinessSpec


class StubPredictor:
    def __init__(self, labels):
        self.labels = frozenset(labels)
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        # An extremely high argmax must never override a non-singleton set.
        return ConformalPrediction(self.labels, {"clarity": .99, "other": .01}, .6, .1)


def resolver(intent, options, **kwargs):
    pair = QuestionPredictors(StubPredictor(intent), StubPredictor(options), **kwargs)
    return AllQuestionMLResolver({"D8": pair}, {"D8": frozenset({"D8_NONE", "D8_SHARES"})})


@pytest.mark.parametrize("labels", [(), ("clarity", "undecided"), ("clarity", "confusion")])
def test_intent_abstention_never_calls_option_classifier_or_uses_argmax(labels):
    ml = resolver(labels, {"D8_NONE"})
    result = ml.resolve("D8", "an answer")
    assert result.model_uncertain
    assert result.ordinary_bucket is None
    assert result.candidate_option_id is None
    assert ml.predictors["D8"].options.calls == 0


@pytest.mark.parametrize("label,bucket", [("undecided", "Undecided"), ("confusion", "Confusion")])
def test_known_nonclear_semantics_do_not_propose_financial_option(label, bucket):
    ml = resolver({label}, {"D8_SHARES"})
    result = ml.resolve("D8", "an answer")
    assert result.ordinary_bucket == bucket
    assert result.candidate_option_id is None
    assert not result.model_uncertain
    assert result.semantic_undecided == (label == "undecided")
    assert ml.predictors["D8"].options.calls == 0


@pytest.mark.parametrize("labels", [(), ("D8_NONE", "D8_SHARES"), ("D12_SHORT",)])
def test_clarity_alone_cannot_accept_ambiguous_empty_or_cross_question_options(labels):
    result = resolver({"clarity"}, labels).resolve("D8", "an answer")
    assert result.model_uncertain
    assert result.ordinary_bucket is None
    assert result.candidate_option_id is None
    assert result.intent_set == frozenset({"clarity"})


def test_double_singleton_creates_traceable_candidate_only():
    result = resolver({"clarity"}, {"D8_NONE"}).resolve("D8", "none")
    assert result.ordinary_bucket == "Clarity"
    assert result.candidate_option_id == "D8_NONE"
    assert result.source == "all_question_ml"
    assert result.reason == "option_identified_requires_confirmation"
    assert result.intent_set == frozenset({"clarity"})
    assert result.option_set == frozenset({"D8_NONE"})
    assert result.diagnostics["intent_alpha"] == .1
    assert not hasattr(result, "recorded_value")


def test_known_one_word_is_allowed_but_oov_and_sparse_support_abstain():
    known = lambda text: LexicalSupport(1, 1.0)
    assert resolver({"clarity"}, {"D8_NONE"}, intent_support=known,
                    option_support=known).resolve("D8", "none").candidate_option_id == "D8_NONE"
    for support in (LexicalSupport(0, 0), LexicalSupport(1, .05)):
        ml = resolver({"clarity"}, {"D8_NONE"}, intent_support=lambda text: support)
        result = ml.resolve("D8", "unsupported wording")
        assert result.model_uncertain and result.candidate_option_id is None
        assert ml.predictors["D8"].intent.calls == 0


def test_empty_reply_and_missing_model_fail_closed():
    ml = resolver({"clarity"}, {"D8_NONE"})
    assert ml.resolve("D8", " ").reason == "empty_reply"
    assert ml.resolve("D99", "none").reason == "question_model_not_configured"


class ProbabilityStub:
    classes = ("clarity", "undecided", "confusion")

    def predict_probability_matrix(self, texts):
        return np.tile([.8, .1, .1], (len(texts), 1))

    def predict_proba(self, text):
        return dict(zip(self.classes, [.8, .1, .1]))


def test_tiny_calibration_uses_full_set_instead_of_clipping_rank():
    predictor = FiniteSampleConformalClassifier(ProbabilityStub(), alpha=.1)
    predictor.calibrate(["one"] * 3, ["clarity"] * 3)
    assert predictor.threshold == 1.0
    assert predictor.predict("answer").prediction_set == frozenset(ProbabilityStub.classes)


def test_sufficient_calibration_uses_correct_order_statistic():
    predictor = FiniteSampleConformalClassifier(ProbabilityStub(), alpha=.1)
    predictor.calibrate(["one"] * 9, ["clarity"] * 9)
    assert predictor.threshold == pytest.approx(.2)


def write_dataset(path, split, suffix=""):
    rows = []
    for index, (bucket, option) in enumerate([
        ("Clarity", "D8_NONE"), ("Clarity", "D8_SHARES"),
        ("Undecided", ""), ("Confusion", ""),
    ]):
        rows.append(dict(id=f"{split}-{index}", question_id="D8", text=f"{split} text {index}{suffix}",
                         bucket=bucket, option_id=option, source="fictional_test", group_id=f"{split}-{index}", split=split))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_factory_ignores_locked_test_and_caches_by_training_sources(tmp_path, monkeypatch):
    from ai import all_question_ml

    spec = BusinessSpec({"version": "unit-test", "question_order": ["D8"], "questions": [
        {"id": "D8", "order": 1, "topic": "experience", "question": "What have you held?",
         "main_options": [{"id": "D8_NONE", "label": "None"}, {"id": "D8_SHARES", "label": "Shares"}]}]})
    write_dataset(tmp_path / "train.csv", "train")
    write_dataset(tmp_path / "calibration.csv", "calibration")
    (tmp_path / "test.csv").write_text("This must not be parsed or trained on.")
    calls = []

    def fit(train, calibration, target):
        calls.append(target)
        return StubPredictor({"clarity" if target == "bucket" else "D8_NONE"}), lambda text: LexicalSupport(1, 1)

    monkeypatch.setattr(all_question_ml, "_fit_predictor", fit)
    all_question_ml._CACHE.clear()
    first = build_all_question_ml_resolver(spec, tmp_path)
    assert build_all_question_ml_resolver(spec, tmp_path) is first
    assert calls == ["bucket", "option_id"]
    write_dataset(tmp_path / "calibration.csv", "calibration", " changed")
    second = build_all_question_ml_resolver(spec, tmp_path)
    assert second is not first and second.source_hash != first.source_hash
    assert calls == ["bucket", "option_id", "bucket", "option_id"]


def test_factory_rejects_training_calibration_group_leakage(tmp_path):
    from ai import all_question_ml

    write_dataset(tmp_path / "train.csv", "train")
    write_dataset(tmp_path / "calibration.csv", "calibration")
    payload = (tmp_path / "calibration.csv").read_text().replace("calibration-0,calibration", "train-0,calibration")
    (tmp_path / "calibration.csv").write_text(payload)
    spec = BusinessSpec({"version": "unit-test-leak", "question_order": [], "questions": []})
    with pytest.raises(ValueError, match="paraphrase groups overlap"):
        all_question_ml.build_all_question_ml_resolver(spec, tmp_path)
