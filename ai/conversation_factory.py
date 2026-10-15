from __future__ import annotations

import csv
from pathlib import Path

from ai.conformal import SplitConformalClassifier
from ai.conversation_engine import ConversationEngine
from ai.d12_adapter import ExistingBoundedInterpreterAdapter
from ai.horizon_mapper import map_explicit_horizon
from ai.interpretation import BoundedQuestionInterpreter
from ai.question_spec import BusinessSpec
from ai.text_classifier import BoundedTextClassifier


ROOT = Path(__file__).resolve().parent.parent

INTENT_TRAIN = ROOT / "data/nlp/splits/d12_intent_train.csv"
INTENT_CALIBRATION = ROOT / "data/nlp/splits/d12_intent_calibration.csv"

OPTION_TRAIN = ROOT / "data/nlp/splits/d12_options_train.csv"
OPTION_CALIBRATION = ROOT / "data/nlp/splits/d12_options_calibration.csv"

BUSINESS_SPEC = ROOT / "config/business-decision-spec.json"

CONFORMAL_ALPHA = 0.20


def load_csv(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])

    return texts, labels


def build_predictor(
    train_path: Path,
    calibration_path: Path,
) -> SplitConformalClassifier:
    train_texts, train_labels = load_csv(train_path)
    calibration_texts, calibration_labels = load_csv(
        calibration_path
    )

    classifier = BoundedTextClassifier()
    classifier.fit(
        train_texts,
        train_labels,
    )

    predictor = SplitConformalClassifier(
        classifier=classifier,
        alpha=CONFORMAL_ALPHA,
    )

    predictor.calibrate(
        calibration_texts,
        calibration_labels,
    )

    return predictor


def build_d12_interpreter() -> BoundedQuestionInterpreter:
    intent_predictor = build_predictor(
        INTENT_TRAIN,
        INTENT_CALIBRATION,
    )

    option_predictor = build_predictor(
        OPTION_TRAIN,
        OPTION_CALIBRATION,
    )

    return BoundedQuestionInterpreter(
        question_id="D12",
        intent_predictor=intent_predictor,
        option_predictor=option_predictor,
        deterministic_mapper=map_explicit_horizon,
    )


def build_conversation_engine(*, interpretation_mode: str = "ml") -> ConversationEngine:
    spec = BusinessSpec.load(BUSINESS_SPEC)

    if interpretation_mode == "ml":
        from ai.all_question_ml import build_all_question_ml_resolver
        resolver = build_all_question_ml_resolver(spec, ROOT / "data/nlp/all_questions")
        engine = ConversationEngine(business_spec=spec, question_resolver=resolver)
        engine.interpretation_mode = "ml"
        return engine
    if interpretation_mode != "legacy":
        raise ValueError("interpretation_mode must be ml or legacy")

    d12_interpreter = build_d12_interpreter()

    d12_resolver = ExistingBoundedInterpreterAdapter(
        d12_interpreter
    )

    engine = ConversationEngine(
        business_spec=spec,
        d12_resolver=d12_resolver,
    )
    engine.interpretation_mode = "legacy"
    return engine


def build_legacy_comparison_engine() -> ConversationEngine:
    """Archived hybrid for policy regression/comparison; not the app default."""
    return build_conversation_engine(interpretation_mode="legacy")


if __name__ == "__main__":
    engine = build_conversation_engine()

    print("Conversation engine ready")
    print("Business spec:", engine.spec.version)
    print("Current question:", engine.current_question.id)
