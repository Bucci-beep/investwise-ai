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


def build_conversation_engine() -> ConversationEngine:
    spec = BusinessSpec.load(BUSINESS_SPEC)

    d12_interpreter = build_d12_interpreter()

    d12_resolver = ExistingBoundedInterpreterAdapter(
        d12_interpreter
    )

    return ConversationEngine(
        business_spec=spec,
        d12_resolver=d12_resolver,
    )


if __name__ == "__main__":
    engine = build_conversation_engine()

    print("Conversation engine ready")
    print("Business spec:", engine.spec.version)
    print("Current question:", engine.current_question.id)
