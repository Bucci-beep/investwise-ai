"""Deterministic dialogue safety policy for InvestWise interpretation.

The policy converts conformal prediction sets into safe conversational
actions.

It does not classify text and it does not make financial suitability
decisions.
"""

from dataclasses import dataclass
from enum import Enum

from ai.conformal import ConformalPrediction


class DialogueClass(str, Enum):
    """Semantic classes for the first dialogue classification stage."""

    CLARITY = "clarity"
    UNDECIDED = "undecided"
    CONFUSION = "confusion"


class DialogueAction(str, Enum):
    """Actions available to the bounded dialogue system."""

    CONTINUE = "continue"
    CLARIFY = "clarify"
    RESTATE = "restate"
    CONFIRM = "confirm"
    ABSTAIN = "abstain"


class DialogueReason(str, Enum):
    """Internal reason for the selected dialogue action."""

    CLEAR = "clear"
    SEMANTIC_UNDECIDED = "semantic_undecided"
    SEMANTIC_CONFUSION = "semantic_confusion"
    MODEL_UNCERTAINTY = "model_uncertainty"
    EMPTY_PREDICTION_SET = "empty_prediction_set"
    OPTION_IDENTIFIED = "option_identified"


@dataclass(frozen=True)
class DialogueDecision:
    """Deterministic decision produced from a conformal prediction set."""

    action: DialogueAction
    reason: DialogueReason
    prediction_set: frozenset[str]
    proposed_value: str | None = None
    can_record: bool = False
    requires_confirmation: bool = False


def decide_intent(
    prediction: ConformalPrediction,
) -> DialogueDecision:
    """Apply dialogue policy to the first stage semantic prediction."""

    prediction_set = prediction.prediction_set

    if not prediction_set:
        return DialogueDecision(
            action=DialogueAction.ABSTAIN,
            reason=DialogueReason.EMPTY_PREDICTION_SET,
            prediction_set=prediction_set,
        )

    if len(prediction_set) > 1:
        return DialogueDecision(
            action=DialogueAction.CLARIFY,
            reason=DialogueReason.MODEL_UNCERTAINTY,
            prediction_set=prediction_set,
        )

    predicted_class = next(iter(prediction_set))

    if predicted_class == DialogueClass.CLARITY.value:
        return DialogueDecision(
            action=DialogueAction.CONTINUE,
            reason=DialogueReason.CLEAR,
            prediction_set=prediction_set,
        )

    if predicted_class == DialogueClass.UNDECIDED.value:
        return DialogueDecision(
            action=DialogueAction.CLARIFY,
            reason=DialogueReason.SEMANTIC_UNDECIDED,
            prediction_set=prediction_set,
        )

    if predicted_class == DialogueClass.CONFUSION.value:
        return DialogueDecision(
            action=DialogueAction.RESTATE,
            reason=DialogueReason.SEMANTIC_CONFUSION,
            prediction_set=prediction_set,
        )

    return DialogueDecision(
        action=DialogueAction.ABSTAIN,
        reason=DialogueReason.MODEL_UNCERTAINTY,
        prediction_set=prediction_set,
    )


def decide_option(
    prediction: ConformalPrediction,
) -> DialogueDecision:
    """Apply policy to the second stage fixed option prediction."""

    prediction_set = prediction.prediction_set

    if not prediction_set:
        return DialogueDecision(
            action=DialogueAction.ABSTAIN,
            reason=DialogueReason.EMPTY_PREDICTION_SET,
            prediction_set=prediction_set,
        )

    if len(prediction_set) > 1:
        return DialogueDecision(
            action=DialogueAction.CLARIFY,
            reason=DialogueReason.MODEL_UNCERTAINTY,
            prediction_set=prediction_set,
        )

    proposed_value = next(iter(prediction_set))

    return DialogueDecision(
        action=DialogueAction.CONFIRM,
        reason=DialogueReason.OPTION_IDENTIFIED,
        prediction_set=prediction_set,
        proposed_value=proposed_value,
        can_record=False,
        requires_confirmation=True,
    )
