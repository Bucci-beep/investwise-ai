"""Bounded natural language interpretation for InvestWise.

Architecture:

free text
→ intent classifier
→ intent conformal prediction
→ dialogue policy
→ optional deterministic bounded mapper
→ fixed option classifier when required
→ option conformal prediction
→ confirmation requirement

No financial suitability decision is made here.
"""

from dataclasses import dataclass
from typing import Callable

from ai.conformal import ConformalPrediction, SplitConformalClassifier
from ai.dialogue_policy import (
    DialogueAction,
    DialogueDecision,
    decide_intent,
    decide_option,
)
from ai.horizon_mapper import (
    HorizonMapping,
    HorizonMappingStatus,
)


@dataclass(frozen=True)
class InterpretationResult:
    """Traceable result from one bounded interpretation turn."""

    question_id: str
    raw_text: str

    intent_prediction: ConformalPrediction
    intent_decision: DialogueDecision

    deterministic_mapping: HorizonMapping | None = None

    option_prediction: ConformalPrediction | None = None
    option_decision: DialogueDecision | None = None

    proposed_value: str | None = None
    requires_confirmation: bool = False
    confirmed: bool = False
    recorded_value: str | None = None

    @property
    def model_uncertain(self) -> bool:
        if self.intent_prediction.is_model_uncertain:
            return True

        if (
            self.option_prediction is not None
            and self.option_prediction.is_model_uncertain
        ):
            return True

        return False

    @property
    def semantic_undecided(self) -> bool:
        return (
            self.intent_prediction.prediction_set
            == frozenset({"undecided"})
        )


class BoundedQuestionInterpreter:
    """Orchestrates bounded interpretation before financial governance."""

    def __init__(
        self,
        question_id: str,
        intent_predictor: SplitConformalClassifier,
        option_predictor: SplitConformalClassifier,
        deterministic_mapper: Callable[[str], HorizonMapping] | None = None,
    ) -> None:
        self.question_id = question_id
        self.intent_predictor = intent_predictor
        self.option_predictor = option_predictor
        self.deterministic_mapper = deterministic_mapper

    def interpret(self, text: str) -> InterpretationResult:
        """Interpret one free text answer without recording anything."""

        intent_prediction = self.intent_predictor.predict(text)
        intent_decision = decide_intent(intent_prediction)

        # Semantic intent has authority over option mapping.
        #
        # Undecided, confusion, model uncertainty, and empty sets must
        # never progress into a financial category.
        if intent_decision.action != DialogueAction.CONTINUE:
            return InterpretationResult(
                question_id=self.question_id,
                raw_text=text,
                intent_prediction=intent_prediction,
                intent_decision=intent_decision,
            )

        deterministic_mapping = None

        if self.deterministic_mapper is not None:
            deterministic_mapping = self.deterministic_mapper(text)

            # Explicit durations spanning financial boundaries must not
            # be reduced to one option by the NLP model.
            if (
                deterministic_mapping.status
                == HorizonMappingStatus.AMBIGUOUS
            ):
                return InterpretationResult(
                    question_id=self.question_id,
                    raw_text=text,
                    intent_prediction=intent_prediction,
                    intent_decision=intent_decision,
                    deterministic_mapping=deterministic_mapping,
                )

            # Explicit financial facts take precedence over statistical
            # option classification. Confirmation is still mandatory.
            if (
                deterministic_mapping.status
                == HorizonMappingStatus.MAPPED
            ):
                return InterpretationResult(
                    question_id=self.question_id,
                    raw_text=text,
                    intent_prediction=intent_prediction,
                    intent_decision=intent_decision,
                    deterministic_mapping=deterministic_mapping,
                    proposed_value=deterministic_mapping.category,
                    requires_confirmation=True,
                )

        # No explicit deterministic mapping was available.
        # Fall back to bounded option classification plus conformal set.
        option_prediction = self.option_predictor.predict(text)
        option_decision = decide_option(option_prediction)

        return InterpretationResult(
            question_id=self.question_id,
            raw_text=text,
            intent_prediction=intent_prediction,
            intent_decision=intent_decision,
            deterministic_mapping=deterministic_mapping,
            option_prediction=option_prediction,
            option_decision=option_decision,
            proposed_value=option_decision.proposed_value,
            requires_confirmation=option_decision.requires_confirmation,
        )


def confirm_interpretation(
    result: InterpretationResult,
    confirmed: bool,
) -> InterpretationResult:
    """Apply explicit customer confirmation to a proposed value."""

    if not confirmed:
        return InterpretationResult(
            question_id=result.question_id,
            raw_text=result.raw_text,
            intent_prediction=result.intent_prediction,
            intent_decision=result.intent_decision,
            deterministic_mapping=result.deterministic_mapping,
            option_prediction=result.option_prediction,
            option_decision=result.option_decision,
            proposed_value=result.proposed_value,
            requires_confirmation=result.requires_confirmation,
            confirmed=False,
            recorded_value=None,
        )

    if (
        result.proposed_value is None
        or not result.requires_confirmation
    ):
        raise ValueError(
            "cannot confirm an interpretation that has no confirmable option."
        )

    return InterpretationResult(
        question_id=result.question_id,
        raw_text=result.raw_text,
        intent_prediction=result.intent_prediction,
        intent_decision=result.intent_decision,
        deterministic_mapping=result.deterministic_mapping,
        option_prediction=result.option_prediction,
        option_decision=result.option_decision,
        proposed_value=result.proposed_value,
        requires_confirmation=False,
        confirmed=True,
        recorded_value=result.proposed_value,
    )
