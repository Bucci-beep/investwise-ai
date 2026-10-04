"""Reusable backend runtime for bounded D12 conversations.

This module owns conversational state transitions only. Classification,
conformal uncertainty, dialogue policy, and duration mapping remain owned by
the existing bounded interpreter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from ai.conversation_state import ConversationState, ConversationTurn, QuestionStatus
from ai.dialogue_policy import DialogueAction
from ai.horizon_mapper import HorizonMappingStatus
from ai.pre_model_controls import ControlType, check_pre_model_controls


_YES = re.compile(r"^\s*(?:yes|y|yeah|yep|correct|right|that's right|that is right)\s*[.!]?\s*$", re.I)
_NO = re.compile(r"^\s*(?:no|n|nope|incorrect|that's wrong|that is wrong)\s*[.!]?\s*$", re.I)


@dataclass(frozen=True)
class RuntimeResponse:
    question_id: str
    user_text: str
    action: str
    message: str
    proposed_value: str | None = None
    recorded_value: str | None = None
    requires_confirmation: bool = False
    model_uncertain: bool = False
    semantic_undecided: bool = False
    deterministic_mapping_status: str | None = None
    deterministic_durations_years: tuple[float, ...] = ()
    conformal_prediction_set: tuple[str, ...] = ()
    model_probabilities: dict[str, float] = field(default_factory=dict)
    option_conformal_prediction_set: tuple[str, ...] = ()
    option_probabilities: dict[str, float] = field(default_factory=dict)
    governance_flags: tuple[str, ...] = ()
    clarification_count: int = 0
    question_status: str = QuestionStatus.NOT_STARTED.value
    control_type: str = ControlType.NONE.value
    control_matched: bool = False
    matched_rule: str | None = None
    should_call_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationRuntime:
    """Stateful adapter around an existing bounded question interpreter."""

    def __init__(self, interpreter: Any) -> None:
        self.interpreter = interpreter

    def handle_reply(
        self,
        question_id: str,
        user_text: str,
        state: ConversationState,
    ) -> RuntimeResponse:
        if question_id != "D12":
            raise ValueError("This runtime currently supports D12 only")
        if getattr(self.interpreter, "question_id", question_id) != question_id:
            raise ValueError("Interpreter question does not match requested question")

        question = state.questions.get(question_id) or state.start_question(question_id)

        if state.stopped:
            return self._respond(
                state, question_id, user_text, "stopped", "The conversation has stopped."
            )

        # Controls intentionally precede confirmation parsing and all model use.
        control = check_pre_model_controls(user_text, include_semantic_undecided=False)
        if control.matched:
            for flag in control.flags:
                state.add_governance_flag(question_id, flag)
            question.pending_confirmation = False
            question.proposed_value = None
            question.recorded_value = None
            control_trace = {
                "control_type": control.control_type.value,
                "control_matched": True,
                "matched_rule": control.matched_rule,
                "should_call_model": False,
            }
            if control.control_type in {ControlType.DISTRESS, ControlType.PAUSE}:
                state.pause()
            elif control.control_type in {ControlType.STOP, ControlType.SELF_HARM}:
                state.stop()
            elif control.control_type == ControlType.REFUSAL:
                question.status = QuestionStatus.INCOMPLETE
            return self._respond(
                state,
                question_id,
                user_text,
                control.action or "control",
                control.message or control.reason or "Control handled.",
                proposed_value=None,
                recorded_value=None,
                requires_confirmation=False,
                **control_trace,
            )

        if state.paused:
            return self._respond(state, question_id, user_text, "paused", "The conversation is paused.")

        if question.pending_confirmation:
            if _YES.match(user_text):
                recorded = state.confirm_value(question_id)
                return self._respond(
                    state, question_id, user_text, "recorded",
                    f"Recorded {recorded} after explicit confirmation.",
                    proposed_value=question.proposed_value,
                    recorded_value=recorded,
                )
            if _NO.match(user_text):
                state.reject_confirmation(question_id)
                return self._respond(
                    state, question_id, user_text, "reask",
                    "The proposed value was cleared. Please answer the D12 question again.",
                )
            return self._respond(
                state, question_id, user_text, "confirm",
                f"Please answer yes or no: should I record {question.proposed_value}?",
                proposed_value=question.proposed_value,
                requires_confirmation=True,
            )

        result = self.interpreter.interpret(user_text)
        intent_prediction = result.intent_prediction
        deterministic = result.deterministic_mapping
        option_prediction = result.option_prediction
        trace = {
            "deterministic_mapping_status": (
                deterministic.status.value if deterministic is not None else None
            ),
            "deterministic_durations_years": (
                deterministic.durations_years if deterministic is not None else ()
            ),
            "conformal_prediction_set": tuple(sorted(intent_prediction.prediction_set)),
            "model_probabilities": intent_prediction.probabilities,
            "option_conformal_prediction_set": (
                tuple(sorted(option_prediction.prediction_set)) if option_prediction else ()
            ),
            "option_probabilities": option_prediction.probabilities if option_prediction else {},
            "should_call_model": True,
        }

        if result.model_uncertain:
            return self._clarify(
                state, question_id, user_text, "clarify",
                "I could not interpret that safely. Please state the earliest realistic time more directly.",
                model_uncertain=True,
                **trace,
            )

        if result.semantic_undecided:
            return self._clarify(
                state, question_id, user_text, "reoffer",
                "No category was chosen. Please give the earliest realistic time you may need any of the money.",
                semantic_undecided=True,
                **trace,
            )

        if deterministic is not None and deterministic.status == HorizonMappingStatus.AMBIGUOUS:
            return self._clarify(
                state, question_id, user_text, "clarify",
                "Those durations cross D12 boundaries. Which is the earliest realistic need?",
                **trace,
            )

        if result.intent_decision.action == DialogueAction.RESTATE:
            return self._clarify(
                state, question_id, user_text, "explain/restate",
                "This asks for the earliest realistic time you might need any of this money.",
                **trace,
            )

        if result.proposed_value is not None and result.requires_confirmation:
            state.set_proposed_value(question_id, result.proposed_value, user_text)
            return self._respond(
                state, question_id, user_text, "confirm",
                f"I interpreted that as {result.proposed_value}. Should I record it?",
                proposed_value=result.proposed_value,
                requires_confirmation=True,
                **trace,
            )

        return self._clarify(
            state, question_id, user_text, "clarify",
            "I could not produce a safe D12 value. Please clarify the earliest realistic need.",
            **trace,
        )

    def _clarify(
        self,
        state: ConversationState,
        question_id: str,
        user_text: str,
        action: str,
        message: str,
        **trace: Any,
    ) -> RuntimeResponse:
        count = state.increment_clarification(question_id)
        question = state.get_question(question_id)
        if question.status == QuestionStatus.INCOMPLETE:
            action = "incomplete"
            message = "Maximum clarification attempts reached; D12 remains incomplete."
        return self._respond(state, question_id, user_text, action, message, **trace)

    def _respond(
        self,
        state: ConversationState,
        question_id: str,
        user_text: str,
        action: str,
        message: str,
        **values: Any,
    ) -> RuntimeResponse:
        question = state.get_question(question_id)
        question.last_user_text = user_text
        question.last_action = action
        response = RuntimeResponse(
            question_id=question_id,
            user_text=user_text,
            action=action,
            message=message,
            proposed_value=values.pop("proposed_value", question.proposed_value),
            recorded_value=values.pop("recorded_value", question.recorded_value),
            requires_confirmation=values.pop("requires_confirmation", question.pending_confirmation),
            model_uncertain=values.pop("model_uncertain", False),
            semantic_undecided=values.pop("semantic_undecided", False),
            governance_flags=tuple(question.governance_flags),
            clarification_count=question.clarification_count,
            question_status=question.status.value,
            **values,
        )
        state.add_turn(
            ConversationTurn(
                turn_index=len(state.turns),
                question_id=question_id,
                user_text=user_text,
                action=action,
                proposed_value=response.proposed_value,
                recorded_value=response.recorded_value,
                model_uncertain=response.model_uncertain,
                semantic_undecided=response.semantic_undecided,
                governance_flags=list(response.governance_flags),
                metadata={
                    "control_type": response.control_type,
                    "control_matched": response.control_matched,
                    "matched_rule": response.matched_rule,
                    "should_call_model": response.should_call_model,
                    "deterministic_mapping_status": response.deterministic_mapping_status,
                    "conformal_prediction_set": list(response.conformal_prediction_set),
                },
            )
        )
        return response
