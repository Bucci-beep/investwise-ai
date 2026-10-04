from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerStatus(str, Enum):
    PENDING = "pending"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    UNDECIDED = "undecided"
    CONFUSION = "confusion"
    DECLINED = "declined"
    PAUSED = "paused"


class QuestionStatus(str, Enum):
    """Lifecycle for a bounded backend question."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    PAUSED = "paused"
    INCOMPLETE = "incomplete"


@dataclass
class ConversationTurn:
    turn_index: int
    question_id: str
    user_text: str
    action: str
    proposed_value: str | None = None
    recorded_value: str | None = None
    model_uncertain: bool = False
    semantic_undecided: bool = False
    governance_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionState:
    question_id: str
    status: QuestionStatus = QuestionStatus.NOT_STARTED
    clarification_count: int = 0
    proposed_value: str | None = None
    recorded_value: str | None = None
    pending_confirmation: bool = False
    last_user_text: str | None = None
    last_action: str | None = None
    governance_flags: list[str] = field(default_factory=list)


@dataclass
class AnswerRecord:
    question_id: str
    status: AnswerStatus = AnswerStatus.PENDING
    raw_text: str | None = None
    candidate_option_id: str | None = None
    selected_option_id: str | None = None
    selected_option_label: str | None = None
    source: str | None = None
    reason: str | None = None
    clarification_turns: int = 0
    independent_flags: set[str] = field(default_factory=set)
    confirmed_version: int | None = None


@dataclass
class AuditEvent:
    event_type: str
    profile_version: int
    question_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    # The first fields preserve the existing D1-D15 engine interface.
    question_order: list[str] = field(default_factory=list)
    answers: dict[str, AnswerRecord] = field(default_factory=dict)
    current_index: int = 0
    profile_version: int = 1
    paused: bool = False
    stopped_for_safety: bool = False
    active_holds: set[str] = field(default_factory=set)
    final_accuracy_version: int | None = None
    save_consent_version: int | None = None
    saved_version: int | None = None
    audit: list[AuditEvent] = field(default_factory=list)

    # Reusable bounded-runtime state. Kept alongside the legacy shell fields so
    # current callers remain runnable while new question runtimes share one API.
    questions: dict[str, QuestionState] = field(default_factory=dict)
    turns: list[ConversationTurn] = field(default_factory=list)
    active_question_id: str | None = None
    stopped: bool = False
    max_clarifications_per_question: int = 3

    @property
    def current_question_id(self) -> str | None:
        if self.current_index >= len(self.question_order):
            return None
        return self.question_order[self.current_index]

    def bump_version(self) -> None:
        self.profile_version += 1
        self.final_accuracy_version = None
        self.save_consent_version = None
        self.saved_version = None

    def log(self, event_type: str, question_id: str | None = None, **payload: Any) -> None:
        self.audit.append(
            AuditEvent(
                event_type=event_type,
                profile_version=self.profile_version,
                question_id=question_id,
                payload=payload,
            )
        )

    def start_question(self, question_id: str) -> QuestionState:
        question = self.questions.setdefault(question_id, QuestionState(question_id))
        if question.status in {QuestionStatus.NOT_STARTED, QuestionStatus.PAUSED}:
            question.status = QuestionStatus.IN_PROGRESS
        self.active_question_id = question_id
        return question

    def get_question(self, question_id: str) -> QuestionState:
        if question_id not in self.questions:
            raise KeyError(question_id)
        return self.questions[question_id]

    def set_proposed_value(self, question_id: str, value: str, user_text: str | None = None) -> QuestionState:
        question = self.start_question(question_id)
        question.proposed_value = value
        question.recorded_value = None
        question.pending_confirmation = True
        question.status = QuestionStatus.AWAITING_CONFIRMATION
        question.last_user_text = user_text
        question.last_action = "confirm"
        return question

    def confirm_value(self, question_id: str) -> str:
        question = self.get_question(question_id)
        if not question.pending_confirmation or question.proposed_value is None:
            raise ValueError(f"{question_id} has no value awaiting confirmation")
        question.recorded_value = question.proposed_value
        question.pending_confirmation = False
        question.status = QuestionStatus.CONFIRMED
        question.last_action = "recorded"
        return question.recorded_value

    def reject_confirmation(self, question_id: str) -> QuestionState:
        question = self.get_question(question_id)
        question.proposed_value = None
        question.recorded_value = None
        question.pending_confirmation = False
        question.status = QuestionStatus.IN_PROGRESS
        question.last_action = "reask"
        return question

    def increment_clarification(self, question_id: str) -> int:
        question = self.start_question(question_id)
        question.clarification_count += 1
        question.pending_confirmation = False
        question.proposed_value = None
        question.recorded_value = None
        if question.clarification_count >= self.max_clarifications_per_question:
            question.status = QuestionStatus.INCOMPLETE
        else:
            question.status = QuestionStatus.IN_PROGRESS
        return question.clarification_count

    def add_governance_flag(self, question_id: str, flag: str) -> None:
        question = self.start_question(question_id)
        if flag not in question.governance_flags:
            question.governance_flags.append(flag)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def pause(self) -> None:
        self.paused = True
        if self.active_question_id in self.questions:
            self.questions[self.active_question_id].status = QuestionStatus.PAUSED

    def resume(self) -> None:
        self.paused = False
        if self.active_question_id in self.questions:
            question = self.questions[self.active_question_id]
            if question.status == QuestionStatus.PAUSED:
                question.status = (
                    QuestionStatus.AWAITING_CONFIRMATION
                    if question.pending_confirmation
                    else QuestionStatus.IN_PROGRESS
                )

    def stop(self) -> None:
        self.stopped = True

    def summary(self) -> dict[str, Any]:
        def serialise(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, set):
                return sorted(value)
            if isinstance(value, list):
                return [serialise(item) for item in value]
            if isinstance(value, dict):
                return {key: serialise(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {
                    key: serialise(getattr(value, key))
                    for key in value.__dataclass_fields__
                }
            return value

        return {
            "questions": serialise(self.questions),
            "turns": serialise(self.turns),
            "active_question_id": self.active_question_id,
            "paused": self.paused,
            "stopped": self.stopped,
            "max_clarifications_per_question": self.max_clarifications_per_question,
        }
