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
    question_order: list[str]
    answers: dict[str, AnswerRecord]
    current_index: int = 0
    profile_version: int = 1
    paused: bool = False
    stopped_for_safety: bool = False
    active_holds: set[str] = field(default_factory=set)
    final_accuracy_version: int | None = None
    save_consent_version: int | None = None
    saved_version: int | None = None
    audit: list[AuditEvent] = field(default_factory=list)

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
