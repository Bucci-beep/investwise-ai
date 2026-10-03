"""Shared decision schema for the InvestWise dialogue interpretation layer.

This module defines the internal states passed between language
interpretation, conformal prediction, dialogue policy, and deterministic
governance.

It contains no machine learning and no financial suitability logic.
"""

from dataclasses import dataclass, field
from enum import Enum


class InterpretationState(str, Enum):
    """Semantic meaning of the customer's response."""

    CLARITY = "clarity"
    CONFUSION = "confusion"
    UNDECIDED = "undecided"


class CandidateStatus(str, Enum):
    """Status of a possible structured financial interpretation."""

    UNCONFIRMED = "unconfirmed_candidate"
    NO_CANDIDATE = "unresolved_or_control_no_candidate"


class GovernanceFlag(str, Enum):
    """Known deterministic governance conditions."""

    REFUSAL = "refusal"
    PAUSE = "pause"
    CORRECTION = "correction"
    MATERIAL_CONFLICT = "material_conflict"
    SAME_MONEY_DEPENDENCY = "same_money_dependency"
    SAFETY_OR_DISTRESS = "safety_or_distress"


class DecisionAction(str, Enum):
    """Permitted next actions after interpretation and governance."""

    CONTINUE_TO_OPTION = "continue_to_option"
    CONFIRM = "confirm"
    CLARIFY = "clarify"
    RESTATE = "restate"
    ACCEPT_CONTROL = "accept_control"
    RESOLVE_CONFLICT = "resolve_conflict"
    HOLD = "hold"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class DecisionState:
    """Structured state for one bounded customer response.

    Semantic interpretation, candidate status, governance flags, and
    model uncertainty remain separate dimensions.
    """

    interpretation_state: InterpretationState
    candidate_status: CandidateStatus

    financial_category: str | None = None

    governance_flags: frozenset[GovernanceFlag] = field(
        default_factory=frozenset
    )

    action: DecisionAction = DecisionAction.HOLD

    model_uncertain: bool = False
    requires_confirmation: bool = False
    can_record: bool = False

    def __post_init__(self) -> None:
        if (
            self.candidate_status == CandidateStatus.NO_CANDIDATE
            and self.financial_category is not None
        ):
            raise ValueError(
                "a no-candidate state cannot contain a financial category."
            )

        if (
            self.candidate_status == CandidateStatus.UNCONFIRMED
            and self.financial_category is None
        ):
            raise ValueError(
                "an unconfirmed candidate requires a financial category."
            )

        if self.can_record and self.requires_confirmation:
            raise ValueError(
                "an answer cannot be recordable while confirmation is pending."
            )

        if self.can_record and self.model_uncertain:
            raise ValueError(
                "a model-uncertain answer cannot be recordable."
            )
