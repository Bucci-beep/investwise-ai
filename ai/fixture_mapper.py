"""Map authored InvestWise business fixtures into the decision schema.

This module is for governance and regression fixtures.

It does not train the NLP classifier and does not perform conformal
prediction.
"""

from dataclasses import dataclass

from ai.decision_schema import (
    CandidateStatus,
    DecisionAction,
    DecisionState,
    GovernanceFlag,
    InterpretationState,
)
from ai.dialogue_data import DialogueFixture


@dataclass(frozen=True)
class FixtureMapping:
    """Decision state expected from an authored business fixture."""

    fixture_id: str
    source_bucket: str
    decision_state: DecisionState


_FLAG_MAP = {
    "refusal": GovernanceFlag.REFUSAL,
    "pause": GovernanceFlag.PAUSE,
    "correction": GovernanceFlag.CORRECTION,
    "material_conflict": GovernanceFlag.MATERIAL_CONFLICT,
    "same_money_dependency": GovernanceFlag.SAME_MONEY_DEPENDENCY,
    "safety_or_distress": GovernanceFlag.SAFETY_OR_DISTRESS,
}


def _map_flags(
    flags: tuple[str, ...],
) -> frozenset[GovernanceFlag]:
    """Map recognised workbook flags into governance enums."""

    return frozenset(
        _FLAG_MAP[flag]
        for flag in flags
        if flag in _FLAG_MAP
    )


def _interpretation_state(
    fixture: DialogueFixture,
    flags: frozenset[GovernanceFlag],
) -> InterpretationState:
    """Resolve the semantic state represented by the fixture."""

    # A settled refusal is a clear control, not indecision.
    if GovernanceFlag.REFUSAL in flags:
        return InterpretationState.CLARITY

    bucket = fixture.ordinary_bucket.strip().lower()

    if bucket == "clarity":
        return InterpretationState.CLARITY

    if bucket == "undecided":
        return InterpretationState.UNDECIDED

    if bucket in {"confusion", "off-topic", "off topic"}:
        return InterpretationState.CONFUSION

    raise ValueError(
        f"unsupported ordinary bucket for {fixture.fixture_id}: "
        f"{fixture.ordinary_bucket!r}"
    )


def map_fixture(
    fixture: DialogueFixture,
) -> FixtureMapping:
    """Convert one authored fixture into its expected decision state."""

    flags = _map_flags(fixture.independent_flags)

    interpretation = _interpretation_state(
        fixture,
        flags,
    )

    financial_category = (
        fixture.candidate_ids[0]
        if len(fixture.candidate_ids) == 1
        else None
    )

    if (
        GovernanceFlag.MATERIAL_CONFLICT in flags
    ):
        action = DecisionAction.RESOLVE_CONFLICT
        candidate_status = CandidateStatus.NO_CANDIDATE
        financial_category = None

    elif GovernanceFlag.REFUSAL in flags:
        action = DecisionAction.ACCEPT_CONTROL
        candidate_status = CandidateStatus.NO_CANDIDATE
        financial_category = None

    elif interpretation == InterpretationState.CONFUSION:
        action = DecisionAction.RESTATE
        candidate_status = CandidateStatus.NO_CANDIDATE
        financial_category = None

    elif interpretation == InterpretationState.UNDECIDED:
        action = DecisionAction.CLARIFY
        candidate_status = CandidateStatus.NO_CANDIDATE
        financial_category = None

    elif financial_category is not None:
        action = DecisionAction.CONFIRM
        candidate_status = CandidateStatus.UNCONFIRMED

    else:
        action = DecisionAction.HOLD
        candidate_status = CandidateStatus.NO_CANDIDATE

    state = DecisionState(
        interpretation_state=interpretation,
        candidate_status=candidate_status,
        financial_category=financial_category,
        governance_flags=flags,
        action=action,
        model_uncertain=False,
        requires_confirmation=(
            action == DecisionAction.CONFIRM
        ),
        can_record=False,
    )

    return FixtureMapping(
        fixture_id=fixture.fixture_id,
        source_bucket=fixture.ordinary_bucket,
        decision_state=state,
    )
