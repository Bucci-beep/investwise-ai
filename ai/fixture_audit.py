"""Audit business fixtures against InvestWise system responsibilities.

The audit keeps four responsibilities separate:

1. Intent interpretation
2. Fixed option interpretation
3. Deterministic governance
4. End-to-end regression testing

Business fixtures are specification and regression evidence. They are
not automatically treated as classifier training or calibration data.
"""

from dataclasses import dataclass

from ai.decision_schema import (
    CandidateStatus,
    GovernanceFlag,
    InterpretationState,
)
from ai.dialogue_data import DialogueFixture
from ai.fixture_mapper import map_fixture


@dataclass(frozen=True)
class FixtureAudit:
    fixture_id: str

    intent_relevant: bool
    option_relevant: bool
    governance_relevant: bool
    regression_relevant: bool

    intent_target: str | None
    option_target: str | None

    notes: tuple[str, ...]


def audit_fixture(
    fixture: DialogueFixture,
) -> FixtureAudit:
    """Determine which system responsibilities a fixture exercises."""

    mapping = map_fixture(fixture)
    state = mapping.decision_state

    notes: list[str] = []

    # Every bounded free-text fixture is useful for testing whether
    # Stage 1 interprets the customer's semantic response correctly.
    intent_relevant = True
    intent_target = state.interpretation_state.value

    # Stage 2 is only valid when a supported financial candidate exists.
    option_relevant = (
        state.interpretation_state == InterpretationState.CLARITY
        and state.candidate_status == CandidateStatus.UNCONFIRMED
        and state.financial_category is not None
    )

    option_target = (
        state.financial_category
        if option_relevant
        else None
    )

    # Independent flags belong to deterministic governance rather than
    # being collapsed into the NLP class.
    governance_relevant = bool(state.governance_flags)

    # Every authored business fixture remains useful for end-to-end
    # regression behaviour.
    regression_relevant = True

    if GovernanceFlag.REFUSAL in state.governance_flags:
        notes.append(
            "clear control: do not create a financial candidate"
        )

    if GovernanceFlag.MATERIAL_CONFLICT in state.governance_flags:
        notes.append(
            "deterministic conflict: resolve before progression"
        )

    if GovernanceFlag.SAME_MONEY_DEPENDENCY in state.governance_flags:
        notes.append(
            "same-money dependency requires governance handling"
        )

    if GovernanceFlag.SAFETY_OR_DISTRESS in state.governance_flags:
        notes.append(
            "safety/distress flag remains independent of financial category"
        )

    if GovernanceFlag.CORRECTION in state.governance_flags:
        notes.append(
            "correction requires controlled replacement behaviour"
        )

    if state.interpretation_state == InterpretationState.CONFUSION:
        notes.append(
            "do not run fixed option mapping"
        )

    if state.interpretation_state == InterpretationState.UNDECIDED:
        notes.append(
            "do not infer a financial category"
        )

    if option_relevant:
        notes.append(
            "eligible for fixed-option interpretation testing"
        )

    return FixtureAudit(
        fixture_id=fixture.fixture_id,
        intent_relevant=intent_relevant,
        option_relevant=option_relevant,
        governance_relevant=governance_relevant,
        regression_relevant=regression_relevant,
        intent_target=intent_target,
        option_target=option_target,
        notes=tuple(notes),
    )
