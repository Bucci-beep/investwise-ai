"""Suitability decision synthesis for InvestWise AI.

This module combines financial risk capacity, stated risk tolerance and
consistency checks into a final decision-support outcome.

It does not recommend specific investments or financial products.
"""

from dataclasses import dataclass, field
from enum import Enum

from ai.consistency import ConsistencyResult, ConsistencyStatus
from ai.models import CustomerProfile, RiskLevel
from ai.risk_capacity import RiskCapacityResult


class SuitabilityDecision(str, Enum):
    SUITABLE = "suitable"
    REVIEW_REQUIRED = "review_required"
    NOT_SUITABLE = "not_suitable"


@dataclass
class SuitabilityAssessment:
    """Final suitability decision and supporting evidence."""

    decision: SuitabilityDecision
    effective_risk_level: RiskLevel | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_human_review: bool = False


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


def _lower_risk_level(
    first: RiskLevel,
    second: RiskLevel,
) -> RiskLevel:
    """Return the more conservative of two risk levels."""

    if _RISK_ORDER[first] <= _RISK_ORDER[second]:
        return first

    return second


def assess_suitability(
    profile: CustomerProfile,
    capacity: RiskCapacityResult,
    consistency: ConsistencyResult,
) -> SuitabilityAssessment:
    """Synthesize existing assessment evidence into a suitability outcome.

    The effective risk level is bounded by both stated risk tolerance and
    assessed financial capacity. Higher financial capacity never overrides
    a customer's lower stated willingness to take risk.
    """

    reasons: list[str] = []
    warnings: list[str] = []

    warnings.extend(capacity.warnings)
    warnings.extend(consistency.warnings)

    # Strong contradictions stop automatic suitability assessment.
    if consistency.status == ConsistencyStatus.CONFLICT:
        reasons.append(
            "Material contradictions were detected between the customer's "
            "stated preferences and assessment evidence."
        )
        reasons.extend(consistency.conflicts)

        return SuitabilityAssessment(
            decision=SuitabilityDecision.REVIEW_REQUIRED,
            effective_risk_level=None,
            reasons=reasons,
            warnings=warnings,
            requires_human_review=True,
        )

    # Less severe inconsistencies still require clarification.
    if consistency.status == ConsistencyStatus.REVIEW_REQUIRED:
        effective_level = _lower_risk_level(
            profile.stated_risk_tolerance,
            capacity.level,
        )

        reasons.append(
            "The assessment contains a risk mismatch that should be "
            "clarified before an automatic suitability outcome is used."
        )

        return SuitabilityAssessment(
            decision=SuitabilityDecision.REVIEW_REQUIRED,
            effective_risk_level=effective_level,
            reasons=reasons,
            warnings=warnings,
            requires_human_review=True,
        )

    # If the evidence is consistent, use the more conservative boundary.
    effective_level = _lower_risk_level(
        profile.stated_risk_tolerance,
        capacity.level,
    )

    if capacity.level == RiskLevel.LOW:
        reasons.append(
            "The customer's assessed financial risk capacity is low."
        )

        return SuitabilityAssessment(
            decision=SuitabilityDecision.SUITABLE,
            effective_risk_level=RiskLevel.LOW,
            reasons=reasons,
            warnings=warnings,
            requires_human_review=False,
        )

    reasons.append(
        "Stated risk tolerance and assessed financial risk capacity are "
        "sufficiently aligned for the assessment to proceed."
    )

    if capacity.level != profile.stated_risk_tolerance:
        reasons.append(
            "The effective risk level uses the more conservative of stated "
            "risk tolerance and assessed financial capacity."
        )

    return SuitabilityAssessment(
        decision=SuitabilityDecision.SUITABLE,
        effective_risk_level=effective_level,
        reasons=reasons,
        warnings=warnings,
        requires_human_review=False,
    )
