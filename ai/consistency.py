"""Consistency checking for the InvestWise AI decision engine.

This module compares a customer's stated preferences with their assessed
financial risk capacity and identifies contradictions that may require
clarification or human review.
"""

from dataclasses import dataclass, field
from enum import Enum

from ai.models import CustomerProfile, RiskLevel
from ai.risk_capacity import RiskCapacityResult


class ConsistencyStatus(str, Enum):
    CONSISTENT = "consistent"
    REVIEW_REQUIRED = "review_required"
    CONFLICT = "conflict"


@dataclass
class ConsistencyResult:
    """Result of consistency checks across the customer assessment."""

    status: ConsistencyStatus
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}


def check_consistency(
    profile: CustomerProfile,
    capacity: RiskCapacityResult,
) -> ConsistencyResult:
    """Compare stated preferences with assessed financial capacity."""

    conflicts: list[str] = []
    warnings: list[str] = []
    explanations: list[str] = []

    stated_level = _RISK_ORDER[profile.stated_risk_tolerance]
    capacity_level = _RISK_ORDER[capacity.level]
    risk_gap = stated_level - capacity_level

    # Stated tolerance exceeds assessed financial capacity.
    if risk_gap >= 2:
        conflicts.append(
            "Stated risk tolerance is substantially higher than assessed "
            "financial risk capacity."
        )
        explanations.append(
            "The customer reports high willingness to take investment risk, "
            "but their financial circumstances indicate low capacity to "
            "absorb losses."
        )

    elif risk_gap == 1:
        warnings.append(
            "Stated risk tolerance is higher than assessed financial "
            "risk capacity."
        )
        explanations.append(
            "The customer's willingness to take risk exceeds their assessed "
            "financial capacity by one risk category."
        )

    # Capacity can exceed willingness, but this should never be interpreted
    # as permission to increase the customer's preferred risk level.
    elif risk_gap < 0:
        explanations.append(
            "Assessed financial risk capacity exceeds stated risk tolerance. "
            "The customer's lower stated tolerance should still be respected."
        )

    else:
        explanations.append(
            "Stated risk tolerance is aligned with assessed financial "
            "risk capacity."
        )

    # Short horizon combined with high stated risk preference.
    if (
        profile.investment_horizon_years < 3
        and profile.stated_risk_tolerance == RiskLevel.HIGH
    ):
        conflicts.append(
            "High stated risk tolerance conflicts with a short investment "
            "horizon."
        )
        explanations.append(
            "A short investment horizon may provide insufficient time to "
            "recover from significant market losses."
        )

    # High liquidity requirement combined with high stated risk preference.
    if (
        profile.liquidity_need.value == "high"
        and profile.stated_risk_tolerance == RiskLevel.HIGH
    ):
        conflicts.append(
            "High stated risk tolerance conflicts with high liquidity needs."
        )
        explanations.append(
            "The customer reports a strong need for accessible funds while "
            "also expressing high willingness to accept investment risk."
        )

    # Very low loss tolerance conflicts with high stated risk tolerance.
    if (
        profile.loss_tolerance_percent is not None
        and profile.loss_tolerance_percent < 5
        and profile.stated_risk_tolerance == RiskLevel.HIGH
    ):
        conflicts.append(
            "High stated risk tolerance conflicts with very low reported "
            "loss tolerance."
        )
        explanations.append(
            "The customer describes themselves as highly risk tolerant but "
            "reports being able to tolerate less than a five percent loss."
        )

    # Determine overall consistency status.
    if conflicts:
        status = ConsistencyStatus.CONFLICT
    elif warnings:
        status = ConsistencyStatus.REVIEW_REQUIRED
    else:
        status = ConsistencyStatus.CONSISTENT

    return ConsistencyResult(
        status=status,
        conflicts=conflicts,
        warnings=warnings,
        explanations=explanations,
    )
