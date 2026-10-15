"""End-to-end assessment pipeline for InvestWise AI.

The pipeline coordinates the individual decision components and returns
one structured assessment result.

Decision logic remains inside the specialist modules. The orchestrator
does not independently score or modify customer risk.
"""

from dataclasses import dataclass

from ai.consistency import ConsistencyResult, check_consistency
from ai.explanations import ExplanationResult, generate_explanation
from ai.models import CustomerProfile
from ai.risk_capacity import RiskCapacityResult, assess_risk_capacity
from ai.suitability import SuitabilityAssessment, assess_suitability


@dataclass
class InvestWiseAssessment:
    """Complete result produced by the InvestWise assessment pipeline."""

    profile: CustomerProfile
    risk_capacity: RiskCapacityResult
    consistency: ConsistencyResult
    suitability: SuitabilityAssessment
    explanation: ExplanationResult


def run_assessment(profile: CustomerProfile) -> InvestWiseAssessment:
    """Run a customer profile through the complete assessment pipeline."""

    capacity = assess_risk_capacity(profile)

    consistency = check_consistency(
        profile=profile,
        capacity=capacity,
    )

    suitability = assess_suitability(
        profile=profile,
        capacity=capacity,
        consistency=consistency,
    )

    explanation = generate_explanation(
        profile=profile,
        capacity=capacity,
        consistency=consistency,
        suitability=suitability,
    )

    return InvestWiseAssessment(
        profile=profile,
        risk_capacity=capacity,
        consistency=consistency,
        suitability=suitability,
        explanation=explanation,
    )
