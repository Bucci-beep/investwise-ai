"""Evidence-based explanations for InvestWise AI.

This module converts outputs from the deterministic assessment pipeline into
clear explanations. It does not change, override or generate decisions.
"""

from dataclasses import dataclass, field

from ai.consistency import ConsistencyResult, ConsistencyStatus
from ai.models import CustomerProfile
from ai.risk_capacity import RiskCapacityResult
from ai.suitability import SuitabilityAssessment, SuitabilityDecision


@dataclass
class ExplanationResult:
    """Human-readable explanation of an assessment."""

    summary: str
    key_factors: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_action: str | None = None


def generate_explanation(
    profile: CustomerProfile,
    capacity: RiskCapacityResult,
    consistency: ConsistencyResult,
    suitability: SuitabilityAssessment,
) -> ExplanationResult:
    """Generate a traceable explanation from assessment evidence."""

    key_factors: list[str] = []
    conflicts: list[str] = []
    warnings: list[str] = []

    # Preserve evidence produced by the risk capacity engine.
    key_factors.extend(capacity.explanations)
    warnings.extend(capacity.warnings)

    # Preserve consistency evidence.
    key_factors.extend(consistency.explanations)
    conflicts.extend(consistency.conflicts)
    warnings.extend(consistency.warnings)

    # Preserve suitability reasoning.
    key_factors.extend(suitability.reasons)
    warnings.extend(suitability.warnings)

    # Remove duplicates while preserving original order.
    key_factors = list(dict.fromkeys(key_factors))
    conflicts = list(dict.fromkeys(conflicts))
    warnings = list(dict.fromkeys(warnings))

    if suitability.decision == SuitabilityDecision.REVIEW_REQUIRED:
        if consistency.status == ConsistencyStatus.CONFLICT:
            summary = (
                "The assessment requires review because material "
                "contradictions were identified in the customer's "
                "risk information."
            )
        else:
            summary = (
                "The assessment requires review because stated risk "
                "preferences and assessed financial capacity are not "
                "fully aligned."
            )

        next_action = (
            "Clarify the conflicting or uncertain information before "
            "continuing the suitability assessment."
        )

    elif suitability.decision == SuitabilityDecision.SUITABLE:
        if suitability.effective_risk_level is None:
            summary = (
                "The assessment can proceed, but no effective risk level "
                "was established."
            )
        else:
            summary = (
                "The assessment can proceed with an effective risk "
                f"boundary of {suitability.effective_risk_level.value}."
            )

        next_action = (
            "Continue to the next stage of the assessment using the "
            "established risk boundary."
        )

    else:
        summary = (
            "The current assessment does not support proceeding under "
            "the evaluated conditions."
        )

        next_action = (
            "Review the assessment evidence before continuing."
        )

    return ExplanationResult(
        summary=summary,
        key_factors=key_factors,
        conflicts=conflicts,
        warnings=warnings,
        next_action=next_action,
    )
