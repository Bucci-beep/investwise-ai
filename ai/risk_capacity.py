"""Transparent risk capacity assessment for InvestWise AI.

Risk capacity describes a customer's financial ability to absorb investment
risk. It is deliberately kept separate from stated risk tolerance.

This module provides deterministic decision support. It does not provide
financial advice or select financial products.
"""

from dataclasses import dataclass, field

from ai.models import CustomerProfile, RiskLevel


@dataclass
class RiskCapacityResult:
    """Detailed output from the risk capacity assessment."""

    level: RiskLevel
    score: int
    warnings: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)


def _validate_profile(profile: CustomerProfile) -> None:
    """Validate values required by the risk capacity assessment."""

    if profile.age < 18:
        raise ValueError("Customer must be at least 18 years old.")

    if profile.investment_horizon_years < 0:
        raise ValueError("Investment horizon cannot be negative.")

    if profile.annual_income_gbp < 0:
        raise ValueError("Annual income cannot be negative.")

    if (
        profile.financial_commitments_gbp is not None
        and profile.financial_commitments_gbp < 0
    ):
        raise ValueError("Financial commitments cannot be negative.")

    if (
        profile.loss_tolerance_percent is not None
        and not 0 <= profile.loss_tolerance_percent <= 100
    ):
        raise ValueError("Loss tolerance must be between 0 and 100 percent.")


def assess_risk_capacity(profile: CustomerProfile) -> RiskCapacityResult:
    """Assess financial risk capacity using transparent rules.

    The score is an initial prototype score. Thresholds must be reviewed
    against the final questionnaire, business rules and evidence before the
    system is treated as a validated suitability assessment.
    """

    _validate_profile(profile)

    score = 0
    warnings: list[str] = []
    explanations: list[str] = []

    # Investment horizon
    if profile.investment_horizon_years >= 10:
        score += 2
        explanations.append(
            "A long investment horizon increases the capacity to absorb "
            "short term market volatility."
        )
    elif profile.investment_horizon_years >= 5:
        score += 1
        explanations.append(
            "A medium investment horizon provides some capacity to absorb "
            "market volatility."
        )
    elif profile.investment_horizon_years >= 3:
        explanations.append(
            "The investment horizon provides limited time to recover from "
            "significant market losses."
        )
    else:
        score -= 2
        warnings.append("Short investment horizon.")
        explanations.append(
            "A short investment horizon reduces the capacity to recover "
            "from market losses."
        )

    # Liquidity requirements
    if profile.liquidity_need.value == "low":
        score += 1
        explanations.append(
            "Low liquidity requirements increase the ability to keep money "
            "invested during market volatility."
        )
    elif profile.liquidity_need.value == "high":
        score -= 2
        warnings.append("High liquidity requirement.")
        explanations.append(
            "A high need for accessible funds reduces capacity for "
            "investment risk."
        )
    else:
        explanations.append(
            "Moderate liquidity requirements do not materially change the "
            "initial risk capacity score."
        )

    # Emergency reserves
    if profile.emergency_fund:
        score += 1
        explanations.append(
            "An emergency fund reduces the likelihood that invested assets "
            "must be sold to cover unexpected short term expenses."
        )
    else:
        score -= 1
        warnings.append("No emergency fund reported.")
        explanations.append(
            "Without an emergency fund, unexpected expenses may create a "
            "need to access invested money."
        )

    # Financial commitments relative to income
    if profile.financial_commitments_gbp is not None:
        if profile.annual_income_gbp == 0:
            if profile.financial_commitments_gbp > 0:
                score -= 2
                warnings.append(
                    "Financial commitments reported with no annual income."
                )
                explanations.append(
                    "Reported financial commitments combined with no annual "
                    "income reduce financial capacity for investment losses."
                )
        else:
            commitment_ratio = (
                profile.financial_commitments_gbp
                / profile.annual_income_gbp
            )

            if commitment_ratio >= 0.75:
                score -= 2
                warnings.append(
                    "Financial commitments are high relative to annual income."
                )
                explanations.append(
                    "High financial commitments relative to income reduce "
                    "the ability to absorb investment losses."
                )
            elif commitment_ratio >= 0.50:
                score -= 1
                explanations.append(
                    "Financial commitments consume a substantial proportion "
                    "of annual income."
                )
            else:
                explanations.append(
                    "Reported financial commitments are below half of annual "
                    "income."
                )

    # Explicit ability to tolerate loss
    if profile.loss_tolerance_percent is not None:
        if profile.loss_tolerance_percent >= 20:
            score += 2
            explanations.append(
                "The reported ability to tolerate a substantial loss "
                "increases the initial risk capacity score."
            )
        elif profile.loss_tolerance_percent >= 10:
            score += 1
            explanations.append(
                "The reported loss tolerance provides some capacity for "
                "investment losses."
            )
        elif profile.loss_tolerance_percent < 5:
            score -= 2
            warnings.append("Very low reported loss tolerance.")
            explanations.append(
                "Very low reported loss tolerance reduces financial risk "
                "capacity."
            )
        else:
            score -= 1
            explanations.append(
                "The reported loss tolerance indicates limited capacity "
                "for investment losses."
            )

    # Convert the transparent score into a capacity category
    if score >= 4:
        level = RiskLevel.HIGH
    elif score >= 1:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    return RiskCapacityResult(
        level=level,
        score=score,
        warnings=warnings,
        explanations=explanations,
    )
