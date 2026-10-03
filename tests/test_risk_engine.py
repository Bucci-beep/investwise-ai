"""Tests for the InvestWise risk capacity engine."""

import pytest

from ai.models import (
    CustomerProfile,
    ExperienceLevel,
    LiquidityNeed,
    RiskLevel,
)
from ai.risk_capacity import assess_risk_capacity


def make_profile(**overrides) -> CustomerProfile:
    """Create a valid baseline customer profile for testing."""

    data = {
        "age": 35,
        "investment_horizon_years": 7,
        "annual_income_gbp": 45_000,
        "emergency_fund": True,
        "investment_experience": ExperienceLevel.BEGINNER,
        "liquidity_need": LiquidityNeed.MEDIUM,
        "stated_risk_tolerance": RiskLevel.MEDIUM,
        "goal": "long term investing",
        "financial_commitments_gbp": 10_000,
        "loss_tolerance_percent": 10,
    }

    data.update(overrides)

    return CustomerProfile(**data)


def test_high_capacity_profile():
    profile = make_profile(
        investment_horizon_years=15,
        liquidity_need=LiquidityNeed.LOW,
        emergency_fund=True,
        financial_commitments_gbp=10_000,
        loss_tolerance_percent=25,
    )

    result = assess_risk_capacity(profile)

    assert result.level == RiskLevel.HIGH
    assert result.score >= 4


def test_low_capacity_profile():
    profile = make_profile(
        investment_horizon_years=1,
        liquidity_need=LiquidityNeed.HIGH,
        emergency_fund=False,
        financial_commitments_gbp=40_000,
        loss_tolerance_percent=2,
    )

    result = assess_risk_capacity(profile)

    assert result.level == RiskLevel.LOW
    assert result.score <= 0
    assert len(result.warnings) > 0


def test_medium_capacity_profile():
    profile = make_profile(
        investment_horizon_years=5,
        liquidity_need=LiquidityNeed.MEDIUM,
        emergency_fund=True,
        financial_commitments_gbp=10_000,
        loss_tolerance_percent=10,
    )

    result = assess_risk_capacity(profile)

    assert result.level == RiskLevel.MEDIUM
    assert 1 <= result.score < 4


def test_short_horizon_generates_warning():
    profile = make_profile(
        investment_horizon_years=2,
    )

    result = assess_risk_capacity(profile)

    assert "Short investment horizon." in result.warnings


def test_high_liquidity_need_generates_warning():
    profile = make_profile(
        liquidity_need=LiquidityNeed.HIGH,
    )

    result = assess_risk_capacity(profile)

    assert "High liquidity requirement." in result.warnings


def test_missing_emergency_fund_generates_warning():
    profile = make_profile(
        emergency_fund=False,
    )

    result = assess_risk_capacity(profile)

    assert "No emergency fund reported." in result.warnings


def test_high_commitments_generate_warning():
    profile = make_profile(
        annual_income_gbp=40_000,
        financial_commitments_gbp=35_000,
    )

    result = assess_risk_capacity(profile)

    assert (
        "Financial commitments are high relative to annual income."
        in result.warnings
    )


def test_zero_income_with_commitments_is_handled():
    profile = make_profile(
        annual_income_gbp=0,
        financial_commitments_gbp=10_000,
    )

    result = assess_risk_capacity(profile)

    assert (
        "Financial commitments reported with no annual income."
        in result.warnings
    )


def test_age_does_not_automatically_reduce_capacity():
    younger_profile = make_profile(
        age=30,
        investment_horizon_years=15,
        liquidity_need=LiquidityNeed.LOW,
        loss_tolerance_percent=25,
    )

    older_profile = make_profile(
        age=65,
        investment_horizon_years=15,
        liquidity_need=LiquidityNeed.LOW,
        loss_tolerance_percent=25,
    )

    younger_result = assess_risk_capacity(younger_profile)
    older_result = assess_risk_capacity(older_profile)

    assert younger_result.score == older_result.score
    assert younger_result.level == older_result.level


def test_negative_investment_horizon_is_rejected():
    profile = make_profile(
        investment_horizon_years=-1,
    )

    with pytest.raises(
        ValueError,
        match="Investment horizon cannot be negative",
    ):
        assess_risk_capacity(profile)


def test_negative_income_is_rejected():
    profile = make_profile(
        annual_income_gbp=-1,
    )

    with pytest.raises(
        ValueError,
        match="Annual income cannot be negative",
    ):
        assess_risk_capacity(profile)


@pytest.mark.parametrize(
    "loss_tolerance",
    [-1, 101],
)
def test_invalid_loss_tolerance_is_rejected(loss_tolerance):
    profile = make_profile(
        loss_tolerance_percent=loss_tolerance,
    )

    with pytest.raises(
        ValueError,
        match="Loss tolerance must be between 0 and 100 percent",
    ):
        assess_risk_capacity(profile)


def test_under_18_customer_is_rejected():
    profile = make_profile(
        age=17,
    )

    with pytest.raises(
        ValueError,
        match="Customer must be at least 18 years old",
    ):
        assess_risk_capacity(profile)


# ---------------------------------------------------------------------------
# Consistency checking
# ---------------------------------------------------------------------------

from ai.consistency import ConsistencyStatus, check_consistency


def test_matching_tolerance_and_capacity_are_consistent():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.MEDIUM,
        investment_horizon_years=5,
        liquidity_need=LiquidityNeed.MEDIUM,
        emergency_fund=True,
        loss_tolerance_percent=10,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert capacity.level == RiskLevel.MEDIUM
    assert result.status == ConsistencyStatus.CONSISTENT
    assert result.conflicts == []


def test_high_tolerance_with_low_capacity_is_conflict():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=1,
        liquidity_need=LiquidityNeed.HIGH,
        emergency_fund=False,
        financial_commitments_gbp=40_000,
        loss_tolerance_percent=2,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert capacity.level == RiskLevel.LOW
    assert result.status == ConsistencyStatus.CONFLICT
    assert len(result.conflicts) > 0


def test_tolerance_one_level_above_capacity_requires_review():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=5,
        liquidity_need=LiquidityNeed.MEDIUM,
        emergency_fund=True,
        financial_commitments_gbp=10_000,
        loss_tolerance_percent=10,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert capacity.level == RiskLevel.MEDIUM
    assert result.status == ConsistencyStatus.REVIEW_REQUIRED


def test_capacity_above_tolerance_does_not_raise_customer_risk():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.LOW,
        investment_horizon_years=15,
        liquidity_need=LiquidityNeed.LOW,
        emergency_fund=True,
        financial_commitments_gbp=5_000,
        loss_tolerance_percent=25,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert capacity.level == RiskLevel.HIGH
    assert result.status == ConsistencyStatus.CONSISTENT
    assert any(
        "lower stated tolerance should still be respected" in explanation
        for explanation in result.explanations
    )


def test_high_risk_with_short_horizon_is_conflict():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=2,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert result.status == ConsistencyStatus.CONFLICT
    assert any(
        "short investment horizon" in conflict
        for conflict in result.conflicts
    )


def test_high_risk_with_high_liquidity_need_is_conflict():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        liquidity_need=LiquidityNeed.HIGH,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert result.status == ConsistencyStatus.CONFLICT
    assert any(
        "high liquidity needs" in conflict
        for conflict in result.conflicts
    )


def test_high_risk_with_very_low_loss_tolerance_is_conflict():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        loss_tolerance_percent=2,
    )

    capacity = assess_risk_capacity(profile)
    result = check_consistency(profile, capacity)

    assert result.status == ConsistencyStatus.CONFLICT
    assert any(
        "very low reported loss tolerance" in conflict
        for conflict in result.conflicts
    )


# ---------------------------------------------------------------------------
# Suitability decision synthesis
# ---------------------------------------------------------------------------

from ai.suitability import SuitabilityDecision, assess_suitability


def run_full_assessment(profile):
    """Run the assessment pipeline used by suitability tests."""

    capacity = assess_risk_capacity(profile)
    consistency = check_consistency(profile, capacity)
    suitability = assess_suitability(
        profile,
        capacity,
        consistency,
    )

    return capacity, consistency, suitability


def test_consistent_medium_profile_is_suitable():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.MEDIUM,
        investment_horizon_years=5,
        liquidity_need=LiquidityNeed.MEDIUM,
        emergency_fund=True,
        financial_commitments_gbp=10_000,
        loss_tolerance_percent=10,
    )

    _, consistency, suitability = run_full_assessment(profile)

    assert consistency.status == ConsistencyStatus.CONSISTENT
    assert suitability.decision == SuitabilityDecision.SUITABLE
    assert suitability.effective_risk_level == RiskLevel.MEDIUM
    assert suitability.requires_human_review is False


def test_material_conflict_requires_human_review():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=1,
        liquidity_need=LiquidityNeed.HIGH,
        emergency_fund=False,
        financial_commitments_gbp=40_000,
        loss_tolerance_percent=2,
    )

    _, consistency, suitability = run_full_assessment(profile)

    assert consistency.status == ConsistencyStatus.CONFLICT
    assert suitability.decision == SuitabilityDecision.REVIEW_REQUIRED
    assert suitability.effective_risk_level is None
    assert suitability.requires_human_review is True


def test_minor_risk_mismatch_requires_review():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=5,
        liquidity_need=LiquidityNeed.MEDIUM,
        emergency_fund=True,
        financial_commitments_gbp=10_000,
        loss_tolerance_percent=10,
    )

    capacity, consistency, suitability = run_full_assessment(profile)

    assert capacity.level == RiskLevel.MEDIUM
    assert consistency.status == ConsistencyStatus.REVIEW_REQUIRED
    assert suitability.decision == SuitabilityDecision.REVIEW_REQUIRED
    assert suitability.effective_risk_level == RiskLevel.MEDIUM
    assert suitability.requires_human_review is True


def test_high_capacity_does_not_override_low_customer_tolerance():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.LOW,
        investment_horizon_years=15,
        liquidity_need=LiquidityNeed.LOW,
        emergency_fund=True,
        financial_commitments_gbp=5_000,
        loss_tolerance_percent=25,
    )

    capacity, consistency, suitability = run_full_assessment(profile)

    assert capacity.level == RiskLevel.HIGH
    assert consistency.status == ConsistencyStatus.CONSISTENT
    assert suitability.decision == SuitabilityDecision.SUITABLE
    assert suitability.effective_risk_level == RiskLevel.LOW


def test_consistent_low_capacity_profile_stays_low_risk():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.LOW,
        investment_horizon_years=2,
        liquidity_need=LiquidityNeed.HIGH,
        emergency_fund=False,
        financial_commitments_gbp=35_000,
        loss_tolerance_percent=2,
    )

    capacity, consistency, suitability = run_full_assessment(profile)

    assert capacity.level == RiskLevel.LOW
    assert consistency.status == ConsistencyStatus.CONSISTENT
    assert suitability.decision == SuitabilityDecision.SUITABLE
    assert suitability.effective_risk_level == RiskLevel.LOW


def test_conflict_does_not_produce_effective_risk_level():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=1,
        liquidity_need=LiquidityNeed.HIGH,
        loss_tolerance_percent=2,
    )

    _, _, suitability = run_full_assessment(profile)

    assert suitability.decision == SuitabilityDecision.REVIEW_REQUIRED
    assert suitability.effective_risk_level is None
