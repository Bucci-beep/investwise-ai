"""End-to-end tests for the InvestWise assessment pipeline."""

from ai.consistency import ConsistencyStatus
from ai.models import (
    CustomerProfile,
    ExperienceLevel,
    LiquidityNeed,
    RiskLevel,
)
from ai.pipeline import run_assessment
from ai.suitability import SuitabilityDecision


def make_profile(**overrides) -> CustomerProfile:
    """Create a baseline customer profile for pipeline testing."""

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


def test_pipeline_returns_complete_assessment():
    profile = make_profile()

    result = run_assessment(profile)

    assert result.profile == profile
    assert result.risk_capacity is not None
    assert result.consistency is not None
    assert result.suitability is not None
    assert result.explanation is not None


def test_pipeline_handles_consistent_customer():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.MEDIUM,
        investment_horizon_years=5,
        liquidity_need=LiquidityNeed.MEDIUM,
        emergency_fund=True,
        loss_tolerance_percent=10,
    )

    result = run_assessment(profile)

    assert result.risk_capacity.level == RiskLevel.MEDIUM
    assert result.consistency.status == ConsistencyStatus.CONSISTENT
    assert result.suitability.decision == SuitabilityDecision.SUITABLE
    assert result.suitability.effective_risk_level == RiskLevel.MEDIUM
    assert "can proceed" in result.explanation.summary


def test_pipeline_routes_material_conflict_to_review():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.HIGH,
        investment_horizon_years=1,
        liquidity_need=LiquidityNeed.HIGH,
        emergency_fund=False,
        financial_commitments_gbp=40_000,
        loss_tolerance_percent=2,
    )

    result = run_assessment(profile)

    assert result.risk_capacity.level == RiskLevel.LOW
    assert result.consistency.status == ConsistencyStatus.CONFLICT
    assert (
        result.suitability.decision
        == SuitabilityDecision.REVIEW_REQUIRED
    )
    assert result.suitability.requires_human_review is True
    assert result.suitability.effective_risk_level is None
    assert "requires review" in result.explanation.summary


def test_pipeline_preserves_customer_lower_risk_preference():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.LOW,
        investment_horizon_years=15,
        liquidity_need=LiquidityNeed.LOW,
        emergency_fund=True,
        financial_commitments_gbp=5_000,
        loss_tolerance_percent=25,
    )

    result = run_assessment(profile)

    assert result.risk_capacity.level == RiskLevel.HIGH
    assert result.consistency.status == ConsistencyStatus.CONSISTENT
    assert result.suitability.decision == SuitabilityDecision.SUITABLE
    assert result.suitability.effective_risk_level == RiskLevel.LOW
    assert "low" in result.explanation.summary


def test_pipeline_output_keeps_traceable_evidence():
    profile = make_profile(
        stated_risk_tolerance=RiskLevel.LOW,
        investment_horizon_years=2,
        liquidity_need=LiquidityNeed.HIGH,
        emergency_fund=False,
        loss_tolerance_percent=2,
    )

    result = run_assessment(profile)

    assert len(result.risk_capacity.explanations) > 0
    assert len(result.explanation.key_factors) > 0
    assert "Short investment horizon." in result.explanation.warnings
    assert "High liquidity requirement." in result.explanation.warnings
    assert "No emergency fund reported." in result.explanation.warnings
