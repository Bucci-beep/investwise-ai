"""Core data models for the InvestWise AI decision engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExperienceLevel(str, Enum):
    NONE = "none"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERIENCED = "experienced"


class LiquidityNeed(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SuitabilityStatus(str, Enum):
    SUITABLE = "suitable"
    REVIEW_REQUIRED = "review_required"
    UNSUITABLE = "unsuitable"


@dataclass
class CustomerProfile:
    """Information supplied to the investment suitability engine."""

    age: int
    investment_horizon_years: int
    annual_income_gbp: float
    emergency_fund: bool
    investment_experience: ExperienceLevel
    liquidity_need: LiquidityNeed
    stated_risk_tolerance: RiskLevel

    goal: Optional[str] = None
    financial_commitments_gbp: Optional[float] = None
    loss_tolerance_percent: Optional[float] = None


@dataclass
class RiskAssessment:
    """Risk assessment produced from a customer profile."""

    risk_capacity: RiskLevel
    stated_risk_tolerance: RiskLevel
    warnings: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)


@dataclass
class SuitabilityResult:
    """Final output returned by the suitability engine."""

    status: SuitabilityStatus
    risk_assessment: RiskAssessment
    confidence: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
