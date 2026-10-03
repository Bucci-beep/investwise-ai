from __future__ import annotations

import re
from dataclasses import dataclass

from ai.question_spec import BusinessSpec


@dataclass(frozen=True)
class DeterministicResolution:
    candidate_option_id: str | None = None
    reason: str | None = None
    source: str | None = None
    independent_flags: tuple[str, ...] = ()


_ALL_SAVINGS = re.compile(r"\b(all|every)\s+(?:of\s+)?my\s+savings\b", re.I)


def detect_d12_independent_flags(text: str) -> set[str]:
    """Detect only high precision independent D12 flags.

    This function never maps a D12 financial category. Explicit horizon
    mapping remains inside the existing BoundedQuestionInterpreter after
    the semantic/conformal clarity gate.
    """
    flags: set[str] = set()

    if _ALL_SAVINGS.search(text):
        flags.add("same_money_dependency")

    return flags


def match_fixed_choice(
    spec: BusinessSpec,
    question_id: str,
    text: str,
) -> DeterministicResolution | None:
    option = spec.match_exact_choice(question_id, text)

    if option is None:
        return None

    return DeterministicResolution(
        candidate_option_id=option.id,
        reason="exact_fixed_choice",
        source="fixed_choice",
    )
