"""Deterministic mapping for explicit D12 time-horizon statements.

This module handles explicit durations only.

It does not determine semantic intent, customer uncertainty,
vulnerability, or financial suitability.
"""

from dataclasses import dataclass
from enum import Enum
import re


class HorizonMappingStatus(str, Enum):
    MAPPED = "mapped"
    AMBIGUOUS = "ambiguous"
    NOT_EXPLICIT = "not_explicit"


@dataclass(frozen=True)
class HorizonMapping:
    status: HorizonMappingStatus
    category: str | None
    durations_years: tuple[float, ...]
    reason: str


_NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}


def _category(years: float) -> str:
    if years <= 3:
        return "D12_SHORT"

    if years <= 10:
        return "D12_MEDIUM"

    return "D12_LONG"


def _parse_number(token: str) -> float:
    token = token.lower()

    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]

    return float(token)


def _extract_year_values(text: str) -> list[float]:
    """Extract explicit durations and convert them to years."""

    normalised = text.lower()

    number_words = "|".join(_NUMBER_WORDS.keys())
    number_token = rf"(?:\d+(?:\.\d+)?|{number_words})"

    values: list[float] = []
    consumed_spans: list[tuple[int, int]] = []

    pair_pattern = re.compile(
        rf"\b(?:between\s+)?"
        rf"({number_token})"
        rf"\s+(?:and|or|to|through)\s+"
        rf"({number_token})"
        rf"\s*(years?|months?)\b"
    )

    for match in pair_pattern.finditer(normalised):
        first = _parse_number(match.group(1))
        second = _parse_number(match.group(2))
        unit = match.group(3)

        divisor = 12.0 if unit.startswith("month") else 1.0

        values.extend(
            [
                first / divisor,
                second / divisor,
            ]
        )

        consumed_spans.append(match.span())

    def inside_consumed_span(
        start: int,
        end: int,
    ) -> bool:
        return any(
            start >= consumed_start
            and end <= consumed_end
            for consumed_start, consumed_end in consumed_spans
        )

    single_pattern = re.compile(
        rf"\b({number_token})\s*(years?|months?)\b"
    )

    for match in single_pattern.finditer(normalised):
        if inside_consumed_span(*match.span()):
            continue

        value = _parse_number(match.group(1))
        unit = match.group(2)

        if unit.startswith("month"):
            value /= 12.0

        values.append(value)

    if re.search(r"\bnext year\b", normalised):
        values.append(1.0)

    return values


def map_explicit_horizon(
    text: str,
) -> HorizonMapping:
    """Map explicit durations to the bounded D12 category.

    Multiple durations are accepted only when every extracted duration
    belongs to the same D12 category.
    """

    values = _extract_year_values(text)

    if not values:
        return HorizonMapping(
            status=HorizonMappingStatus.NOT_EXPLICIT,
            category=None,
            durations_years=(),
            reason="no explicit supported duration found",
        )

    categories = {
        _category(value)
        for value in values
    }

    if len(categories) != 1:
        return HorizonMapping(
            status=HorizonMappingStatus.AMBIGUOUS,
            category=None,
            durations_years=tuple(values),
            reason="explicit durations cross D12 category boundaries",
        )

    category = next(iter(categories))

    return HorizonMapping(
        status=HorizonMappingStatus.MAPPED,
        category=category,
        durations_years=tuple(values),
        reason="explicit durations map to one D12 category",
    )
