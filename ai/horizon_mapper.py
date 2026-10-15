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

    unit_divisors = {"year": 1.0, "month": 12.0, "week": 365.25 / 7.0, "day": 365.25}

    def years(value: float, unit: str) -> float:
        return value / unit_divisors[unit.rstrip("s")]

    # A smaller-unit qualifier is part of the same duration, e.g. three
    # years plus one day. Do not reduce it to exactly three years, or treat
    # the day as a second competing horizon.
    compound_pattern = re.compile(
        rf"\b({number_token})\s*(years?|months?|weeks?)"
        rf"\s*(?:plus|and|\+)\s*({number_token})\s*(months?|weeks?|days?)\b"
    )
    for match in compound_pattern.finditer(normalised):
        first_unit, second_unit = match.group(2), match.group(4)
        if unit_divisors[first_unit.rstrip("s")] >= unit_divisors[second_unit.rstrip("s")]:
            continue
        values.append(years(_parse_number(match.group(1)), first_unit) + years(_parse_number(match.group(3)), second_unit))
        consumed_spans.append(match.span())

    pair_pattern = re.compile(
        rf"\b(?:between\s+)?"
        rf"({number_token})"
        rf"(?:\s+(?:and|or|to|through)\s+|\s*[-–—]\s*)"
        rf"({number_token})"
        rf"\s*(years?|months?|weeks?|days?)\b"
    )

    for match in pair_pattern.finditer(normalised):
        if any(match.start() >= start and match.end() <= end for start, end in consumed_spans):
            continue
        first = _parse_number(match.group(1))
        second = _parse_number(match.group(2))
        unit = match.group(3)

        values.extend(
            [
                years(first, unit),
                years(second, unit),
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
        rf"\b({number_token})\s*(years?|months?|weeks?|days?)\b"
    )

    for match in single_pattern.finditer(normalised):
        if inside_consumed_span(*match.span()):
            continue

        value = _parse_number(match.group(1))
        unit = match.group(2)

        values.append(years(value, unit))

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


def extract_supported_need_clause(text: str) -> str | None:
    """Separate an explicit actual need from an ideal or superseded horizon.

    This is premise separation, not a rule to choose the smallest number.
    Two competing actual needs or a conditional/unknown need remain intact
    for ordinary clarification. The caller retains the original full reply.
    """
    normalised = text.replace("’", "'")
    need = re.compile(
        r"\b(?:earliest(?: realistic)? need|first realistic need|"
        r"i (?:(?:will|realistically|actually|now|may realistically) )?need|"
        r"must (?:use|withdraw)|have to (?:use|withdraw))\b", re.I,
    )
    uncertainty = re.compile(
        r"\b(?:if|unless|maybe|perhaps|possibly|not sure|don't know|do not know|"
        r"cannot determine|can't determine|depends|depending|either|or after)\b", re.I,
    )
    goal = re.compile(r"\b(?:ideal\w*|want|would like|hope|wish|aim|goal|plan|intend)\b", re.I)
    correction = re.search(r"\b(?:correction|actually|i meant|i mean|replace|instead of|rather than)\b", normalised, re.I)
    clauses = [
        part.strip(" ,;.") for part in re.split(
            r"\b(?:but|however|although|even though|yet|while)\b|[.!?;]\s*", normalised, flags=re.I,
        ) if part.strip(" ,;.")
    ]
    actual = [part for part in clauses if need.search(part)]
    if len(actual) != 1:
        return None
    clause = actual[0]
    if uncertainty.search(clause):
        return None
    excluded = [part for part in clauses if part != clause]
    if correction:
        # Remove only an expressly superseded duration after the supported
        # need, rather than ignoring additional actual timing statements.
        clause = re.split(r"\b(?:replace|instead of|rather than)\b", clause, maxsplit=1, flags=re.I)[0].strip(" ,;.")
        excluded_is_superseded = all(
            goal.search(part) or re.search(r"\b(?:replace|instead of|rather than)\b", part, re.I)
            or not _extract_year_values(part)
            for part in excluded
        )
        if not excluded_is_superseded:
            return None
    elif not excluded or not any(goal.search(part) and _extract_year_values(part) for part in excluded):
        return None
    # A need clause spanning financial boundaries is still unresolved.
    if map_explicit_horizon(clause).status != HorizonMappingStatus.MAPPED:
        return None
    return clause
