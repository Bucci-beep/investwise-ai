"""Dataset utilities for the InvestWise bounded dialogue layer.

The business handoff workbook is the authoritative source for authored
synthetic governance fixtures.

These fixtures are kept separate from model-generated or expanded NLP
training data so that governance examples are not accidentally treated
as independent classifier evaluation evidence.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_WORKBOOK = Path(
    "data/business/challenge-2-business-handoff.xlsx"
)


@dataclass(frozen=True)
class DialogueFixture:
    fixture_id: str
    question: str
    reply_group: str
    text: str
    ordinary_bucket: str
    candidate_ids: tuple[str, ...]
    domain: str | None
    independent_flags: tuple[str, ...]


def _clean(value: object) -> str:
    """Convert workbook values to clean strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def _parse_candidate_cell(
    value: object,
) -> tuple[tuple[str, ...], str | None]:
    """Parse candidate IDs and domain from the workbook cell.

    Example:

        D12_SHORT
        horizon

    becomes:

        ("D12_SHORT",), "horizon"

    A cell containing only the domain produces no candidate ID.
    """

    raw = _clean(value)

    if not raw:
        return (), None

    parts = [
        part.strip()
        for part in raw.splitlines()
        if part.strip()
    ]

    candidates = tuple(
        part
        for part in parts
        if part.startswith("D")
    )

    domains = [
        part
        for part in parts
        if not part.startswith("D")
    ]

    domain = domains[-1] if domains else None

    return candidates, domain


def _parse_flags(value: object) -> tuple[str, ...]:
    """Parse independent governance flags."""

    raw = _clean(value)

    if not raw:
        return ()

    return tuple(
        flag.strip()
        for flag in raw.split(",")
        if flag.strip()
    )


def load_examples(
    workbook: Path | str = DEFAULT_WORKBOOK,
) -> pd.DataFrame:
    """Load the authored example bank using its actual header row."""

    return pd.read_excel(
        workbook,
        sheet_name="Examples",
        header=4,
    )


def load_question_fixtures(
    question: str,
    workbook: Path | str = DEFAULT_WORKBOOK,
) -> list[DialogueFixture]:
    """Return structured fixtures for one bounded question."""

    df = load_examples(workbook)

    rows = df[
        df["Question"].astype(str).str.strip() == question
    ]

    fixtures: list[DialogueFixture] = []

    for _, row in rows.iterrows():
        candidates, domain = _parse_candidate_cell(
            row["Candidate IDs and domain"]
        )

        fixtures.append(
            DialogueFixture(
                fixture_id=_clean(row["Fixture ID"]),
                question=_clean(row["Question"]),
                reply_group=_clean(row["Reply group"]),
                text=_clean(row["Free-text reply"]),
                ordinary_bucket=_clean(row["Ordinary bucket"]),
                candidate_ids=candidates,
                domain=domain,
                independent_flags=_parse_flags(
                    row["Independent flags"]
                ),
            )
        )

    return fixtures
