"""Audit NLP development datasets against InvestWise responsibility boundaries.

The purpose of this module is to catch examples that should not be used
as ordinary ML targets because they represent deterministic controls or
governance conditions.

This is a development-time audit. It does not perform runtime financial
decision making.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AuditFinding:
    dataset: str
    row_number: int
    text: str
    label: str
    finding: str
    severity: str


CONTROL_PATTERNS = {
    "possible_refusal": (
        "prefer not to",
        "rather not",
        "do not want to answer",
        "don't want to answer",
        "not comfortable giving",
        "refuse to answer",
    ),
    "possible_pause": (
        "pause",
        "stop for now",
        "come back to this",
        "continue later",
    ),
}


GOVERNANCE_PATTERNS = {
    "possible_same_money_dependency": (
        "all my savings",
        "same money",
        "same pot",
        "same included pot",
    ),
    "possible_material_conflict": (
        "but i might need",
        "but i may need",
        "but i could need",
        "but i need",
    ),
}


def audit_intent_dataset(
    path: str | Path,
) -> list[AuditFinding]:
    """Flag intent examples that may belong outside ordinary ML labels."""

    path = Path(path)
    df = pd.read_csv(path)

    findings: list[AuditFinding] = []

    for index, row in df.iterrows():
        text = str(row["text"])
        label = str(row["label"])
        normalised = text.lower()

        for finding, patterns in CONTROL_PATTERNS.items():
            if any(pattern in normalised for pattern in patterns):
                findings.append(
                    AuditFinding(
                        dataset=path.name,
                        row_number=index + 2,
                        text=text,
                        label=label,
                        finding=finding,
                        severity="review",
                    )
                )

        for finding, patterns in GOVERNANCE_PATTERNS.items():
            if any(pattern in normalised for pattern in patterns):
                findings.append(
                    AuditFinding(
                        dataset=path.name,
                        row_number=index + 2,
                        text=text,
                        label=label,
                        finding=finding,
                        severity="review",
                    )
                )

    return findings


def audit_option_dataset(
    path: str | Path,
) -> list[AuditFinding]:
    """Flag option examples containing likely governance conditions."""

    path = Path(path)
    df = pd.read_csv(path)

    findings: list[AuditFinding] = []

    for index, row in df.iterrows():
        text = str(row["text"])
        label = str(row["label"])
        normalised = text.lower()

        for finding, patterns in CONTROL_PATTERNS.items():
            if any(pattern in normalised for pattern in patterns):
                findings.append(
                    AuditFinding(
                        dataset=path.name,
                        row_number=index + 2,
                        text=text,
                        label=label,
                        finding=finding,
                        severity="block",
                    )
                )

        for finding, patterns in GOVERNANCE_PATTERNS.items():
            if any(pattern in normalised for pattern in patterns):
                findings.append(
                    AuditFinding(
                        dataset=path.name,
                        row_number=index + 2,
                        text=text,
                        label=label,
                        finding=finding,
                        severity="review",
                    )
                )

    return findings
