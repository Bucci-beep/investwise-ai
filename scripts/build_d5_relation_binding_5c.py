import csv
import hashlib
import json
from pathlib import Path


OUTPUT_DIR = Path("data/nlp/experiments/d5_relation_binding")
OUTPUT_CSV = OUTPUT_DIR / "d5_relation_binding_training.csv"
PROVENANCE_JSON = OUTPUT_DIR / "provenance.json"


# Each sentence deliberately contains TWO identifiable sources.
# Exactly one source actively funds current living costs and the other
# is explicitly inactive.
#
# Each case becomes two training rows:
#   TARGET=<active source>   -> active
#   TARGET=<inactive source> -> inactive
#
# This is designed to teach source-specific relation binding rather
# than simple source vocabulary recognition.

CASES = [
    # salary + benefits_support
    {
        "id": "RB-01",
        "text": (
            "My wages are kept aside, while benefit payments "
            "currently cover my household expenses."
        ),
        "active": "benefits_support",
        "inactive": "salary",
    },
    {
        "id": "RB-02",
        "text": (
            "Benefits come in but I save them; my salary is what "
            "currently pays the regular bills."
        ),
        "active": "salary",
        "inactive": "benefits_support",
    },
    {
        "id": "RB-03",
        "text": (
            "I earn wages but do not spend them on everyday costs; "
            "regular family support pays those expenses."
        ),
        "active": "benefits_support",
        "inactive": "salary",
    },
    {
        "id": "RB-04",
        "text": (
            "Family support is put aside rather than spent, and my "
            "employment income covers my normal living costs."
        ),
        "active": "salary",
        "inactive": "benefits_support",
    },

    # salary + pension_payments
    {
        "id": "RB-05",
        "text": (
            "My salary goes into savings, while the pension payments "
            "I receive are used for my everyday expenses."
        ),
        "active": "pension_payments",
        "inactive": "salary",
    },
    {
        "id": "RB-06",
        "text": (
            "I receive pension income but keep it untouched; my wages "
            "currently pay the household bills."
        ),
        "active": "salary",
        "inactive": "pension_payments",
    },
    {
        "id": "RB-07",
        "text": (
            "Employment income still comes in but is not used for "
            "living costs; my pension payments cover them."
        ),
        "active": "pension_payments",
        "inactive": "salary",
    },
    {
        "id": "RB-08",
        "text": (
            "The pension money is saved rather than spent, whereas "
            "my salary funds my current day-to-day expenses."
        ),
        "active": "salary",
        "inactive": "pension_payments",
    },

    # salary + other_regular
    {
        "id": "RB-09",
        "text": (
            "My wages are not used for household spending; the rental "
            "income I receive currently pays those costs."
        ),
        "active": "other_regular",
        "inactive": "salary",
    },
    {
        "id": "RB-10",
        "text": (
            "Rental receipts stay in the property account, while my "
            "salary pays my regular living expenses."
        ),
        "active": "salary",
        "inactive": "other_regular",
    },
    {
        "id": "RB-11",
        "text": (
            "I keep my employment income separate from household "
            "spending; recurring royalties cover my everyday costs."
        ),
        "active": "other_regular",
        "inactive": "salary",
    },
    {
        "id": "RB-12",
        "text": (
            "My licence income is saved rather than used for bills, "
            "and my wages currently fund my living costs."
        ),
        "active": "salary",
        "inactive": "other_regular",
    },

    # salary + withdrawals
    {
        "id": "RB-13",
        "text": (
            "My salary is saved each month, while withdrawals from "
            "my investments pay my regular expenses."
        ),
        "active": "withdrawals",
        "inactive": "salary",
    },
    {
        "id": "RB-14",
        "text": (
            "I make savings withdrawals but keep that money aside; "
            "my wages are what pay the household bills."
        ),
        "active": "salary",
        "inactive": "withdrawals",
    },
    {
        "id": "RB-15",
        "text": (
            "Employment income does not go towards my living costs; "
            "money drawn from savings currently covers them."
        ),
        "active": "withdrawals",
        "inactive": "salary",
    },
    {
        "id": "RB-16",
        "text": (
            "The money I withdraw from investments is reinvested, "
            "whereas my salary funds my everyday spending."
        ),
        "active": "salary",
        "inactive": "withdrawals",
    },

    # salary + self_employment
    {
        "id": "RB-17",
        "text": (
            "I still receive wages but save them; earnings from my "
            "freelance work currently pay my living costs."
        ),
        "active": "self_employment",
        "inactive": "salary",
    },
    {
        "id": "RB-18",
        "text": (
            "My freelance income stays in the business, while my "
            "salary covers my normal household expenses."
        ),
        "active": "salary",
        "inactive": "self_employment",
    },
    {
        "id": "RB-19",
        "text": (
            "The wages from my job are kept aside, and payments from "
            "my own business fund my day-to-day costs."
        ),
        "active": "self_employment",
        "inactive": "salary",
    },
    {
        "id": "RB-20",
        "text": (
            "Client earnings are reinvested rather than spent on "
            "living costs; my employment income pays those bills."
        ),
        "active": "salary",
        "inactive": "self_employment",
    },

    # Non-salary pairings
    {
        "id": "RB-21",
        "text": (
            "My pension payments are saved, while rental income "
            "currently covers my household expenses."
        ),
        "active": "other_regular",
        "inactive": "pension_payments",
    },
    {
        "id": "RB-22",
        "text": (
            "I receive benefits but keep that money aside; withdrawals "
            "from savings currently pay my living costs."
        ),
        "active": "withdrawals",
        "inactive": "benefits_support",
    },
    {
        "id": "RB-23",
        "text": (
            "Freelance earnings are retained in the business, while "
            "my pension income funds my everyday spending."
        ),
        "active": "pension_payments",
        "inactive": "self_employment",
    },
    {
        "id": "RB-24",
        "text": (
            "Rental receipts are saved rather than spent, and regular "
            "family support currently pays my household bills."
        ),
        "active": "benefits_support",
        "inactive": "other_regular",
    },
        {
        "id": "RB-25",
        "text": (
            "My freelance earnings stay in the business, while benefit "
            "payments currently cover my regular household costs."
        ),
        "active": "benefits_support",
        "inactive": "self_employment",
    },
    {
        "id": "RB-26",
        "text": (
            "Benefits are saved rather than spent, while income from my "
            "own business pays my everyday expenses."
        ),
        "active": "self_employment",
        "inactive": "benefits_support",
    },
    {
        "id": "RB-27",
        "text": (
            "My pension payments are kept aside, while withdrawals from "
            "investments currently fund my living costs."
        ),
        "active": "withdrawals",
        "inactive": "pension_payments",
    },
    {
        "id": "RB-28",
        "text": (
            "Money withdrawn from savings is reinvested, while my pension "
            "income currently pays the household bills."
        ),
        "active": "pension_payments",
        "inactive": "withdrawals",
    },
    {
        "id": "RB-29",
        "text": (
            "Rental income stays in the property account, while earnings "
            "from my freelance work cover my normal living expenses."
        ),
        "active": "self_employment",
        "inactive": "other_regular",
    },
    {
        "id": "RB-30",
        "text": (
            "My business earnings are retained rather than spent, while "
            "the rent I receive currently pays my household costs."
        ),
        "active": "other_regular",
        "inactive": "self_employment",
    },
    {
        "id": "RB-31",
        "text": (
            "Regular family support is put aside, while rental income "
            "currently funds my day-to-day spending."
        ),
        "active": "other_regular",
        "inactive": "benefits_support",
    },
    {
        "id": "RB-32",
        "text": (
            "The rent I receive is saved rather than spent, while benefit "
            "payments currently cover my normal living expenses."
        ),
        "active": "benefits_support",
        "inactive": "other_regular",
    },
    {
        "id": "RB-33",
        "text": (
            "My freelance income is kept in the business, while money "
            "drawn from savings currently pays my regular bills."
        ),
        "active": "withdrawals",
        "inactive": "self_employment",
    },
    {
        "id": "RB-34",
        "text": (
            "Withdrawals from investments are kept aside, while earnings "
            "from my own business currently fund my living costs."
        ),
        "active": "self_employment",
        "inactive": "withdrawals",
    },
]


VALID_SOURCES = {
    "salary",
    "self_employment",
    "pension_payments",
    "benefits_support",
    "withdrawals",
    "other_regular",
}


def build():
    if len(CASES) != 34:
        raise ValueError(f"Expected 24 cases, found {len(CASES)}")

    case_ids = [case["id"] for case in CASES]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs")

    normalized = [
        " ".join(case["text"].lower().split())
        for case in CASES
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate normalized case text")

    rows = []

    for case in CASES:
        active = case["active"]
        inactive = case["inactive"]

        if active not in VALID_SOURCES:
            raise ValueError(
                f'{case["id"]}: invalid active source {active}'
            )

        if inactive not in VALID_SOURCES:
            raise ValueError(
                f'{case["id"]}: invalid inactive source {inactive}'
            )

        if active == inactive:
            raise ValueError(
                f'{case["id"]}: active and inactive source match'
            )

        for source, attribution in [
            (active, "active"),
            (inactive, "inactive"),
        ]:
            rows.append(
                {
                    "id": f'{case["id"]}-{source}',
                    "case_id": case["id"],
                    "question_id": "D5",
                    "source": source,
                    "attribution": attribution,
                    "text": case["text"],
                    "split": "train",
                    "origin": (
                        "fictional_authored_d5_relation_binding_5c"
                    ),
                }
            )

    if len(rows) != 68:
        raise ValueError(f"Expected 48 rows, found {len(rows)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "case_id",
        "question_id",
        "source",
        "attribution",
        "text",
        "split",
        "origin",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(
        OUTPUT_CSV.read_bytes()
    ).hexdigest()

    provenance = {
        "experiment": "D5 Experiment 5C",
        "purpose": (
            "Train explicit source-specific relation binding using "
            "paired responses containing one active and one inactive "
            "funding source."
        ),
        "case_count": 34,
        "training_row_count": 68,
        "rows_per_case": 2,
        "sha256": digest,
        "design": {
            "salary_benefits": 4,
            "salary_pension": 4,
            "salary_other_regular": 4,
            "salary_withdrawals": 4,
            "salary_self_employment": 4,
            "non_salary_pairings": 4,
            "additional_non_salary_pairings": 10,
        },
        "limitations": [
            (
                "Synthetic authored development data is not evidence "
                "of real-customer performance."
            ),
            (
                "The dataset was designed after observing source "
                "attribution failures in earlier D5 experiments."
            ),
            (
                "Observed development and stress sets are not "
                "independent final validation."
            ),
            (
                "The dataset tests targeted relation binding and "
                "does not establish conformal coverage or deployment "
                "performance."
            ),
        ],
    }

    PROVENANCE_JSON.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(CASES)} paired cases")
    print(f"Wrote {len(rows)} training rows to {OUTPUT_CSV}")
    print(f"Wrote provenance to {PROVENANCE_JSON}")
    print(f"SHA256: {digest}")

    print("\nAttribution counts:")
    for label in ("active", "inactive"):
        count = sum(
            row["attribution"] == label
            for row in rows
        )
        print(f"  {label}: {count}")

    print("\nTarget-source counts:")
    for source in sorted(VALID_SOURCES):
        count = sum(
            row["source"] == source
            for row in rows
        )
        print(f"  {source}: {count}")


if __name__ == "__main__":
    build()
