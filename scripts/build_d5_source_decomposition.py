import csv
from pathlib import Path


V1_PATH = Path("data/nlp/all_questions/train.csv")
V2_PATH = Path("data/nlp/experiments/d5_v2/d5_training_additions.csv")

OUTPUT_DIR = Path("data/nlp/experiments/d5_source_decomposition")
OUTPUT_PATH = OUTPUT_DIR / "d5_source_training.csv"

SOURCE_COLUMNS = [
    "salary",
    "self_employment",
    "pension_payments",
    "benefits_support",
    "withdrawals",
    "other_regular",
    "no_regular",
]

SINGLE_OPTION_TO_SOURCE = {
    "D5_SALARY": "salary",
    "D5_SELF_EMPLOYMENT": "self_employment",
    "D5_PENSION_PAYMENTS": "pension_payments",
    "D5_BENEFITS_SUPPORT": "benefits_support",
    "D5_WITHDRAWALS": "withdrawals",
    "D5_OTHER_REGULAR": "other_regular",
    "D5_NO_REGULAR": "no_regular",
}

# Explicitly audited from the 12 existing D5_MULTIPLE training examples.
# These mappings are not inferred automatically from text.
MULTIPLE_SOURCE_MAP = {
    "SYNTH-V1-D5-TRAIN-043": {"salary", "withdrawals"},
    "SYNTH-V1-D5-TRAIN-044": {"salary", "benefits_support"},
    "SYNTH-V1-D5-TRAIN-045": {"pension_payments", "other_regular"},
    "SYNTH-V1-D5-TRAIN-046": {"self_employment", "benefits_support"},
    "SYNTH-V1-D5-TRAIN-047": {"salary", "other_regular"},
    "SYNTH-V1-D5-TRAIN-048": {"pension_payments", "withdrawals"},
    "SYNTH-D5-V2-043": {"salary", "other_regular"},
    "SYNTH-D5-V2-044": {"salary", "withdrawals"},
    "SYNTH-D5-V2-045": {"pension_payments", "withdrawals"},
    "SYNTH-D5-V2-046": {"pension_payments", "other_regular"},
    "SYNTH-D5-V2-047": {"self_employment", "benefits_support"},
    "SYNTH-D5-V2-048": {"salary", "withdrawals"},
}


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_clarity_rows():
    rows = load_csv(V1_PATH) + load_csv(V2_PATH)

    return [
        row
        for row in rows
        if (
            row["question_id"] == "D5"
            and row["bucket"] == "clarity"
            and row["option_id"]
        )
    ]


def active_sources(row):
    option = row["option_id"]

    if option in SINGLE_OPTION_TO_SOURCE:
        return {SINGLE_OPTION_TO_SOURCE[option]}

    if option == "D5_MULTIPLE":
        if row["id"] not in MULTIPLE_SOURCE_MAP:
            raise ValueError(
                f'Missing audited MULTIPLE mapping for {row["id"]}'
            )
        return MULTIPLE_SOURCE_MAP[row["id"]]

    raise ValueError(
        f'Unexpected D5 clarity option {option!r} for {row["id"]}'
    )


def validate(rows):
    if len(rows) != 96:
        raise ValueError(f"Expected 96 clarity rows, found {len(rows)}")

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training IDs found")

    multiple_ids = {
        row["id"]
        for row in rows
        if row["option_id"] == "D5_MULTIPLE"
    }

    if multiple_ids != set(MULTIPLE_SOURCE_MAP):
        missing = multiple_ids - set(MULTIPLE_SOURCE_MAP)
        extra = set(MULTIPLE_SOURCE_MAP) - multiple_ids
        raise ValueError(
            f"MULTIPLE mapping mismatch. Missing={missing}, extra={extra}"
        )


def build():
    rows = load_clarity_rows()
    validate(rows)

    output_rows = []

    for row in rows:
        active = active_sources(row)

        # NO_REGULAR is mutually exclusive with positive funding sources.
        if "no_regular" in active and len(active) != 1:
            raise ValueError(
                f'Invalid NO_REGULAR combination for {row["id"]}'
            )

        output = {
            "id": row["id"],
            "question_id": "D5",
            "text": row["text"],
            "original_option": row["option_id"],
            "group_id": row.get("group_id", ""),
        }

        for source in SOURCE_COLUMNS:
            output[source] = int(source in active)

        output_rows.append(output)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question_id",
        "text",
        "original_option",
        "group_id",
        *SOURCE_COLUMNS,
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {OUTPUT_PATH}")

    print("\nPositive-label counts:")
    for source in SOURCE_COLUMNS:
        count = sum(int(row[source]) for row in output_rows)
        print(f"  {source}: {count}")


if __name__ == "__main__":
    build()
