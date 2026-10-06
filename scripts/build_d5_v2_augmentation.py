#!/usr/bin/env python3
"""Build the fictional D5-only v2 training augmentation.

The utterances in this module are newly authored development examples.  The
generator reads the v1 training split only to reject exact normalized overlap;
it does not read calibration or test data and does not integrate the output
into model training or runtime behaviour.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "nlp" / "experiments" / "d5_v2"
OUTPUT = DESTINATION / "d5_training_additions.csv"
PROVENANCE = DESTINATION / "provenance.json"
V1_TRAIN = ROOT / "data" / "nlp" / "all_questions" / "train.csv"
FIELDS = ("id", "question_id", "text", "bucket", "option_id", "source", "group_id", "split")
SOURCE = "fictional_authored_d5_v2_experiment_2026-10-05"
CREATION_DATE = "2026-10-05"

OPTION_COUNTS = {
    "D5_SALARY": 6,
    "D5_SELF_EMPLOYMENT": 6,
    "D5_PENSION_PAYMENTS": 6,
    "D5_BENEFITS_SUPPORT": 6,
    "D5_WITHDRAWALS": 6,
    "D5_OTHER_REGULAR": 6,
    "D5_NO_REGULAR": 6,
    "D5_MULTIPLE": 6,
}
BUCKET_COUNTS = {"clarity": 48, "undecided": 8, "confusion": 8}

# Values are (semantic family, utterance). Repeated family names deliberately
# keep closely related constructions together for any future group-aware split.
CLARITY = {
    "D5_SALARY": [
        ("wages_direct", "My wages cover the rent, food and the usual monthly bills."),
        ("wages_direct", "Just my pay packet — that's what keeps the household running."),
        ("employment_funding_contrast", "I do freelance bits too, but I don't use that money for living costs; my salary pays those."),
        ("employment_funding_contrast", "Having a rental flat sounds relevant, but its income stays in the property account. I live on my wages."),
        ("current_source_correction", "Used to draw from savings. These days it's my salary that pays for day-to-day life."),
        ("messy_salary", "wages atm... they pay my normal bills, yep (sorry, bit rushed)"),
    ],
    "D5_SELF_EMPLOYMENT": [
        ("business_drawings", "I pay my household expenses from the money I earn running my own plumbing business."),
        ("business_drawings", "The café's earnings are what I take home for groceries and bills."),
        ("contractor_current", "I'm self-employed now, and my client work funds my everyday living costs."),
        ("contractor_current", "My invoicing income pays the mortgage and day-to-day stuff — no salary these days."),
        ("past_salary_correction", "Not wages anymore; I left that job. It's earnings from my own design work that currently support me."),
        ("messy_self_employed", "sole trader income pays the bills rn... varies a bit but thats the source"),
    ],
    "D5_PENSION_PAYMENTS": [
        ("pension_income_direct", "My monthly pension payments pay for my regular household costs."),
        ("pension_income_direct", "I live on the pension income that lands in my account each month."),
        ("pension_holding_contrast", "I own a separate pension pot, but that's untouched; the pension already in payment covers my bills."),
        ("pension_holding_contrast", "It isn't merely a pension balance — I'm receiving payments from it and using them for everyday expenses."),
        ("retirement_change", "I lived on wages before retiring; now my pension payments fund day-to-day life."),
        ("messy_pension", "pension paymnts, monthly... thats my food / council tax money now"),
    ],
    "D5_BENEFITS_SUPPORT": [
        ("benefits_direct", "My benefits currently cover the rent and my normal living expenses."),
        ("benefits_direct", "Universal Credit is the money I use for food, utilities and the rest."),
        ("person_support", "My sister sends me money every month, and that regular help pays my living costs."),
        ("person_support", "I don't live off my occasional gig earnings; my partner's regular support covers the household bills."),
        ("benefits_and_support_one_category", "Benefits cover most things and my dad regularly helps with the rest of my living costs."),
        ("messy_support", "benefits + a regular bit from mum, thats how I manage the usual bills"),
    ],
    "D5_WITHDRAWALS": [
        ("savings_drawdown", "I transfer money out of my savings each month to pay my living expenses."),
        ("savings_drawdown", "At the moment I'm running the household by drawing down my cash savings."),
        ("investment_sales", "Regular withdrawals from my investment account fund the mortgage and groceries."),
        ("investment_sales", "I sell a small amount of my funds each month and use the proceeds for everyday costs."),
        ("income_not_used_contrast", "I still receive a tiny wage, but save all of it; withdrawals from investments actually pay my bills."),
        ("messy_withdrawals", "taking ££ out of savings for rent n food atm — that's what I'm living on"),
    ],
    "D5_OTHER_REGULAR": [
        ("rental_income", "The rent from my lodger is the regular income I use for household costs."),
        ("rental_income", "My living expenses come from monthly rent paid by tenants in my other property."),
        ("named_recurring_source", "Royalties from my back catalogue arrive regularly and pay my normal bills."),
        ("named_recurring_source", "A monthly annuity payment, separate from any pension, funds my day-to-day spending."),
        ("source_change_other", "I used to use my salary, but now recurring licence fees cover my living costs."),
        ("messy_other_regular", "tenant rent every month = groceries, power, all the regular stuff for me"),
    ],
    "D5_NO_REGULAR": [
        ("explicit_none", "I don't currently have any regular source paying my living costs."),
        ("explicit_none", "There is no recurring money for my everyday expenses at the moment."),
        ("one_off_not_regular", "No regular source — I'm getting by on a one-off cash gift for now."),
        ("one_off_not_regular", "Nothing comes in regularly for bills; I sold my bike once to cover this month."),
        ("unemployment_explicit_none", "I'm out of work, and to be explicit, no other regular source is funding my living costs."),
        ("messy_none", "none right now... no regular money for rent/food etc"),
    ],
    "D5_MULTIPLE": [
        ("salary_plus_other", "My wages pay part of the bills and rental income covers the rest."),
        ("salary_plus_other", "Both my salary and monthly investment withdrawals fund our everyday costs."),
        ("pension_plus_other", "I use pension payments for groceries and draw from savings for the household bills."),
        ("pension_plus_other", "Living costs come from two places now: my pension income and rent from a tenant."),
        ("self_employed_plus_support", "My business earnings cover most expenses, with benefits paying some regular bills too."),
        ("messy_mixed", "bit of wages + bit drawn from savings... both go on the day-to-day stuff"),
    ],
}

NON_CLARITY = {
    "undecided": [
        ("needs_records", "I'm not sure which money actually covers the regular bills; I'd need to check my statements."),
        ("needs_records", "Could I come back to this? I can't tell what source is paying for day-to-day costs."),
        ("between_sources_uncertain", "It might be savings or support from my family, but I genuinely don't know which is funding things now."),
        ("between_sources_uncertain", "I'm unsure whether the rent money or my wages is being used for household spending."),
        ("changing_unresolved", "Everything has just changed this week, so I don't yet know what will regularly cover my living costs."),
        ("changing_unresolved", "My old source stopped and the new arrangement isn't settled — not sure yet."),
        ("short_undecided", "Honestly, no idea which source to put."),
        ("short_undecided", "not sure atm, need to look"),
    ],
    "confusion": [
        ("asks_scope", "Do you mean what pays my essential bills, or every bit of money that comes into my account?"),
        ("asks_scope", "What counts as a regular living cost here?"),
        ("source_without_funding", "I have a job and a pension pot, but I don't understand which one you want me to mention."),
        ("source_without_funding", "There's money in savings, although I haven't said what pays the bills — is that what you're asking?"),
        ("pension_definition", "Does owning a pension count even though I don't receive payments from it?"),
        ("category_definition", "Are benefits and the money my partner sends treated as one choice or two?"),
        ("off_topic_misread", "Are you asking how much my monthly expenses add up to?"),
        ("off_topic_misread", "Sorry, is this about where my accounts are held? I'm confused."),
    ],
}


def normalize(text: str) -> str:
    """Normalize only case and whitespace for exact-overlap checks."""
    return re.sub(r"\s+", " ", text.strip().lower())


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sequence = 0

    def append(text: str, bucket: str, option_id: str, family: str) -> None:
        nonlocal sequence
        sequence += 1
        rows.append({
            "id": f"SYNTH-D5-V2-{sequence:03d}",
            "question_id": "D5",
            "text": text,
            "bucket": bucket,
            "option_id": option_id,
            "source": SOURCE,
            "group_id": f"D5:v2:{bucket}:{family}",
            "split": "train",
        })

    for option_id, examples in CLARITY.items():
        for family, text in examples:
            append(text, "clarity", option_id, f"{option_id}:{family}")
    for bucket, examples in NON_CLARITY.items():
        for family, text in examples:
            append(text, bucket, "", family)
    return rows


def existing_normalized_texts(path: Path = V1_TRAIN) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {normalize(row["text"]) for row in csv.DictReader(stream)}


def validate(rows: list[dict[str, str]], existing_texts: set[str] | None = None) -> dict:
    if len(rows) != 64:
        raise ValueError(f"row count must be exactly 64, got {len(rows)}")
    if any(tuple(row) != FIELDS for row in rows):
        raise ValueError("rows must use the exact required columns in order")
    if any(row["question_id"] != "D5" for row in rows):
        raise ValueError("every question_id must be D5")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("every split must be train")
    if any(row["source"] != SOURCE for row in rows):
        raise ValueError("unexpected source identifier")

    bucket_counts = Counter(row["bucket"] for row in rows)
    if dict(bucket_counts) != BUCKET_COUNTS:
        raise ValueError(f"incorrect bucket counts: {dict(bucket_counts)}")
    if any(row["option_id"] for row in rows if row["bucket"] in {"undecided", "confusion"}):
        raise ValueError("undecided/confusion rows must have blank option_id")
    if any(not row["option_id"] for row in rows if row["bucket"] == "clarity"):
        raise ValueError("clarity rows must have an option_id")
    option_counts = Counter(row["option_id"] for row in rows if row["option_id"])
    if dict(option_counts) != OPTION_COUNTS:
        raise ValueError(f"incorrect option counts: {dict(option_counts)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("IDs must be unique")

    normalized = [normalize(row["text"]) for row in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("text duplicated after lowercase/whitespace normalization")
    overlap = set(normalized) & (existing_texts if existing_texts is not None else existing_normalized_texts())
    if overlap:
        raise ValueError(f"generated text overlaps v1 train.csv: {sorted(overlap)!r}")

    return {
        "rows": len(rows),
        "class_counts": dict(bucket_counts),
        "option_counts": dict(option_counts),
    }


def write_outputs(rows: list[dict[str, str]], summary: dict) -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    provenance = {
        "creation_date": CREATION_DATE,
        "fictional": True,
        "contains_real_customer_data": False,
        "purpose": "Controlled D5-only synthetic training-data development experiment.",
        "source_identifier": SOURCE,
        "row_counts": {"total": summary["rows"], "train": summary["rows"]},
        "class_counts": summary["class_counts"],
        "option_counts": summary["option_counts"],
        "sha256": {OUTPUT.name: digest},
        "v1_data_statement": "The existing v1 train.csv, calibration.csv, and test.csv were not modified.",
        "runtime_statement": "This augmentation is not integrated into training or runtime behaviour.",
        "limitations": [
            "This is synthetic development augmentation, not evidence of real-customer performance.",
            "Authored coverage cannot establish representativeness, semantic independence, or conformal coverage for real customers.",
        ],
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    summary = validate(rows)
    write_outputs(rows, summary)
    display_path = OUTPUT.relative_to(ROOT) if OUTPUT.is_relative_to(ROOT) else OUTPUT
    print(f"Wrote {summary['rows']} rows to {display_path}")


if __name__ == "__main__":
    main()
