import csv
import hashlib
import json
from pathlib import Path


OUTPUT_DIR = Path("data/nlp/experiments/d5_source_attribution")
OUTPUT_CSV = OUTPUT_DIR / "d5_attribution_training.csv"
PROVENANCE_JSON = OUTPUT_DIR / "provenance.json"

SOURCE_EXAMPLES = {
    "salary": {
        "active": [
            "My salary currently pays for my regular household expenses.",
            "The wages from my job are what I use for rent, food and bills.",
            "I live on my employment income and use it for everyday costs.",
            "My monthly pay from work funds my normal living expenses.",
            "The money I earn as an employee currently covers my household spending.",
            "My wages go towards the regular costs of running my home.",
            "I use the salary from my job to meet my day-to-day expenses.",
            "Employment income is currently what pays my usual living costs.",
        ],
        "inactive": [
            "I receive a salary, but I put it into savings rather than use it for living costs.",
            "My wages still come in, but they do not pay my household expenses.",
            "I have employment income, although none of it currently goes towards my regular bills.",
            "My salary is kept aside and is not used for my day-to-day spending.",
            "I get paid by my employer, but that money does not fund my current living costs.",
            "The wages from my job are invested instead of being used for household expenses.",
            "I still earn a salary, although my regular living costs are paid from somewhere else.",
            "Employment income comes in each month, but I do not use it for rent, food or bills.",
        ],
    },
    "self_employment": {
        "active": [
            "Income from my freelance work currently pays my regular living costs.",
            "The money I make from my business is what I use for household expenses.",
            "Client payments from my self-employed work cover my everyday bills.",
            "My current living costs are funded by earnings from working for myself.",
            "The income from my own business pays for my rent, food and regular expenses.",
            "I use my freelance earnings to meet my normal household costs.",
            "Payments for the contract work I do currently fund my day-to-day spending.",
            "Money earned through my self-employment is what supports my regular living expenses.",
        ],
        "inactive": [
            "I do some freelance work, but I keep that income separate from my living expenses.",
            "My business earns money, although I do not use those earnings for household costs.",
            "I receive client payments, but they are retained in the business rather than spent on my bills.",
            "My self-employed income exists, but none of it currently funds my everyday spending.",
            "I earn money from contract work, although it is not used for my regular living costs.",
            "The income from my business is reinvested and does not pay my household expenses.",
            "I still have freelance earnings, but those payments are not what I live on.",
            "Money comes in from my own work, but I keep it aside instead of using it for rent, food or bills.",
        ],
    },
    "pension_payments": {
        "active": [
            "The pension payments I receive currently cover my regular living costs.",
            "I use my monthly pension income to pay my household bills.",
            "Money paid to me from my pension funds my everyday expenses.",
            "My current pension payments are what I live on.",
            "The pension income arriving each month pays for my rent, food and other regular costs.",
            "I rely on the pension payments I receive for my normal household spending.",
            "My day-to-day expenses are currently funded by pension income.",
            "The money being paid out from my pension covers my usual living costs.",
        ],
        "inactive": [
            "I receive pension payments, but I save them rather than use them for living costs.",
            "Pension income comes in, although it does not currently pay my household expenses.",
            "I have money being paid from a pension, but none of it goes towards my regular bills.",
            "My pension payments are kept aside and are not used for everyday spending.",
            "I receive pension income each month, although my living costs are funded elsewhere.",
            "The pension money I get is invested rather than spent on household expenses.",
            "Pension payments arrive regularly, but they are not what I currently live on.",
            "I get income from a pension, but I do not use it for rent, food or normal bills.",
        ],
    },
    "benefits_support": {
        "active": [
            "The benefits I receive currently pay my regular living costs.",
            "I use benefit payments to cover my rent, food and household bills.",
            "Regular financial support from my family funds my everyday expenses.",
            "The support I receive each month is what I use for normal living costs.",
            "Benefit income currently covers my usual household spending.",
            "Money regularly provided by a family member pays my day-to-day expenses.",
            "I rely on the benefits coming in to meet my regular bills.",
            "Regular support from someone else is currently funding my living costs.",
        ],
        "inactive": [
            "I receive benefits, but I save them instead of using them for living costs.",
            "Benefit payments come in, although they do not currently pay my household expenses.",
            "My family gives me regular financial support, but I keep that money aside.",
            "I receive support each month, although none of it is used for my normal bills.",
            "Benefits are paid to me, but my current living costs are funded elsewhere.",
            "A family member regularly sends me money, but it does not pay my household expenses.",
            "I still receive benefit income, although that money is not what I live on.",
            "Regular support comes in from someone else, but I do not use it for rent, food or bills.",
        ],
    },
    "withdrawals": {
        "active": [
            "I regularly withdraw money from my savings to pay my living costs.",
            "Money I take from my investments currently covers my household expenses.",
            "I draw from my savings to meet my normal day-to-day bills.",
            "Regular withdrawals from my investments are what I use to live on.",
            "I use money taken out of my savings for rent, food and other regular costs.",
            "My everyday spending is currently funded by withdrawals from investments.",
            "I take money from my savings regularly to cover household bills.",
            "The withdrawals I make from my investments pay my usual living expenses.",
        ],
        "inactive": [
            "I make withdrawals from savings, but I do not use that money for living costs.",
            "Money comes out of my investments, although it does not pay my household expenses.",
            "I sometimes withdraw from savings, but those withdrawals are kept aside from my regular bills.",
            "I take money from investments, although none of it currently funds my everyday spending.",
            "Savings withdrawals occur, but my living costs are paid from somewhere else.",
            "I withdraw money from my investments and reinvest it rather than use it for household expenses.",
            "I still take money from savings occasionally, but it is not what I live on.",
            "Withdrawals from my investments are not used for my rent, food or normal bills.",
        ],
    },
    "other_regular": {
        "active": [
            "The rental income I receive currently pays my regular living costs.",
            "Royalties coming in each month are what I use for household expenses.",
            "Regular licence payments currently fund my everyday spending.",
            "The rent I receive from a property pays my normal living costs.",
            "Recurring royalty income covers my rent, food and household bills.",
            "I use regular licensing income to meet my day-to-day expenses.",
            "Money from a tenant comes in regularly and funds my household spending.",
            "Another recurring income source, rental receipts, currently pays my living expenses.",
        ],
        "inactive": [
            "I receive rental income, but I keep it separate from my living expenses.",
            "Royalties come in regularly, although they do not currently pay my household bills.",
            "I receive licence payments, but that money is saved rather than used for everyday spending.",
            "Rental income arrives each month, although none of it funds my regular living costs.",
            "I have recurring royalty income, but my household expenses are paid from somewhere else.",
            "The rent from my property is reinvested and is not used for my day-to-day costs.",
            "Regular licensing income comes in, but it is not what I currently live on.",
            "I receive money from a tenant, although I do not use it for rent, food or household bills.",
        ],
    },
}


def build():
    rows = []

    for source, labels in SOURCE_EXAMPLES.items():
        for attribution in ("active", "inactive"):
            examples = labels[attribution]

            if len(examples) != 8:
                raise ValueError(
                    f"{source}/{attribution}: expected 8 examples, "
                    f"found {len(examples)}"
                )

            for index, text in enumerate(examples, start=1):
                rows.append(
                    {
                        "id": (
                            f"D5-ATTR-{source.upper()}-"
                            f"{attribution.upper()}-{index:02d}"
                        ),
                        "question_id": "D5",
                        "source": source,
                        "attribution": attribution,
                        "text": text,
                        "split": "train",
                        "origin": "fictional_authored_d5_attribution_5b",
                    }
                )

    if len(rows) != 96:
        raise ValueError(f"Expected 96 rows, found {len(rows)}")

    normalized = [
        " ".join(row["text"].lower().split())
        for row in rows
    ]

    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate normalized text detected")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question_id",
        "source",
        "attribution",
        "text",
        "split",
        "origin",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(OUTPUT_CSV.read_bytes()).hexdigest()

    provenance = {
        "experiment": "D5 Experiment 5B",
        "purpose": (
            "Balanced synthetic development augmentation for learning "
            "whether a specifically named source actively funds current "
            "living costs or is explicitly inactive."
        ),
        "row_count": len(rows),
        "sources": list(SOURCE_EXAMPLES),
        "active_examples_per_source": 8,
        "inactive_examples_per_source": 8,
        "sha256": digest,
        "limitations": [
            "Synthetic authored development data is not evidence of real-customer performance.",
            "The dataset was created after observing weaknesses in earlier D5 development experiments.",
            "Observed development and stress examples must not be treated as independent validation.",
            "The augmentation tests source attribution only and does not establish final D5 classification performance.",
        ],
    }

    PROVENANCE_JSON.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote provenance to {PROVENANCE_JSON}")
    print(f"SHA256: {digest}")

    print("\nCounts:")
    for source in SOURCE_EXAMPLES:
        active = sum(
            row["source"] == source and row["attribution"] == "active"
            for row in rows
        )
        inactive = sum(
            row["source"] == source and row["attribution"] == "inactive"
            for row in rows
        )
        print(f"  {source}: active={active}, inactive={inactive}")


if __name__ == "__main__":
    build()
