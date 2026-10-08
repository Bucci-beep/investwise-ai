import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ATTR_PATH = (
    "data/nlp/experiments/d5_source_attribution/"
    "d5_attribution_training.csv"
)

RELATION_PATH = (
    "data/nlp/experiments/d5_relation_binding/"
    "d5_relation_binding_training.csv"
)

DEV_PATH = (
    "data/nlp/experiments/d5_v2/"
    "d5_development_validation.csv"
)

COMPOSITIONAL_PATH = (
    "data/nlp/experiments/d5_v2/"
    "d5_compositional_challenge.csv"
)

COMPETING_PATH = (
    "data/nlp/experiments/d5_v2/"
    "d5_competing_source_challenge.csv"
)


SOURCE_PATTERNS = {
    "salary": [
        r"\bsalary\b",
        r"\bwages?\b",
        r"\bemployment income\b",
        r"\bemployer\b",
        r"\bpay from work\b",
        r"\bmonthly pay\b",
    ],
    "self_employment": [
        r"\bself[- ]?employ",
        r"\bfreelance\b",
        r"\bsole trader\b",
        r"\bbusiness income\b",
        r"\bbusiness earnings?\b",
        r"\bclient payments?\b",
        r"\bclient",
        r"\bcontract work\b",
        r"\bworking for myself\b",
        r"\bown business\b",
        r"\binvoicing income\b",
        r"\bown design work\b",
    ],
    "pension_payments": [
        r"\bpension\b",
        r"\bretirement income\b",
    ],
    "benefits_support": [
        r"\bbenefits?\b",
        r"\bbenefit payments?\b",
        r"\bbenefit income\b",
        r"\buniversal credit\b",
        r"\bfamily support\b",
        r"\bsupport from\b",
        r"\bfinancial support\b",
        r"\bfamily member\b",
        r"\bbrother helps\b",
    ],
    "withdrawals": [
        r"\bwithdraw",
        r"\bwithdrawals?\b",
        r"\bdraw from\b",
        r"\bdrawn from\b",
        r"\bdipping into\b",
        r"\bdip into\b",
        r"\btake from (?:my )?savings\b",
        r"\btake from (?:my )?investments\b",
        r"\bmoney from (?:my )?savings\b",
        r"\bmoney from (?:my )?investments\b",
        r"\buse savings\b",
        r"\bcash in\b",
    ],
    "other_regular": [
        r"\brental income\b",
        r"\brental receipts?\b",
        r"\brent from\b",
        r"\brent i receive\b",
        r"\broyalt",
        r"\blicen[cs]e (?:fees|payments?|income)\b",
        r"\btenant\b",
        r"\bannuity\b",
    ],
}


SOURCE_TO_OPTION = {
    "salary": "D5_SALARY",
    "self_employment": "D5_SELF_EMPLOYMENT",
    "pension_payments": "D5_PENSION_PAYMENTS",
    "benefits_support": "D5_BENEFITS_SUPPORT",
    "withdrawals": "D5_WITHDRAWALS",
    "other_regular": "D5_OTHER_REGULAR",
}


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mentioned_sources(text):
    lower = text.lower()

    return [
        source
        for source, patterns in SOURCE_PATTERNS.items()
        if any(re.search(pattern, lower) for pattern in patterns)
    ]


def target_text(source, text):
    # Explicit target marker is the central 5C change.
    return f"TARGET_SOURCE_{source} {text}"


class RelationBindingModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

    def fit(self, rows):
        x_text = [
            target_text(row["source"], row["text"])
            for row in rows
        ]

        y = [row["attribution"] for row in rows]

        x = self.vectorizer.fit_transform(x_text)
        self.model.fit(x, y)

    def predict(self, source, text):
        x = self.vectorizer.transform(
            [target_text(source, text)]
        )
        return self.model.predict(x)[0]


def explicit_no_regular(text):
    patterns = [
        r"\bno regular source\b",
        r"\bno recurring source\b",
        r"\bno ongoing source\b",
        r"\bno regular money\b",
        r"\bnothing regular\b",
        r"\bnothing comes in regularly\b",
        r"\bno recurring money\b",
        r"\bdon't have any regular money\b",
        r"\bdo not have any regular money\b",
    ]

    lower = text.lower()

    return any(
        re.search(pattern, lower)
        for pattern in patterns
    )


def aggregate(active_sources, text):
    no_regular = explicit_no_regular(text)

    if no_regular and active_sources:
        return "ABSTAIN_CONFLICT"

    if no_regular:
        return "D5_NO_REGULAR"

    if len(active_sources) == 0:
        return "ABSTAIN_NO_ACTIVE_SOURCE"

    if len(active_sources) == 1:
        return SOURCE_TO_OPTION[active_sources[0]]

    return "D5_MULTIPLE"


def predict(model, text):
    mentioned = mentioned_sources(text)

    attribution = {
        source: model.predict(source, text)
        for source in mentioned
    }

    active = [
        source
        for source, state in attribution.items()
        if state == "active"
    ]

    return aggregate(active, text), mentioned, attribution, active


def evaluate(name, rows, expected_field, model):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    correct = 0
    abstain = 0
    case_stats = {}

    for row in rows:
        prediction, mentioned, attribution, active = predict(
            model,
            row["text"],
        )

        expected = row[expected_field]

        if prediction == expected:
            correct += 1

        if prediction.startswith("ABSTAIN"):
            abstain += 1

        case_type = row.get("case_type")

        if case_type:
            case_stats.setdefault(case_type, [0, 0])
            case_stats[case_type][1] += 1

            if prediction == expected:
                case_stats[case_type][0] += 1

        if prediction != expected:
            print(
                f'{row["id"]}: expected={expected} '
                f'predicted={prediction}'
            )
            print(f"  mentioned={mentioned}")
            print(f"  attribution={attribution}")
            print(f"  active={active}")
            print(f'  {row["text"]}')

    total = len(rows)

    print("\nSUMMARY")
    print(
        f"Correct: {correct}/{total} "
        f"({correct / total:.1%})"
    )
    print(
        f"Abstain: {abstain}/{total} "
        f"({abstain / total:.1%})"
    )

    if case_stats:
        print("\nBY CASE TYPE")

        for case_type in sorted(case_stats):
            n_correct, n_total = case_stats[case_type]

            print(
                f"{case_type}: "
                f"{n_correct}/{n_total} "
                f"({n_correct / n_total:.1%})"
            )


def main():
    attribution_rows = load_csv(ATTR_PATH)
    relation_rows = load_csv(RELATION_PATH)

    training_rows = attribution_rows + relation_rows

    print("TRAINING")
    print(f"5B attribution rows: {len(attribution_rows)}")
    print(f"5C relation-binding rows: {len(relation_rows)}")
    print(f"Total: {len(training_rows)}")

    model = RelationBindingModel()
    model.fit(training_rows)

    dev = [
        row
        for row in load_csv(DEV_PATH)
        if row["bucket"] == "clarity"
        and row["option_id"]
    ]

    compositional = load_csv(COMPOSITIONAL_PATH)
    competing = load_csv(COMPETING_PATH)

    evaluate(
        "DEVELOPMENT — OPTION",
        dev,
        "option_id",
        model,
    )

    evaluate(
        "COMPOSITIONAL CHALLENGE",
        compositional,
        "expected_option",
        model,
    )

    evaluate(
        "COMPETING-SOURCE CHALLENGE",
        competing,
        "expected_option",
        model,
    )


if __name__ == "__main__":
    main()
