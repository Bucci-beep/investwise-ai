import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = (
    "data/nlp/experiments/d5_source_decomposition/"
    "d5_source_training.csv"
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

POSITIVE_SOURCES = [
    "salary",
    "self_employment",
    "pension_payments",
    "benefits_support",
    "withdrawals",
    "other_regular",
]

ALL_SOURCES = POSITIVE_SOURCES + ["no_regular"]

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


class SourceDecompositionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.models = {}

    def fit(self, rows):
        texts = [row["text"] for row in rows]
        x = self.vectorizer.fit_transform(texts)

        for source in ALL_SOURCES:
            y = [int(row[source]) for row in rows]

            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(x, y)
            self.models[source] = model

    def predict_sources(self, text):
        x = self.vectorizer.transform([text])

        return {
            source: int(self.models[source].predict(x)[0])
            for source in ALL_SOURCES
        }


def aggregate(source_predictions):
    active = [
        source
        for source in POSITIVE_SOURCES
        if source_predictions[source] == 1
    ]

    no_regular = source_predictions["no_regular"] == 1

    if no_regular and active:
        return "ABSTAIN_CONFLICT"

    if no_regular:
        return "D5_NO_REGULAR"

    if len(active) == 0:
        return "ABSTAIN_NO_SOURCE"

    if len(active) == 1:
        return SOURCE_TO_OPTION[active[0]]

    return "D5_MULTIPLE"


def evaluate(name, rows, expected_field, model):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    correct = 0
    abstain = 0

    case_stats = {}

    for row in rows:
        source_predictions = model.predict_sources(row["text"])
        prediction = aggregate(source_predictions)
        expected = row[expected_field]

        if prediction == expected:
            correct += 1

        if prediction.startswith("ABSTAIN"):
            abstain += 1

        case_type = row.get("case_type")
        if case_type:
            if case_type not in case_stats:
                case_stats[case_type] = [0, 0]

            case_stats[case_type][1] += 1

            if prediction == expected:
                case_stats[case_type][0] += 1

        if prediction != expected:
            active = [
                source
                for source, value in source_predictions.items()
                if value == 1
            ]

            print(
                f'{row["id"]}: expected={expected} '
                f'predicted={prediction}'
            )
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
            case_correct, case_total = case_stats[case_type]
            print(
                f"{case_type}: "
                f"{case_correct}/{case_total} "
                f"({case_correct / case_total:.1%})"
            )


def main():
    train = load_csv(TRAIN_PATH)

    model = SourceDecompositionModel()
    model.fit(train)

    dev = [
        row
        for row in load_csv(DEV_PATH)
        if row["bucket"] == "clarity" and row["option_id"]
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
