import csv
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ai.text_classifier import BoundedTextClassifier


V1_PATH = "data/nlp/all_questions/train.csv"
V2_PATH = "data/nlp/experiments/d5_v2/d5_training_additions.csv"
DEV_PATH = "data/nlp/experiments/d5_v2/d5_development_validation.csv"

INTENT_LABELS = ["clarity", "undecided", "confusion"]

OPTION_LABELS = [
    "D5_SALARY",
    "D5_SELF_EMPLOYMENT",
    "D5_PENSION_PAYMENTS",
    "D5_BENEFITS_SUPPORT",
    "D5_WITHDRAWALS",
    "D5_OTHER_REGULAR",
    "D5_NO_REGULAR",
    "D5_MULTIPLE",
]


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def predict_labels(model, rows):
    return [model.predict(row["text"]).predicted_class for row in rows]


def evaluate(name, training_rows, dev_rows):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Stage 1: intent classifier
    intent_model = BoundedTextClassifier()
    intent_model.fit(
        [row["text"] for row in training_rows],
        [row["bucket"] for row in training_rows],
    )

    intent_true = [row["bucket"] for row in dev_rows]
    intent_pred = predict_labels(intent_model, dev_rows)

    print("\nINTENT")
    print("Training rows:", len(training_rows))
    print(
        "Training distribution:",
        Counter(row["bucket"] for row in training_rows),
    )
    print(
        "Macro F1:",
        round(
            f1_score(
                intent_true,
                intent_pred,
                labels=INTENT_LABELS,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    )

    print(
        classification_report(
            intent_true,
            intent_pred,
            labels=INTENT_LABELS,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print(INTENT_LABELS)
    print(
        confusion_matrix(
            intent_true,
            intent_pred,
            labels=INTENT_LABELS,
        )
    )

    # Stage 2: option classifier
    clarity_train = [
        row
        for row in training_rows
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    clarity_dev = [
        row
        for row in dev_rows
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    option_model = BoundedTextClassifier()
    option_model.fit(
        [row["text"] for row in clarity_train],
        [row["option_id"] for row in clarity_train],
    )

    option_true = [row["option_id"] for row in clarity_dev]
    option_pred = predict_labels(option_model, clarity_dev)

    print("\nOPTION")
    print("Training rows:", len(clarity_train))
    print(
        "Training distribution:",
        Counter(row["option_id"] for row in clarity_train),
    )
    print(
        "Macro F1:",
        round(
            f1_score(
                option_true,
                option_pred,
                labels=OPTION_LABELS,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    )

    print(
        classification_report(
            option_true,
            option_pred,
            labels=OPTION_LABELS,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print(OPTION_LABELS)
    print(
        confusion_matrix(
            option_true,
            option_pred,
            labels=OPTION_LABELS,
        )
    )

    print("\nMISCLASSIFICATIONS")

    errors = 0

    for row, prediction in zip(dev_rows, intent_pred):
        if prediction != row["bucket"]:
            errors += 1
            print(
                f'INTENT {row["id"]}: '
                f'expected={row["bucket"]} predicted={prediction}\n'
                f'  {row["text"]}'
            )

    for row, prediction in zip(clarity_dev, option_pred):
        if prediction != row["option_id"]:
            errors += 1
            print(
                f'OPTION {row["id"]}: '
                f'expected={row["option_id"]} predicted={prediction}\n'
                f'  {row["text"]}'
            )

    if errors == 0:
        print("None")


def main():
    v1 = [
        row
        for row in load_csv(V1_PATH)
        if row["question_id"] == "D5"
    ]

    v2 = load_csv(V2_PATH)
    dev = load_csv(DEV_PATH)

    evaluate(
        "MODEL A — V1 D5 ONLY",
        v1,
        dev,
    )

    evaluate(
        "MODEL B — V1 D5 + V2 AUGMENTATION",
        v1 + v2,
        dev,
    )


if __name__ == "__main__":
    main()