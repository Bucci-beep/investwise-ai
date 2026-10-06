import csv
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

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


def build_word_char_model():
    """Experimental model only. Production classifier is unchanged."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_predictions(title, y_true, y_pred, labels):
    print("\n" + title)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    print("Macro F1:", round(macro_f1, 4))

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print(labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))

    return macro_f1


def main():
    v1 = [
        row
        for row in load_csv(V1_PATH)
        if row["question_id"] == "D5"
    ]

    v2 = load_csv(V2_PATH)
    dev = load_csv(DEV_PATH)

    training_rows = v1 + v2

    print("=" * 70)
    print("D5 EXPERIMENT 2 — WORD VS WORD + CHARACTER TF-IDF")
    print("=" * 70)

    print("Training rows:", len(training_rows))
    print("Development rows:", len(dev))
    print(
        "Intent training distribution:",
        Counter(row["bucket"] for row in training_rows),
    )

    # ---------------------------------------------------------
    # INTENT
    # ---------------------------------------------------------

    intent_train_text = [row["text"] for row in training_rows]
    intent_train_labels = [row["bucket"] for row in training_rows]

    intent_true = [row["bucket"] for row in dev]

    # Baseline: exact production BoundedTextClassifier
    word_intent = BoundedTextClassifier()
    word_intent.fit(intent_train_text, intent_train_labels)

    word_intent_pred = [
        word_intent.predict(row["text"]).predicted_class
        for row in dev
    ]

    # Candidate: word + character TF-IDF
    word_char_intent = build_word_char_model()
    word_char_intent.fit(intent_train_text, intent_train_labels)

    word_char_intent_pred = word_char_intent.predict(
        [row["text"] for row in dev]
    )

    word_intent_f1 = evaluate_predictions(
        "INTENT — WORD ONLY",
        intent_true,
        word_intent_pred,
        INTENT_LABELS,
    )

    word_char_intent_f1 = evaluate_predictions(
        "INTENT — WORD + CHARACTER",
        intent_true,
        word_char_intent_pred,
        INTENT_LABELS,
    )

    # ---------------------------------------------------------
    # OPTION
    # ---------------------------------------------------------

    clarity_train = [
        row
        for row in training_rows
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    clarity_dev = [
        row
        for row in dev
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    option_train_text = [row["text"] for row in clarity_train]
    option_train_labels = [row["option_id"] for row in clarity_train]

    option_true = [row["option_id"] for row in clarity_dev]

    print(
        "\nOption training distribution:",
        Counter(option_train_labels),
    )

    word_option = BoundedTextClassifier()
    word_option.fit(option_train_text, option_train_labels)

    word_option_pred = [
        word_option.predict(row["text"]).predicted_class
        for row in clarity_dev
    ]

    word_char_option = build_word_char_model()
    word_char_option.fit(option_train_text, option_train_labels)

    word_char_option_pred = word_char_option.predict(
        [row["text"] for row in clarity_dev]
    )

    word_option_f1 = evaluate_predictions(
        "OPTION — WORD ONLY",
        option_true,
        word_option_pred,
        OPTION_LABELS,
    )

    word_char_option_f1 = evaluate_predictions(
        "OPTION — WORD + CHARACTER",
        option_true,
        word_char_option_pred,
        OPTION_LABELS,
    )

    # ---------------------------------------------------------
    # DIRECT COMPARISON
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Intent macro F1: "
        f"word={word_intent_f1:.4f} "
        f"word+char={word_char_intent_f1:.4f} "
        f"delta={word_char_intent_f1 - word_intent_f1:+.4f}"
    )

    print(
        f"Option macro F1: "
        f"word={word_option_f1:.4f} "
        f"word+char={word_char_option_f1:.4f} "
        f"delta={word_char_option_f1 - word_option_f1:+.4f}"
    )

    print("\nPREDICTION CHANGES")

    changes = 0

    for row, old, new in zip(
        dev,
        word_intent_pred,
        word_char_intent_pred,
    ):
        if old != new:
            changes += 1

            old_correct = old == row["bucket"]
            new_correct = new == row["bucket"]

            print(
                f'INTENT {row["id"]}: '
                f'expected={row["bucket"]} '
                f'word={old} '
                f'word+char={new} '
                f'old_correct={old_correct} '
                f'new_correct={new_correct}\n'
                f'  {row["text"]}'
            )

    for row, old, new in zip(
        clarity_dev,
        word_option_pred,
        word_char_option_pred,
    ):
        if old != new:
            changes += 1

            old_correct = old == row["option_id"]
            new_correct = new == row["option_id"]

            print(
                f'OPTION {row["id"]}: '
                f'expected={row["option_id"]} '
                f'word={old} '
                f'word+char={new} '
                f'old_correct={old_correct} '
                f'new_correct={new_correct}\n'
                f'  {row["text"]}'
            )

    if changes == 0:
        print("No predictions changed.")


if __name__ == "__main__":
    main()