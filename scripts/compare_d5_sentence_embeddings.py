import csv
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ai.text_classifier import BoundedTextClassifier


V1_PATH = "data/nlp/all_questions/train.csv"
V2_PATH = "data/nlp/experiments/d5_v2/d5_training_additions.csv"
DEV_PATH = "data/nlp/experiments/d5_v2/d5_development_validation.csv"
COMPOSITIONAL_PATH = "data/nlp/experiments/d5_v2/d5_compositional_challenge.csv"
COMPETING_PATH = "data/nlp/experiments/d5_v2/d5_competing_source_challenge.csv"

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


def tfidf_predictions(training_rows, eval_rows, target):
    model = BoundedTextClassifier()

    model.fit(
        [row["text"] for row in training_rows],
        [row[target] for row in training_rows],
    )

    return [
        model.predict(row["text"]).predicted_class
        for row in eval_rows
    ]


def embedding_predictions(encoder, training_rows, eval_rows, target):
    train_x = encoder.encode(
        [row["text"] for row in training_rows],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    eval_x = encoder.encode(
        [row["text"] for row in eval_rows],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(
        train_x,
        [row[target] for row in training_rows],
    )

    return model.predict(eval_x)


def macro_f1(rows, predictions, target, labels):
    return f1_score(
        [row[target] for row in rows],
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )


def print_changes(rows, baseline, contextual, target):
    changed = 0

    for row, old, new in zip(rows, baseline, contextual):
        if old != new:
            changed += 1
            expected = row[target]
            print(
                f'{row["id"]}: expected={expected} '
                f'tfidf={old} minilm={new}\n'
                f'  {row["text"]}'
            )

    if changed == 0:
        print("None")


def evaluate_stress(name, rows, baseline, contextual):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    expected = [row["expected_option"] for row in rows]

    baseline_correct = sum(
        prediction == truth
        for prediction, truth in zip(baseline, expected)
    )

    contextual_correct = sum(
        prediction == truth
        for prediction, truth in zip(contextual, expected)
    )

    print(
        f"TF-IDF: {baseline_correct}/{len(rows)} "
        f"({baseline_correct / len(rows):.1%})"
    )

    print(
        f"MiniLM: {contextual_correct}/{len(rows)} "
        f"({contextual_correct / len(rows):.1%})"
    )

    if "case_type" in rows[0]:
        print("\nBY CASE TYPE")

        for case_type in sorted({row["case_type"] for row in rows}):
            indices = [
                i
                for i, row in enumerate(rows)
                if row["case_type"] == case_type
            ]

            tfidf_correct = sum(
                baseline[i] == expected[i]
                for i in indices
            )

            minilm_correct = sum(
                contextual[i] == expected[i]
                for i in indices
            )

            print(
                f"{case_type}: "
                f"TF-IDF {tfidf_correct}/{len(indices)}, "
                f"MiniLM {minilm_correct}/{len(indices)}"
            )

    print("\nPREDICTION CHANGES")
    print_changes(
        rows,
        baseline,
        contextual,
        "expected_option",
    )


def main():
    v1 = [
        row
        for row in load_csv(V1_PATH)
        if row["question_id"] == "D5"
    ]

    v2 = load_csv(V2_PATH)
    dev = load_csv(DEV_PATH)
    compositional = load_csv(COMPOSITIONAL_PATH)
    competing = load_csv(COMPETING_PATH)

    training = v1 + v2

    clarity_train = [
        row
        for row in training
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    clarity_dev = [
        row
        for row in dev
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    print("Loading frozen encoder:", ENCODER_NAME)
    encoder = SentenceTransformer(
        ENCODER_NAME,
        device="cpu",
    )

    # ---------------------------------------------------------
    # Intent
    # ---------------------------------------------------------

    tfidf_intent = tfidf_predictions(
        training,
        dev,
        "bucket",
    )

    minilm_intent = embedding_predictions(
        encoder,
        training,
        dev,
        "bucket",
    )

    print("\n" + "=" * 70)
    print("D5 DEVELOPMENT — INTENT")
    print("=" * 70)

    print("Training rows:", len(training))
    print("Distribution:", Counter(row["bucket"] for row in training))

    print(
        "TF-IDF macro F1:",
        round(
            macro_f1(
                dev,
                tfidf_intent,
                "bucket",
                INTENT_LABELS,
            ),
            4,
        ),
    )

    print(
        "MiniLM macro F1:",
        round(
            macro_f1(
                dev,
                minilm_intent,
                "bucket",
                INTENT_LABELS,
            ),
            4,
        ),
    )

    print("\nMiniLM classification report")
    print(
        classification_report(
            [row["bucket"] for row in dev],
            minilm_intent,
            labels=INTENT_LABELS,
            zero_division=0,
        )
    )

    print("MiniLM confusion matrix")
    print(INTENT_LABELS)
    print(
        confusion_matrix(
            [row["bucket"] for row in dev],
            minilm_intent,
            labels=INTENT_LABELS,
        )
    )

    print("\nPREDICTION CHANGES")
    print_changes(
        dev,
        tfidf_intent,
        minilm_intent,
        "bucket",
    )

    # ---------------------------------------------------------
    # Option
    # ---------------------------------------------------------

    tfidf_option = tfidf_predictions(
        clarity_train,
        clarity_dev,
        "option_id",
    )

    minilm_option = embedding_predictions(
        encoder,
        clarity_train,
        clarity_dev,
        "option_id",
    )

    print("\n" + "=" * 70)
    print("D5 DEVELOPMENT — OPTION")
    print("=" * 70)

    print("Training rows:", len(clarity_train))
    print(
        "Distribution:",
        Counter(row["option_id"] for row in clarity_train),
    )

    print(
        "TF-IDF macro F1:",
        round(
            macro_f1(
                clarity_dev,
                tfidf_option,
                "option_id",
                OPTION_LABELS,
            ),
            4,
        ),
    )

    print(
        "MiniLM macro F1:",
        round(
            macro_f1(
                clarity_dev,
                minilm_option,
                "option_id",
                OPTION_LABELS,
            ),
            4,
        ),
    )

    print("\nMiniLM classification report")
    print(
        classification_report(
            [row["option_id"] for row in clarity_dev],
            minilm_option,
            labels=OPTION_LABELS,
            zero_division=0,
        )
    )

    print("\nPREDICTION CHANGES")
    print_changes(
        clarity_dev,
        tfidf_option,
        minilm_option,
        "option_id",
    )

    # ---------------------------------------------------------
    # Frozen stress tests
    # ---------------------------------------------------------

    for name, rows in [
        ("COMPOSITIONAL CHALLENGE", compositional),
        ("COMPETING-SOURCE CHALLENGE", competing),
    ]:
        tfidf_stress = tfidf_predictions(
            clarity_train,
            rows,
            "option_id",
        )

        minilm_stress = embedding_predictions(
            encoder,
            clarity_train,
            rows,
            "option_id",
        )

        evaluate_stress(
            name,
            rows,
            tfidf_stress,
            minilm_stress,
        )


if __name__ == "__main__":
    main()
