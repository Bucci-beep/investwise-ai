import csv
from collections import defaultdict

from ai.text_classifier import BoundedTextClassifier


V1_PATH = "data/nlp/all_questions/train.csv"
V2_PATH = "data/nlp/experiments/d5_v2/d5_training_additions.csv"
CHALLENGE_PATH = (
    "data/nlp/experiments/d5_v2/d5_compositional_challenge.csv"
)


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    v1 = [
        row
        for row in load_csv(V1_PATH)
        if row["question_id"] == "D5"
    ]
    v2 = load_csv(V2_PATH)
    challenge = load_csv(CHALLENGE_PATH)

    # Frozen Experiment 1 training data.
    training_rows = v1 + v2

    # D5 option classifier is trained only on genuine clarity rows.
    clarity_train = [
        row
        for row in training_rows
        if row["bucket"] == "clarity" and row["option_id"]
    ]

    model = BoundedTextClassifier()
    model.fit(
        [row["text"] for row in clarity_train],
        [row["option_id"] for row in clarity_train],
    )

    results = defaultdict(lambda: {"correct": 0, "total": 0})

    print("=" * 70)
    print("D5 EXPERIMENT 3 — COMPOSITIONAL CHALLENGE")
    print("=" * 70)
    print("Training rows:", len(clarity_train))
    print("Challenge rows:", len(challenge))

    overall_correct = 0

    for row in challenge:
        predicted = model.predict(row["text"]).predicted_class
        expected = row["expected_option"]
        correct = predicted == expected
        case_type = row["case_type"]

        results[case_type]["total"] += 1

        if correct:
            results[case_type]["correct"] += 1
            overall_correct += 1

        status = "PASS" if correct else "FAIL"

        print(
            f"\n{status} {row['id']} [{case_type}]"
            f"\nExpected:  {expected}"
            f"\nPredicted: {predicted}"
            f"\nText: {row['text']}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for case_type in [
        "single_source",
        "multiple_source",
        "negative_control",
    ]:
        correct = results[case_type]["correct"]
        total = results[case_type]["total"]
        accuracy = correct / total if total else 0

        print(
            f"{case_type}: "
            f"{correct}/{total} "
            f"({accuracy:.1%})"
        )

    total = len(challenge)

    print(
        f"Overall: {overall_correct}/{total} "
        f"({overall_correct / total:.1%})"
    )


if __name__ == "__main__":
    main()