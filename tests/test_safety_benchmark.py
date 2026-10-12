"""Checks that the safety benchmark is well formed and the evaluator runs.

These tests do not require a minimum recall: the benchmark measures the
current controls, it does not gate them. Recall targets belong in a later
change once the labels have been independently reviewed.
"""

from collections import Counter

from scripts.evaluate_safety import ACCEPTABLE, CATEGORIES, evaluate, load_rows

ALLOWED_CATEGORIES = set(CATEGORIES) | {"benign", "ambiguous"}
ALLOWED_RESPONSES = set(ACCEPTABLE) | {"continue", "human_review"}


def test_benchmark_rows_are_well_formed():
    rows = load_rows()
    ids = [row["case_id"] for row in rows]
    assert len(ids) == len(set(ids)), "case_id values must be unique"
    for row in rows:
        assert row["text"].strip(), row["case_id"]
        assert row["category"] in ALLOWED_CATEGORIES, row["case_id"]
        assert row["expected_response"] in ALLOWED_RESPONSES, row["case_id"]


def test_each_category_has_enough_cases():
    counts = Counter(row["category"] for row in load_rows())
    for category in ALLOWED_CATEGORIES:
        assert counts[category] >= 4, f"{category} has only {counts[category]} cases"


def test_category_and_expected_response_agree():
    expected = {
        "urgent": "urgent_stop",
        "benign": "continue",
        "ambiguous": "human_review",
        "refusal": "respect_refusal",
        "pause": "pause",
    }
    for row in load_rows():
        if row["category"] in expected:
            assert row["expected_response"] == expected[row["category"]], row["case_id"]


def test_evaluator_produces_bounded_metrics():
    report = evaluate(load_rows())
    head = report["headline"]
    assert 0.0 <= head["safety_recall"] <= 1.0
    assert 0.0 <= head["benign_false_alarm_rate"] <= 1.0
    assert head["safety_cases"] > 0 and head["benign_cases"] > 0
