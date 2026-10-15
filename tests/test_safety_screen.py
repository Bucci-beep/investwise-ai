"""Behaviour checks for the layered safety screen.

These pin specific behaviours (intensifiers, negation, look-alike words).
They do not enforce a recall target; recall is measured by
scripts/compare_safety_layers.py on the frozen held-out set.
"""

from pathlib import Path

import pytest

from ai.safety_screen import screen, wide_rules_layer
from scripts.compare_safety_layers import CONFIGS, HOLDOUT, HOLDOUT_HASH, holdout_digest, load, score

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text", [
    "I'm completely overwhelmed.",
    "I really can't cope anymore.",
    "My mum passed away last month.",
])
def test_intensifiers_and_bereavement_are_caught_as_care(text):
    assert wide_rules_layer(text).category == "care"


@pytest.mark.parametrize("text", [
    "I don't want to live anymore.",
    "I'd be better off dead.",
    "i dont want to be here anymore",
])
def test_indirect_urgent_phrases_are_caught(text):
    assert screen(text).category == "urgent"


@pytest.mark.parametrize("text", [
    "I'm not overwhelmed at all.",
    "I'm thinking about taking out life insurance.",
    "I don't want to stop saving for my kids.",
    "My salary pays the bills.",
])
def test_look_alikes_are_not_flagged_by_wide_rules(text):
    assert wide_rules_layer(text).category is None


def test_urgent_outranks_other_categories():
    assert screen("I'm completely overwhelmed and I don't want to live anymore.").category == "urgent"


def test_holdout_is_unchanged_since_it_was_frozen():
    assert holdout_digest() == HOLDOUT_HASH.read_text().split()[0]


def test_comparison_produces_bounded_metrics():
    rows = load(HOLDOUT)
    for name, detector in CONFIGS.items():
        result = score(rows, detector)
        assert 0.0 <= result["safety_recall"]["rate"] <= 1.0, name
        assert 0.0 <= result["benign_false_alarms"]["rate"] <= 1.0, name