"""Layered safety screen: existing rules, wider rules and a similarity net.

Every message should be screened before any financial interpretation. The
screen returns the most serious category it finds, or ``None``.

Layers, in order of trust:

1. ``rules``      - the existing ``check_pre_model_controls`` (exact patterns).
2. ``wide_rules`` - broader patterns that tolerate extra words such as
                    "completely" and cover categories the original rules do
                    not (money stress, financial control, delegation).
3. ``similarity`` - nearest labelled example by TF-IDF cosine similarity.
                    This catches near-paraphrases of known examples. It is a
                    lexical stand-in for a meaning-based model; an embedding
                    backend can replace it behind the same interface.

The screen is deliberately recall-first: a false alarm costs a gentle
check-in, a miss can cost far more. It does not diagnose anything and its
categories only choose which approved response wording to show.

This module is not yet wired into the conversation engine. Measure it with
``scripts/compare_safety_layers.py`` first.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ai.pre_model_controls import check_pre_model_controls

ROOT = Path(__file__).resolve().parent.parent
EXEMPLARS = ROOT / "data/safety/safety_exemplars_v1.csv"

# Most serious first. The screen reports the most serious category found.
SEVERITY_ORDER = ["urgent", "care", "money_stress", "financial_control", "refusal", "delegation", "pause"]

# Similarity threshold, chosen on the development benchmark only
# (data/safety/safety_benchmark_v1.csv), never on the held-out set.
SIMILARITY_THRESHOLD = 0.35  # max dev recall (32/34), ties broken by the higher threshold
SIMILARITY_MARGIN = 0.05

_RULE_ACTION_TO_CATEGORY = {
    "safety_stop": "urgent",
    "care_pause": "care",
    "decline": "refusal",
    "pause": "pause",
    "stop": "pause",
    "pace_answer": "pause",
}

# A gap of up to N words that are not negations ("I'm completely overwhelmed",
# but not "I'm not overwhelmed").
_GAP = r"(?:(?!not\b|never\b|no\b)\w+\s+){0,%d}"


def _gap(n: int) -> str:
    return _GAP % n


_WIDE_RULES: dict[str, list[re.Pattern[str]]] = {
    "urgent": [
        re.compile(r"\b(?:kill|killing|hurt|hurting|harm|harming)\s+myself\b", re.I),
        re.compile(r"\bsuicid(?:e|al)\b", re.I),
        re.compile(r"\b(?:end|ending|take|taking)\s+(?:my\s+(?:own\s+)?life|it\s+all|everything)\b(?!\s+(?:insurance|assurance|policy|cover))", re.I),
        re.compile(r"\b(?:don'?t|do not|no longer)\s+want\s+to\s+(?:live|be\s+(?:alive|here|around))\b", re.I),
        re.compile(r"\bno\s+point\s+(?:in\s+)?(?:living|going\s+on|being\s+alive|carrying\s+on)\b", re.I),
        re.compile(r"\bbetter\s+off\s+(?:dead|without\s+me|gone)\b", re.I),
        re.compile(r"\b(?:won'?t|will\s+not)\s+be\s+(?:around|here|alive)\b", re.I),
        re.compile(r"\bwant\s+to\s+(?:die|disappear)\b", re.I),
        re.compile(r"\b(?:won'?t|will\s+not|wouldn'?t)\s+be\s+alive\b", re.I),
    ],
    "care": [
        re.compile(r"\bi(?:'?m|\s+am)\s+" + _gap(2) + r"(?:overwhelmed|panicking|distressed|grieving|devastated|depressed|heartbroken|in\s+pieces|falling\s+apart)\b", re.I),
        re.compile(r"\b(?:can'?t|cannot|can\s+not)\s+cope\b", re.I),
        re.compile(r"\bfeel(?:ing)?\s+" + _gap(2) + r"(?:low|hopeless|overwhelmed|awful|desperate|scared)\b", re.I),
        re.compile(r"\b(?:mum|mom|mother|dad|father|husband|wife|partner|son|daughter|brother|sister|grandma|grandad)\b.{0,40}\b(?:passed|died|death|funeral)\b", re.I),
        re.compile(r"\b(?:bereaved|bereavement|grieving)\b", re.I),
        re.compile(r"\bdiagnosed\s+with\b", re.I),
        re.compile(r"\beverything\s+" + _gap(3) + r"too\s+much\b", re.I),
        re.compile(r"\b(?:keep|been|started)\s+crying\b", re.I),
        re.compile(r"\bpanic\s+attack\b", re.I),
        re.compile(r"\blost\s+my\s+job\b", re.I),
    ],
    "money_stress": [
        re.compile(r"\b(?:stressed|worried|anxious|panicking)\s+" + _gap(1) + r"about\s+(?:money|my\s+finances|bills|debt)\b", re.I),
        re.compile(r"\bdrowning\b", re.I),
        re.compile(r"\b(?:can'?t|cannot)\s+afford\s+(?:food|to\s+eat|the\s+rent|rent|my\s+rent|bills|the\s+bills|to\s+heat)\b", re.I),
        re.compile(r"\bbailiffs?\b", re.I),
        re.compile(r"\bmissed\s+" + _gap(2) + r"(?:rent|mortgage|payments?)\b", re.I),
        re.compile(r"\bloan\s+shark\b", re.I),
        re.compile(r"\bbehind\s+(?:on|with)\s+" + _gap(3) + r"(?:bills?|rent|mortgage|payments?)\b", re.I),
        re.compile(r"\b(?:can'?t|cannot)\s+pay\s+" + _gap(2) + r"(?:debts?|bills?|rent|mortgage)\b", re.I),
        re.compile(r"\b(?:evicted|eviction|repossess\w*)\b", re.I),
    ],
    "financial_control": [
        re.compile(r"\b(?:husband|wife|partner|boyfriend|girlfriend|son|daughter|family|carer|he|she)\s+" + _gap(2)
                   + r"(?:controls?|takes?|won'?t\s+let\s+me|doesn'?t\s+let\s+me|makes?\s+me|is\s+making\s+me|forces?\s+me)\b", re.I),
        re.compile(r"\b(?:pressured|pressurised|forced|made)\s+" + _gap(1) + r"(?:to\s+)?(?:invest|move|transfer)\b", re.I),
        re.compile(r"\bmaking\s+me\s+(?:invest|move|transfer|do\s+this)\b", re.I),
        re.compile(r"\bsomeone\s+(?:online\s+)?(?:told|asked|is\s+telling|made)\s+me\s+to\b", re.I),
        re.compile(r"\b(?:gives?|giving)\s+me\s+an\s+allowance\b", re.I),
        re.compile(r"\bpermission\s+to\s+spend\b", re.I),
    ],
    "refusal": [
        re.compile(r"\b(?:rather|prefer)\s+not\s+(?:to\s+)?(?:say|answer|share)\b", re.I),
        re.compile(r"\bthat'?s\s+(?:private|personal)\b", re.I),
        re.compile(r"\b(?:don'?t|do\s+not)\s+want\s+to\s+(?:share|answer|say)\b", re.I),
        re.compile(r"\bskip\s+(?:this|that)\b", re.I),
        re.compile(r"\bnot\s+comfortable\s+(?:answering|sharing|saying)\b", re.I),
    ],
    "delegation": [
        re.compile(r"\b(?:you|just)\s+(?:pick|choose|decide)\b", re.I),
        re.compile(r"\b(?:pick|choose|decide)\s+(?:for\s+me|whatever)\b", re.I),
        re.compile(r"\bwhat\s+would\s+you\s+(?:pick|choose|do)\b", re.I),
        re.compile(r"\byou\s+know\s+best\b", re.I),
        re.compile(r"\b(?:partner|husband|wife|adviser|advisor|son|daughter)\s+(?:handles|deals\s+with|decides)\b", re.I),
        re.compile(r"\bcan\s+(?:my\s+)?\w+(?:\s+\w+)?\s+answer\s+(?:instead|for\s+me)\b", re.I),
    ],
    "pause": [
        re.compile(r"\b(?:stop|pause|break)\s+(?:for\s+now|for\s+today|here)\b", re.I),
        re.compile(r"\bneed\s+a\s+break\b", re.I),
        re.compile(r"\bcome\s+back\s+to\s+(?:this|it)\b", re.I),
        re.compile(r"\bhad\s+enough\b", re.I),
        re.compile(r"\b(?:continue|finish|carry\s+on)\s+(?:this\s+)?later\b", re.I),
    ],
}


@dataclass(frozen=True)
class SafetyResult:
    category: str | None
    layer: str | None
    detail: str | None = None
    similarity: float | None = None

    @property
    def flagged(self) -> bool:
        return self.category is not None


def rules_layer(text: str) -> SafetyResult:
    result = check_pre_model_controls(text)
    if result.matched and result.action in _RULE_ACTION_TO_CATEGORY:
        return SafetyResult(_RULE_ACTION_TO_CATEGORY[result.action], "rules", result.action)
    return SafetyResult(None, None)


def wide_rules_layer(text: str) -> SafetyResult:
    for category in SEVERITY_ORDER:
        for pattern in _WIDE_RULES[category]:
            match = pattern.search(text)
            if match:
                return SafetyResult(category, "wide_rules", match.group(0))
    return SafetyResult(None, None)


class SimilarityNet:
    """Nearest-exemplar classifier using word and character TF-IDF."""

    def __init__(self, exemplars: list[tuple[str, str]],
                 threshold: float = SIMILARITY_THRESHOLD, margin: float = SIMILARITY_MARGIN):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.texts = [text for _, text in exemplars]
        self.categories = [category for category, _ in exemplars]
        self.threshold = threshold
        self.margin = margin
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)),
        ])
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def nearest(self, text: str) -> tuple[str | None, float, float]:
        """Return (best non-benign category, its similarity, best benign similarity)."""
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform([text]), self.matrix)[0]
        best_cat, best_sim, best_benign = None, 0.0, 0.0
        for category, sim in zip(self.categories, sims):
            if category == "benign":
                best_benign = max(best_benign, float(sim))
            elif sim > best_sim:
                best_cat, best_sim = category, float(sim)
        return best_cat, best_sim, best_benign

    def __call__(self, text: str) -> SafetyResult:
        category, sim, benign = self.nearest(text)
        if category and sim >= self.threshold and sim >= benign + self.margin:
            return SafetyResult(category, "similarity", None, round(sim, 3))
        return SafetyResult(None, None, None, round(sim, 3))


def load_exemplars(path: Path = EXEMPLARS) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [(row["category"], row["text"]) for row in csv.DictReader(handle)]


@lru_cache(maxsize=1)
def default_similarity_net() -> SimilarityNet:
    return SimilarityNet(load_exemplars())


def _most_serious(results: list[SafetyResult]) -> SafetyResult:
    flagged = [r for r in results if r.flagged]
    if not flagged:
        return SafetyResult(None, None)
    return min(flagged, key=lambda r: SEVERITY_ORDER.index(r.category))


def screen(text: str, *, use_wide_rules: bool = True, use_similarity: bool = True) -> SafetyResult:
    """Run every enabled layer and return the most serious category found."""
    results = [rules_layer(text)]
    if use_wide_rules:
        results.append(wide_rules_layer(text))
    if use_similarity:
        results.append(default_similarity_net()(text))
    return _most_serious(results)