"""D5 lexical structure for a learned TF-IDF/logistic-regression classifier.

Source names here identify *mentions*, including negated, held, historical and
other people's sources. They do not say which money funds living costs. Raw
words near each mention let the fitted classifier learn those distinctions.
The analyzer never emits financial options or decides active source members.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ai.text_classifier import BoundedTextClassifier


D5_FEATURE_VERSION = "d5_source_scoped_raw_ngrams_v6"
FEATURE_VERSION = D5_FEATURE_VERSION
_WORDS = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).build_analyzer()
_TOKEN = re.compile(r"\b\w+(?:['’-]\w+)*\b", re.UNICODE)
_BOUNDARY = re.compile(r"[.;!?\n]+|\s*[,;]\s*|\b(?:but|whereas|while|however|although|yet)\b", re.I)
_OWNER = re.compile(
    r"\b(?:my|our|his|her|their|your|mum|mom|mother|dad|father|parents?|"
    r"partner|husband|wife|spouse|flatmate|housemate|cousin|brother|sister|"
    r"friend|family|relative|aunt|uncle|daughter|son|children|child|niece|nephew|"
    r"grandparents?|grandmother|grandfather|roommate|sibling)\b", re.I
)
_FAMILY = r"(?:mum|mom|mother|dad|father|parents?|family|partner|husband|wife|spouse|flatmate|housemate|roommate|cousin|brother|sister|siblings?|friends?|relatives?|aunt|uncle|daughters?|sons?|children|child|nieces?|nephews?|grandparents?|grandmother|grandfather)"
_CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "mustn't": "must not", "needn't": "need not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "there's": "there is",
}
_CONTRACTION = re.compile(r"\b(?:" + "|".join(re.escape(word) for word in _CONTRACTIONS) + r")\b")
# Inflection aliases remain lexical features. The original form is retained,
# so collapsing 'paid/pay' does not erase a customer's past-tense evidence.
_INFLECTIONS = {
    form: lemma for lemma, forms in {
        "pay": ("pays", "paid", "paying"),
        "use": ("uses", "used", "using"),
        "fund": ("funds", "funded", "funding"),
        "finance": ("finances", "financed", "financing"),
        "cover": ("covers", "covered", "covering"),
        "meet": ("meets", "met", "meeting"),
        "provide": ("provides", "provided", "providing"),
        "spend": ("spends", "spent", "spending"),
        "save": ("saves", "saved", "saving"),
        "receive": ("receives", "received", "receiving"),
        "earn": ("earns", "earned", "earning"),
        "withdraw": ("withdraws", "withdrew", "withdrawn", "withdrawing"),
        "draw": ("draws", "drew", "drawn", "drawing"),
        "sell": ("sells", "sold", "selling"),
        "transfer": ("transfers", "transferred", "transferring"),
        "send": ("sends", "sent", "sending"),
        "give": ("gives", "gave", "given", "giving"),
        "support": ("supports", "supported", "supporting"),
        "help": ("helps", "helped", "helping"),
        "contribute": ("contributes", "contributed", "contributing"),
        "stop": ("stops", "stopped", "stopping"),
        "start": ("starts", "started", "starting"),
        "end": ("ends", "ended", "ending"),
        "finish": ("finishes", "finished", "finishing"),
        "arrive": ("arrives", "arrived", "arriving"),
        "come": ("comes", "came", "coming"),
        "go": ("goes", "went", "going"),
        "keep": ("keeps", "kept", "keeping"),
    }.items() for form in forms
}
_PAIR_WORDS = frozenset({
    "pay", "fund", "finance", "cover", "meet", "use", "spend", "save", "put", "aside",
    "receive", "keep", "untouched", "not", "no", "none", "never", "nothing", "stop", "end",
    "start", "finish", "before", "previously", "formerly", "past", "old", "earlier", "now",
    "currently", "present", "today", "yet", "later", "future", "will", "might", "may", "next",
    "all", "both", "also", "too", "only", "regular", "every", "each", "cost", "costs",
    "expense", "expenses", "bill", "bills", "food", "rent", "living", "household", "support",
    "send", "give", "contribute", "help", "reinvest", "withdraw", "draw", "sell", "transfer",
    "earn", "job", "own", "my", "our", "her", "his", "their", "me", "him", "them",
    "self", "alone", "solely", "either", "or", "unknown", "unsure", "can", "cannot",
})
# Cross-source interactions use predicates, negation, time and coordination.
# Possessives and expense nouns stay in each source's raw/concept features;
# pairing 'my x my' or 'bills x my' mostly encodes sentence complexity.
_PAIR_CUES = frozenset({
    "pay", "fund", "finance", "cover", "meet", "use", "spend", "provide",
    "save", "put", "aside", "receive", "keep", "untouched", "not", "no",
    "none", "never", "nothing", "stop", "end", "start", "finish", "before",
    "previously", "formerly", "past", "old", "earlier", "now", "currently",
    "present", "today", "yet", "later", "future", "will", "might", "may",
    "next", "all", "both", "also", "too", "only", "send", "give", "contribute",
    "help", "reinvest", "withdraw", "draw", "sell", "transfer", "earn", "alone",
    "solely", "either", "or", "unknown", "unsure", "can", "cannot",
})
_CONCEPT_WORDS = {
    **{word: "funding_word" for word in
       ("pay", "fund", "finance", "cover", "meet", "use", "spend", "contribute", "provide")},
    **{word: "cost_word" for word in (
        "cost", "costs", "bill", "bills", "expense", "expenses", "outgoing", "outgoings",
        "food", "groceries", "housing", "accommodation", "rent", "mortgage", "heating",
        "transport", "travel", "electricity", "gas", "water", "broadband", "necessity",
        "necessities", "essential", "essentials", "spending", "shop", "shopping",
    )},
}
_CONCEPT_KEEP = _PAIR_WORDS | frozenset({
    "i", "you", "we", "he", "she", "it", "they", "the", "a", "an", "some", "of", "for",
    "from", "to", "on", "with", "at", "by", "as", "in", "and", "but", "while", "whereas",
    "because", "that", "this", "these", "those", "so", "than", "rather", "be", "is", "are",
    "was", "were", "been", "am", "do", "does", "did", "have", "has", "had", "month", "year",
    "week", "day", "winter", "spring", "summer", "autumn", "daily", "weekly", "monthly",
})
_CONCEPT_VALUES = frozenset(_CONCEPT_WORDS.values())


def _pair_context(tokens: Sequence[str], position: str) -> set[str]:
    """Bounded raw lexical interactions, not activity/financial predicates."""
    return {f"{position}:{_INFLECTIONS.get(word, word)}" for word in tokens
            if _INFLECTIONS.get(word, word) in _PAIR_CUES}


def _concept_tokens(text: str) -> list[str]:
    # A small finite vocabulary avoids adding another full lexical vocabulary.
    # Unknown content becomes a position placeholder only in this extra channel;
    # all original words, ownership anchors and time evidence remain elsewhere.
    return [_CONCEPT_WORDS.get(word, word if word in _CONCEPT_KEEP else "other_word")
            for word in _lexical_tokens(text)]


def _concept_ngrams(text: str) -> list[str]:
    return [gram for gram in _ngrams(_concept_tokens(text), 3)
            if _CONCEPT_VALUES.intersection(gram.split())]


def _concept_pair_context(tokens: Sequence[str], position: str) -> set[str]:
    # Position-independent supplemental cues let 'pay bills' in a source
    # clause share evidence with 'I use them' in its source-free continuation.
    # Raw pair features retain the original left/right/clause distinction.
    return {f"context:{_CONCEPT_WORDS.get(word, word)}" for word in
            (_INFLECTIONS.get(token, token) for token in tokens)
            if word in _PAIR_CUES}

# A deliberately lexical inventory. A pension balance still mentions pension;
# an unused salary still mentions salary. Such mentions are not financial facts.
_SOURCE_PATTERNS = {
    "salary": re.compile(
        r"\b(?:salar(?:y|ies|ied)|wages?|pay\s*packets?|pay\s*cheques?|paychecks?|"
        r"paycheques?|employment (?:income|pay)|paid employment|employees?|"
        r"(?:pay|earnings?|earned) (?:from|at) (?:my |a |the )?(?:job|work|employment)|"
        r"monthly pay|work income|employers?|employed job)\b", re.I
    ),
    "self_employment": re.compile(
        r"\b(?:self[- ]?employ\w*|freelanc\w*|sole trader|business(?: income| earnings)?|"
        r"clients?|contract work|working for myself|invoic\w*|own design work|"
        r"independent (?:\w+ )?(?:work|mechanic|bookkeep\w*|consult\w*|trad\w*)|"
        r"(?:my |the )?own (?:\w+ )?(?:shop|cafe|café|service|firm|practice)|"
        r"(?:shop|cafe|café|service|firm) (?:that )?i (?:run|own))\b", re.I
    ),
    "pension": re.compile(r"\b(?:pensions?(?: payments?| income)?|retirement income)\b", re.I),
    "support": re.compile(
        rf"\b(?:benefits?|universal credit|regular (?:financial )?(?:help|support|assistance)|financial support|"
        rf"support from|help from|support(?:s|ed|ing)? me|help(?:s|ed|ing)? me|{_FAMILY})\b", re.I
    ),
    "withdrawals": re.compile(
        r"\b(?:savings?|investments?|reserves?|pots?|withdraw\w*|draw(?:n|ing)? (?:down|from)|"
        r"drawdowns?|dipp?ing into|dip into|saved cash|investment sales?|"
        r"selling (?:my )?(?:shares|funds|assets)|fund units|sale(?:s)? of (?:my )?(?:shares|funds|investments))\b", re.I
    ),
    "other_regular": re.compile(
        r"\b(?:rent(?:al)? (?:income|receipts?|payments?)|rent (?:from|i receive|received|paid to me)|"
        r"rents? out|renting out|tenants?|lodgers?|royalt\w*|dividends?|annuities|annuity|"
        r"licen[cs]e (?:fees|payments?|income)|parking (?:space|income)|lease payments?|"
        r"(?:recurring|monthly|weekly) rent|(?:let|letting) out (?:my |a |the |our )?(?:\w+ )?"
        r"(?:room|property|flat|apartment|house|studio|land|cottage|workshop|garage|space))\b", re.I
    ),
}


@dataclass(frozen=True)
class SourceMention:
    source: str
    start: int
    end: int


def _normalise(text: str) -> str:
    return text.lower().replace("’", "'")


def source_mentions(text: str) -> tuple[SourceMention, ...]:
    """Return lexical mention spans, without activity or ownership decisions."""
    lower = _normalise(text)
    mentions = [SourceMention(source, match.start(), match.end())
                for source, pattern in _SOURCE_PATTERNS.items()
                for match in pattern.finditer(lower)]
    return tuple(sorted(mentions, key=lambda mention: (mention.start, mention.end, mention.source)))


def _ngrams(tokens: Sequence[str], max_n: int = 2) -> list[str]:
    return [" ".join(tokens[start:start + size])
            for size in range(1, max_n + 1) for start in range(len(tokens) - size + 1)]


def _expand_contractions(text: str) -> str:
    return _CONTRACTION.sub(lambda match: _CONTRACTIONS[match.group()], text)


def _lexical_tokens(text: str) -> list[str]:
    return [_INFLECTIONS.get(token.group(), token.group()) for token in _TOKEN.finditer(text)]


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for boundary in _BOUNDARY.finditer(text):
        if start < boundary.start():
            spans.append((start, boundary.start()))
        start = boundary.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


class D5Analyzer:
    """Original word features plus raw, source-conditioned lexical evidence.

    Clause and window representations deliberately overlap. Clauses preserve
    coordination; small position-aware windows distinguish competing sources.
    A source-free following clause supplies raw anaphoric context, rather than
    assigning a pronoun an inferred financial meaning.
    """

    def __call__(self, text: str) -> list[str]:
        raw = _normalise(text)
        features = list(_WORDS(raw))
        lower = _expand_contractions(raw)
        features.extend(f"lexical:{gram}" for gram in _ngrams(_lexical_tokens(lower), 3))
        features.extend(f"concept:{gram}" for gram in _concept_ngrams(lower))
        mentions = source_mentions(lower)
        if not mentions:
            return features
        tokens = list(_TOKEN.finditer(lower))
        clauses = _clause_spans(lower)
        sources = sorted({mention.source for mention in mentions})
        pair_contexts: dict[str, set[str]] = {source: set() for source in sources}
        concept_pair_contexts: dict[str, set[str]] = {source: set() for source in sources}
        source_concepts: dict[str, set[str]] = {source: set() for source in sources}
        features.extend(f"alias:{source}" for source in sources)
        features.extend(f"mentioned_pair:{left}:{right}" for left, right in combinations(sources, 2))

        # Canonical lexical bigrams share vocabulary between wages/paypacket,
        # etc. Raw originals remain present above. This is alias substitution,
        # not a choice of financial answer.
        canonical = []
        previous = None
        for token in tokens:
            covering = [mention for mention in mentions
                        if mention.start <= token.start() and token.end() <= mention.end]
            word = f"mention_{covering[0].source}" if covering else _INFLECTIONS.get(token.group(), token.group())
            if not covering or previous != (covering[0].source, covering[0].start, covering[0].end):
                canonical.append(word)
            previous = (covering[0].source, covering[0].start, covering[0].end) if covering else None
        features.extend(f"canonical:{gram}" for gram in _ngrams(canonical, 3))

        for mention in mentions:
            clause_index = next((i for i, (start, end) in enumerate(clauses)
                                 if start <= mention.start < end), None)
            if clause_index is None:
                continue
            start, end = clauses[clause_index]
            clause = lower[start:end]
            features.extend(f"clause:{mention.source}:{gram}" for gram in _WORDS(clause))
            features.extend(f"clauselex:{mention.source}:{gram}"
                            for gram in _ngrams(_lexical_tokens(clause), 3))
            features.extend(f"clauseconcept:{mention.source}:{gram}"
                            for gram in _concept_ngrams(clause))
            source_concepts[mention.source].update(_concept_ngrams(clause))
            clause_tokens = [token for token in tokens if start <= token.start() < end]
            before = [token.group() for token in clause_tokens if token.end() <= mention.start][-6:]
            after = [token.group() for token in clause_tokens if token.start() >= mention.end][:10]
            features.extend(f"left:{mention.source}:{gram}" for gram in _ngrams(before))
            features.extend(f"right:{mention.source}:{gram}" for gram in _ngrams(after))
            features.extend(f"leftlex:{mention.source}:{gram}"
                            for gram in _ngrams([_INFLECTIONS.get(word, word) for word in before], 3))
            features.extend(f"rightlex:{mention.source}:{gram}"
                            for gram in _ngrams([_INFLECTIONS.get(word, word) for word in after], 3))
            pair_contexts[mention.source].update(_pair_context(before, "left"))
            pair_contexts[mention.source].update(_pair_context(after, "right"))
            pair_contexts[mention.source].update(_pair_context(_lexical_tokens(clause), "clause"))
            concept_pair_contexts[mention.source].update(_concept_pair_context(before, "left"))
            concept_pair_contexts[mention.source].update(_concept_pair_context(after, "right"))
            concept_pair_contexts[mention.source].update(_concept_pair_context(_lexical_tokens(clause), "clause"))
            for offset, token in enumerate(reversed(before), start=1):
                features.append(f"position:{mention.source}:left{min(offset, 4)}:{token}")
            for offset, token in enumerate(after, start=1):
                features.append(f"position:{mention.source}:right{min(offset, 4)}:{token}")
            for owner in _OWNER.finditer(lower[max(start, mention.start - 40):mention.start]):
                features.append(f"owner_word:{mention.source}:{owner.group()}")

            # Carry raw next-clause context only across a source-free clause.
            # This covers 'my wages arrive; I save them' without guessing that
            # a named source is ACTIVE/INACTIVE or resolving financial labels.
            if clause_index + 1 < len(clauses):
                next_start, next_end = clauses[clause_index + 1]
                if not any(next_start <= other.start < next_end for other in mentions):
                    features.extend(f"following:{mention.source}:{gram}"
                                    for gram in _WORDS(lower[next_start:next_end]))
                    features.extend(f"followinglex:{mention.source}:{gram}"
                                    for gram in _ngrams(_lexical_tokens(lower[next_start:next_end]), 3))
                    pair_contexts[mention.source].update(
                        _pair_context(_lexical_tokens(lower[next_start:next_end]), "following"))
                    features.extend(f"followingconcept:{mention.source}:{gram}"
                                    for gram in _concept_ngrams(lower[next_start:next_end]))
                    source_concepts[mention.source].update(
                        _concept_ngrams(lower[next_start:next_end]))
                    concept_pair_contexts[mention.source].update(
                        _concept_pair_context(_lexical_tokens(lower[next_start:next_end]), "following"))
        # Pool the same lexical grams across a source clause and a source-free
        # continuation. This supplies learnable anaphoric context without
        # assigning a pronoun a source, activity flag, or financial outcome.
        for source, grams in source_concepts.items():
            features.extend(f"sourceconcept:{source}:{gram}" for gram in sorted(grams))
        # Linear LR needs observed feature interactions to learn that two
        # sources with particular *wording* differ from one funded source and
        # an unused mention. No rule decides whether either member is active;
        # 'pay × not', 'pay × pay', ownership and timing words all reach LR.
        for left, right in combinations(sources, 2):
            for left_context in sorted(pair_contexts[left]):
                for right_context in sorted(pair_contexts[right]):
                    # Share the observed lexical relation across all category
                    # pairs. The original source-specific raw features remain,
                    # including ownership evidence for family transfers. This
                    # key asserts only that two distinct lexical source names
                    # occurred with these words; it never counts contributors.
                    first, second = sorted((left_context, right_context))
                    features.append(f"pairlex:{first}|{second}")
            for left_context in sorted(concept_pair_contexts[left]):
                for right_context in sorted(concept_pair_contexts[right]):
                    first, second = sorted((left_context, right_context))
                    features.append(f"pairconcept:{first}|{second}")
        return features


class D5TextClassifier(BoundedTextClassifier):
    """The existing classifier interface with learned D5 lexical interactions."""

    def __init__(self, random_state: int = 42, *, C: float = 1.0,
                 min_df: int = 1, binary: bool = False) -> None:
        super().__init__(random_state=random_state)
        self.C = C
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer=D5Analyzer(), sublinear_tf=not binary,
                                     binary=binary, min_df=min_df)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state,
                                               class_weight="balanced", C=C)),
        ])
        self._raw_vocabulary: frozenset[str] = frozenset()

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "D5TextClassifier":
        super().fit(texts, labels)
        self._raw_vocabulary = frozenset(gram for text in texts for gram in _WORDS(_normalise(text)))
        return self

    def raw_lexical_support(self, text: str):
        return raw_lexical_support(self, text)


def raw_lexical_support(classifier: D5TextClassifier, text: str):
    """Original-vocabulary overlap; aliases cannot make unknown text supported."""
    # Local import avoids a cycle when the all-question loader selects D5.
    from ai.all_question_ml import LexicalSupport

    classifier._require_fitted()
    possible = set(_WORDS(_normalise(text)))
    active = len(possible & classifier._raw_vocabulary)
    return LexicalSupport(active, active / len(possible) if possible else 0.0)
