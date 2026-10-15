"""Meaning-based safety layer using sentence embeddings.

Same idea as ``SimilarityNet`` in ``ai/safety_screen.py`` (find the closest
labelled example phrase), but each sentence is turned into a vector by a
pretrained sentence-embedding model, so sentences with similar *meaning*
end up close together even when they share no words.

Optional dependency: ``pip install -r requirements-safety.txt``. The model
(all-MiniLM-L6-v2, about 90 MB) downloads on first use. If the package is
not installed, ``embeddings_available()`` returns False and nothing else in
the app is affected.

This module is not yet wired into the conversation engine.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from ai.safety_screen import (
    SafetyResult, _most_serious, load_exemplars, rules_layer, wide_rules_layer,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chosen on the development benchmark only (scripts/compare_safety_layers.py
# --sweep-embedding): highest dev recall with at most 1/14 dev false alarms,
# ties broken by the higher threshold. 0.60 -> 29/34 recall, 1/14 false alarms.
EMBEDDING_THRESHOLD = 0.60
EMBEDDING_MARGIN = 0.05

Encoder = Callable[[Sequence[str]], np.ndarray]


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def default_encoder() -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))

    return encode


class EmbeddingNet:
    """Nearest-exemplar classifier on normalised sentence embeddings.

    ``encoder`` turns a list of texts into an (n, d) array of unit vectors.
    It is injectable so tests can run without downloading a model.
    """

    def __init__(self, exemplars: list[tuple[str, str]], encoder: Encoder,
                 threshold: float = EMBEDDING_THRESHOLD, margin: float = EMBEDDING_MARGIN):
        self.categories = [category for category, _ in exemplars]
        self.encoder = encoder
        self.threshold = threshold
        self.margin = margin
        self.matrix = encoder([text for _, text in exemplars])

    def nearest(self, text: str) -> tuple[str | None, float, float]:
        """Return (best non-benign category, its similarity, best benign similarity)."""
        sims = self.matrix @ self.encoder([text])[0]
        best_cat, best_sim, best_benign = None, 0.0, 0.0
        for category, sim in zip(self.categories, sims):
            sim = float(sim)
            if category == "benign":
                best_benign = max(best_benign, sim)
            elif sim > best_sim:
                best_cat, best_sim = category, sim
        return best_cat, best_sim, best_benign

    def __call__(self, text: str) -> SafetyResult:
        category, sim, benign = self.nearest(text)
        if category and sim >= self.threshold and sim >= benign + self.margin:
            return SafetyResult(category, "embedding", None, round(sim, 3))
        return SafetyResult(None, None, None, round(sim, 3))


@lru_cache(maxsize=1)
def default_embedding_net() -> EmbeddingNet:
    return EmbeddingNet(load_exemplars(), default_encoder())


def screen_with_embeddings(text: str, net: EmbeddingNet | None = None) -> SafetyResult:
    """Existing rules + wider rules + embedding layer; most serious category wins."""
    net = net or default_embedding_net()
    return _most_serious([rules_layer(text), wide_rules_layer(text), net(text)])