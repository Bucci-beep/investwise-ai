"""Checks for the embedding safety layer.

The logic tests use a tiny fake encoder, so they run anywhere without
downloading a model. The real-model tests run only when
sentence-transformers is installed (pip install -r requirements-safety.txt).
"""

import numpy as np
import pytest

from ai.safety_embeddings import EmbeddingNet, embeddings_available, screen_with_embeddings

# Fake 3-dimensional "meanings": danger, distress, ordinary.
FAKE_VECTORS = {
    "i want to end my life": [1.0, 0.0, 0.0],
    "i feel overwhelmed": [0.0, 1.0, 0.0],
    "my salary pays the bills": [0.0, 0.0, 1.0],
    "sometimes i think about not waking up": [0.9, 0.1, 0.0],
    "the weather is nice": [0.0, 0.2, 0.98],
}


def fake_encoder(texts):
    vectors = np.array([FAKE_VECTORS[t.lower()] for t in texts], dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


EXEMPLARS = [
    ("urgent", "I want to end my life"),
    ("care", "I feel overwhelmed"),
    ("benign", "My salary pays the bills"),
]


def test_close_meaning_is_flagged_with_the_nearest_category():
    net = EmbeddingNet(EXEMPLARS, fake_encoder, threshold=0.5)
    result = net("Sometimes I think about not waking up")
    assert result.category == "urgent"
    assert result.layer == "embedding"


def test_message_closest_to_a_harmless_example_is_not_flagged():
    net = EmbeddingNet(EXEMPLARS, fake_encoder, threshold=0.1)
    assert net("The weather is nice").category is None


def test_threshold_blocks_weak_matches():
    net = EmbeddingNet(EXEMPLARS, fake_encoder, threshold=0.999)
    assert net("Sometimes I think about not waking up").category is None


def test_screen_combines_layers_and_keeps_most_serious():
    net = EmbeddingNet(EXEMPLARS, fake_encoder, threshold=0.5)
    # The wider rules alone already catch this one as care; the embedding layer adds nothing worse.
    assert screen_with_embeddings("I feel overwhelmed", net).category == "care"
    assert screen_with_embeddings("Sometimes I think about not waking up", net).category == "urgent"


real_model = pytest.mark.skipif(not embeddings_available(), reason="sentence-transformers not installed")


@real_model
def test_real_model_produces_unit_vectors():
    from ai.safety_embeddings import default_encoder

    vectors = default_encoder()(["test sentence", "another one"])
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)