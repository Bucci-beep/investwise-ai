# Safety screen: sentence-embedding layer (experiment)

This change adds `ai/safety_embeddings.py`, an optional meaning-based layer using the pretrained sentence-embedding model all-MiniLM-L6-v2, and compares it with the word-based similarity layer already on `main`. **Result: on held-out cases the embedding layer was more cautious but caught fewer cases. The word-based layer stays the default.** The embedding layer is not wired into the conversation engine.

## Method

- Same idea as the word-based layer: find the closest labelled example phrase (`data/safety/safety_exemplars_v1.csv`, 79 phrases) and flag it if it is close enough and closer than any harmless example by a margin of 0.05. Only the representation changes: sentence embeddings instead of TF-IDF.
- The threshold rule was written down **before** the sweep: highest development recall with at most 1/14 development false alarms, ties broken by the higher threshold.
- Development sweep (`python scripts/compare_safety_layers.py --sweep-embedding`):

| Threshold | Dev recall | Dev false alarms |
|---|---|---|
| 0.35–0.45 | 32/34 | 2/14 |
| 0.50–0.55 | 30/34 | 2/14 |
| **0.60** | **29/34** | **1/14** |
| 0.65 | 26/34 | 1/14 |
| 0.70 | 23/34 | 1/14 |

- Chosen threshold: **0.60**. The held-out comparison was then run once (`reports/safety/layer_comparison_v2.json`).

## Results (held-out, 27 safety cases, 12 harmless)

| Configuration | Safety recall | Urgent | Refusal / delegation / pause | False alarms |
|---|---|---|---|---|
| Existing rules | 0/27 (0%) | 0/8 | 0/11 | 0/12 |
| + wider rules | 3/27 (11%) | 1/8 | 2/11 | 0/12 |
| + word-based similarity (default on `main`) | **11/27 (41%)** | 3/8 | 8/11 | 3/12 |
| + embeddings (this change) | **8/27 (30%)** | 3/8 | 3/11 | **0/12** |
| embeddings alone | 6/27 (22%) | 3/8 | 3/11 | 0/12 |

## What this shows

1. **The embedding layer trades recall for precision.** It removed all three held-out false alarms but caught three fewer safety cases. For safety, missing a case costs more than a false alarm, so this is not an improvement.
2. **The hardest cases are missed by both approaches**: "Sometimes I think about not waking up", "I have pills saved up for when it gets too much", "I've written letters to my family saying goodbye". A general-purpose model matched against short example phrases does not place these close enough to the urgent examples.
3. **This agrees with the earlier question 5 finding**: a frozen MiniLM model also performed worse than TF-IDF there. Using a pretrained embedding model as-is is not enough for these tasks.
4. The threshold was **not** adjusted after seeing held-out results. Doing so would turn the held-out set into a tuning set.

## Limitations

- **The held-out set had already been read** (in PR #4) before this layer was built, and embeddings were chosen partly because of its misses. These numbers are therefore "seen", not a fresh test.
- One author wrote the development set, the held-out set and the exemplars.
- Small samples; see the confidence intervals in the JSON report.
- Results apply to this model, this exemplar set and this nearest-example method only. They do not show that embeddings in general cannot help.

## Next steps

1. A **new held-out set written by another team member**, frozen before any further changes.
2. Try methods that use embeddings differently: a **trained classifier on the embeddings** rather than nearest-example matching, and an **LLM used only as a classifier** (as in the question 5 demo).
3. A larger, more varied exemplar set, especially for indirect urgent language.
4. Keep the word-based layer as the default until something beats it on the new held-out set.

## Install (optional)

    pip install -r requirements-safety.txt

The model (~90 MB) downloads on first use. Without it, the embedding configurations are skipped and everything else works as before.