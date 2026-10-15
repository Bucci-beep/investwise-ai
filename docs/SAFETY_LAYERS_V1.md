# Safety screen v1: wider rules and a similarity net

This change adds `ai/safety_screen.py`, a layered safety screen, and measures it against a held-out set that was frozen before the new layers were written. The screen is **not yet wired into the conversation engine**; that is a separate decision once the team has seen these numbers.

## What was added

| File | Purpose |
|---|---|
| `ai/safety_screen.py` | Three layers: existing rules → wider rules → nearest-example similarity. Returns the most serious category found. |
| `data/safety/safety_exemplars_v1.csv` | 79 labelled example phrases used by the similarity layer (written separately from both test sets) |
| `data/safety/safety_holdout_v1.csv` + `.sha256` | 50 new cases, frozen at 2026-10-15T06:26Z before any new detection code was written |
| `scripts/compare_safety_layers.py` | Compares configurations on the development and held-out sets; `--sweep` tunes the threshold on development only |
| `reports/safety/layer_comparison_v1.json` | The results below |

The similarity threshold (0.35) was chosen on the development set only: highest development recall, ties broken by the higher threshold. The held-out set was then run once.

## Results

| Configuration | Development safety recall | **Held-out safety recall** | Held-out urgent | Held-out false alarms |
|---|---|---|---|---|
| Existing rules (baseline) | 4/34 (12%) | **0/27 (0%)** | 0/8 | 0/12 |
| + wider rules | 34/34 (100%) | **3/27 (11%)** | 1/8 | 0/12 |
| + similarity net | 34/34 (100%) | **11/27 (41%)** | 3/8 | 3/12 |

Reproduced independently on a second machine (Windows) with identical results.

## What this shows

1. **Development scores are misleading.** The wider rules score 100% on the set they were written against and 11% on new wording. Anything tuned by looking at test cases must be judged on a held-out set.
2. **Layers help.** Rules plus similarity raise held-out recall from 0% to 41%, at a cost of 3/12 false alarms ("I've cleared all my debts now" was flagged as money stress).
3. **Word-based methods have hit their limit.** The remaining misses have little or no word overlap with any known example: "Sometimes I think about not waking up", "Nobody would miss me if I was gone", "I have pills saved up for when it gets too much". Catching these needs a model that represents meaning (sentence embeddings or an LLM classifier), not more patterns.
4. **The similarity layer does not understand negation.** "I'm not overwhelmed at all" is flagged as care because it shares words with "I am completely overwhelmed". This is the one development false alarm and a known limitation.
5. **Urgent recall is 3/8 on held-out.** This is not good enough to rely on and must not be described as a working crisis detector.

## Next step

Replace the TF-IDF similarity layer with a sentence-embedding backend behind the same interface, re-tune the threshold on development only, and test on a **new** held-out set written by a different team member. This held-out set has now been read, so it should not be used to tune anything further.

## Limitations

- One author wrote the development set, the held-out set and the exemplars, so they are correlated. A held-out set written by someone else is the next fairness check.
- Small samples: read the confidence intervals in the JSON report, not just the rates.
- Category detection only; response wording and escalation still need expert review.