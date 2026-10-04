# InvestWise: all-question ML prototype

All 15 questions now use question-specific TF-IDF + Logistic Regression models for free-text interpretation. Each question has an intent model with exactly three labels (`clarity`, `undecided`, `confusion`) and a model constrained to that question's approved financial options.

## Runtime flow

Free text → explicit safety/control checks → intent model → intent conformal set → fixed-option model → option conformal set → business validation → user confirmation.

Both prediction sets must be singletons, with intent `clarity`, before the system proposes a financial option. Empty or multi-label sets cause abstention: no argmax fallback and no default middle category. Model uncertainty is an internal unresolved status, not a fourth training label or evidence that the user is undecided/confused.

Business checks can block contradictions, invalid premises, missing currency and boundary disagreements. They cannot create or replace a model proposal. The archived bounded parser remains a validation/comparison tool; it does not supply free-text answers in the app's default ML mode. Financial follow-up evidence uses the same ML option gates, with deterministic resource/usability and consistency checks.

Visible answer shortcuts represent deliberate user selections and still require confirmation. Typing the label is free text and goes through ML. Explicit help, pause/stop, accessibility, urgent safety, confirmation/correction, final accuracy and save-consent controls remain deterministic.

## Data and reproducibility

`data/nlp/all_questions/` contains newly authored fictional training, calibration and evaluation examples. No real financial customer data is used. IDs, question IDs, fixed-option IDs, semantic buckets, provenance, groups and split assignments are retained. Every option and all three buckets must occur in training and calibration; normalized replies and groups cannot cross those partitions.

The model factory reads only `train.csv` and `calibration.csv`. It caches fitted predictors by source hashes and the option manifest; it never selects settings using `test.csv`. New predictors use alpha 0.10, fixed before evaluation. The finite-sample wrapper returns the full label set when the conformal rank exceeds the available calibration sample. Vocabulary-overlap checks add abstention but are not a statistical safety guarantee.

The previous D12 datasets, model implementation and alpha 0.20 configuration remain available unchanged through the explicitly named archived hybrid comparison factory. The app default uses the new all-question registry, including D12. Historical D12 metrics do not describe the new registry.

## Confirmation and care

An ML option is only a candidate. The user must confirm or correct its exact meaning. Corrections invalidate affected conclusions and final accuracy/save permissions. The final playback shows the latest answers, separately reports time horizon, practical capacity for loss in the 20% scenario, stated risk preference and the comfort qualifier, and requires both accuracy confirmation and explicit save consent.

Three factual clarification attempts end with pause/finish-incomplete and a clearly labelled simulated human-support route. Help and accessibility requests do not consume that counter. Urgent safety language stops profiling and saving before model inference. Ordinary distress offers pacing before confirmation. No live agent is contacted and no investment advice is produced.

Existing live sessions retain their previous confirmations and audit provenance when code/models refresh. Future replies use ML; earlier answers are not silently relabelled. Start a new fictional conversation to demonstrate ML from Question 1.

## Test and run

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate_all_question_ml.py --help
.venv/bin/python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

New resolver and integration tests cover both singleton gates, no legacy fallback, all 15 routes, explicit selections, boundary vetoes, urgent safety and clarification exhaustion. Existing policy tests explicitly use `build_legacy_comparison_engine`; their pass count is not all-question ML accuracy evidence. The independent evaluator reports resolver predictions and actual-engine accepted candidates separately, including false Clarity and wrong financial options.

## Evidence limits

Conformal coverage is a marginal property under appropriate exchangeability assumptions; it is not a guarantee of correct singleton predictions or zero false Clarity. Two gated models and extra vetoes do not create an overall 90% correctness guarantee. Independently authored fictional examples remain a small, correlated prototype dataset, not a representative real-user sample. Five-peer comprehension testing and mentor review of the financial boundaries, the 20% example, category wording and clarification limit remain required before broader claims.

The [evaluation record](work/all-question-ml-2026-10-04/EVALUATION_README.md) preserves an initial evaluation, a provenance recheck and a policy repair on the same known test set. The final reused-test run staged 19 correct candidates from 204 fictional replies and abstained on 185 (90.69%). The raw models still proposed one false Clarity and one wrong option; business vetoes blocked both. This is not fresh validation of the repaired system, and the high abstention rate remains a usability limitation.

The live-browser flow check also found that `I am 45 years old` led the raw option model to the wrong age band; the business boundary veto prevented a candidate. Q8 `none` produced the correct pending ML candidate and awaited confirmation. These known smoke cases establish wiring and repair behavior, not representative accuracy. The automated suite has 519 passing checks, including seven actual-default-model UI checks; the suite also contains archived hybrid policy checks.
