# Continuation prompt: improve InvestWise's free-text interpretation

Copy the prompt below into the coding assistant on the other computer. The branch is the tested checkpoint; create a new experiment branch from it. It intentionally does not merge the separate AI-engine branch.

---

You are continuing InvestWise AI, a UKFinnovator Bristol Challenge 2 prototype. Inspect and improve the existing repository; do not rebuild the app or replace its business specification.

## Objective and authorised scope

Help a nervous first-time investor answer bounded financial questions in their own words. Improve the proportion of genuinely clear replies that receive the correct financial candidate without increasing false Clarity or wrong-option proposals. Retain TF-IDF + Logistic Regression + conformal prediction for **all 15 questions**. Investigate better labelled examples first, then a small word-plus-character feature experiment. Implement a controlled experiment, verify it, and show results before making it the app default.

This is a prototype using fictional situations. It must not recommend an investment, portfolio, product, transaction or personalised financial action. Attitude to risk, capacity for loss and time horizon remain separate conclusions. Age, experience or wealth alone must not determine risk preference or suitability. Financial boundaries remain draft business decisions requiring mentor review.

## Get the exact starting point

Repository: https://github.com/Bucci-beep/investwise-ai.git

Checkpoint branch: `feature/all-question-ml-prototype`

On a new checkout:

```bash
git clone --branch feature/all-question-ml-prototype --single-branch https://github.com/Bucci-beep/investwise-ai.git
cd investwise-ai
git log -1 --oneline
git status --short
git switch -c experiment/semantic-coverage
```

If the folder already exists, inspect its remote and working-tree changes first. Fetch the checkpoint branch and use a clean checkout/worktree; do not overwrite, reset or discard existing work. Record the starting commit hash. The checkpoint was developed from `e0ba4d6`. A separate remote `feature/ai-risk-engine` advanced to `629a396` before this checkpoint was published; do not silently merge that different runtime into the experiment.

Use Python 3.12, preferably the tested 3.12.14. For macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-snapshot.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

For Windows PowerShell, use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`; the remaining `python` commands are the same. Do not copy another computer's `.venv` or hardcode its paths. The pinned snapshot records 46 packages from the tested environment, including scikit-learn 1.9.1, Streamlit 1.65.0, numpy 2.5.3, pandas 3.0.6 and scipy 1.18.1. Verify wheel availability. If these versions cannot install on the new OS, report and document substitutions and rerun checks; do not claim exact metric reproduction.

## Read these sources first

1. `README_IMPLEMENTATION.md`: current architecture and safeguards.
2. `config/business-decision-spec.json`: authoritative current question wording, order, fixed options, follow-ups, financial meanings and mentor-review status. Preserve the existing D1–D15 IDs.
3. `ai/all_question_ml.py`: the actual default ML resolver and conformal/lexical gates.
4. `ai/text_classifier.py` and `ai/conformal.py`: shared classifier and original conformal implementation. The all-question resolver uses a conservative finite-sample wrapper.
5. `ai/conversation_factory.py`, `ai/conversation_engine.py`, `ai/bounded_answers.py`, `ai/pre_model_controls.py`, `ai/horizon_mapper.py`: integration, controls and business vetoes.
6. `frontend/app.py`: Streamlit chat UI and state refresh.
7. `data/nlp/all_questions/README.md`, `provenance.json`, `dataset_audit.json`: synthetic data, split roles and limits.
8. `work/all-question-ml-2026-10-04/EVALUATION_README.md` and its three preserved reports: what has actually been evaluated.
9. `tests/test_all_question_ml.py`, `tests/test_ml_conversation.py`, `tests/test_all_question_ml_ui.py`: gate, integration and actual-default-model UI checks.

Do not assume the original PDFs, local business-example bank, phone screenshots or private conversation exports are available on this computer. Executable paths are repository-relative. Historical source-preview metadata is provenance, not a runtime dependency or permission to access another computer.

## Current behaviour: preserve it

`build_conversation_engine()` defaults to all-question ML. Each question has two fitted TF-IDF/LR models: three-way intent and its approved fixed financial options. Per-question models supply the question context.

The intent labels are exactly `clarity`, `undecided`, `confusion`; off-topic/meaning-help belongs to confusion. Model uncertainty is a separate internal abstention condition, not a fourth training label and not evidence that the person is undecided.

Free-text path:

```text
Explicit safety/control checks
→ question-specific intent model + conformal set
→ fixed-option model + conformal set
→ business validation
→ pending candidate
→ exact user confirmation or correction
```

Both sets must be singletons: intent `{clarity}` and one approved option. Empty or multiple-label sets abstain. Never use argmax, the middle category, a similarity match or the old bounded parser as a fallback financial answer. Business checks may veto a proposed meaning or inspect material consistency; they cannot create or replace the ML proposal.

Visible answer shortcuts are deliberate user selections and still need confirmation. Typing an option ID or label is free text and must use ML. Do not hide poor model performance by quietly routing typed answers through business rules.

Keep deterministic urgent-safety, stop/pause/resume, help/accessibility, confirmation/correction and save-consent controls. Ordinary distress gets acknowledgement and pacing control before confirmation. Possible urgent self-harm danger stops profiling and saving before inference. Any human handover is clearly labelled **simulated**; no real agent has been contacted. Three factual clarification attempts end in pause/finish-incomplete, not a forced answer. Help, pauses and accessibility requests do not spend those factual attempts.

Confirmation and saving are different events. Review all 15 answers at the end, including declined/unresolved entries. Corrections invalidate affected confirmations, conclusions and final accuracy/save permissions, and playback must show the newest version. Require final accuracy confirmation **and separate explicit save consent** before recording a completed profile. Do not alter or auto-confirm an existing user's conversation during an experiment.

## Baseline and evidence limits

The fictional v1 data contains 552 train, 660 calibration and 204 test rows, covering 15 questions and 72 fixed options. Fit vocabulary and coefficients on training only. Current all-question settings were fixed before the first test: word unigrams/bigrams, balanced LR, random state 42, alpha 0.10, minimum vocabulary-feature coverage 0.10. The old D12-only comparison uses alpha 0.20; do not confuse its historical results with the new model.

Final dataset SHA-256:

```text
train.csv       96543fe7cc72d8a27403ead432fcf81971b0b93b28a5ed77f9e3770fbda83abf
calibration.csv 3b63ebddf80f8b5758bcbef9cb8d662741e68de38660455dcc4649106d5792be
test.csv        18d0dd2ba9fdbe7fa37b9b5cb27be2bc09be435f472341a12b1050c2dfe029fe
```

The **known reused** 204-case post-policy report staged 19 correct engine candidates, abstained on 185 (90.69%), and recorded zero observed false Clarity/wrong-option candidates. The raw resolver still proposed one false Clarity and one wrong financial option; business checks blocked them. An earlier administrative recheck had one accepted false Clarity before the ownership repair. Preserve that failure and all reports. This is not new independent validation, a guarantee of safety or real-user accuracy. D5, D9, D10, D12 and D15 produced no accepted candidates in that single-turn evaluation.

There were 519 passing automated checks at the checkpoint, including seven actual-default-model UI checks. Several older tests explicitly use `build_legacy_comparison_engine`; their passing count is not evidence of all-question ML accuracy.

Known diagnostic cases:

- D8 `none` → correct pending `D8_NONE` through ML; no confirmation yet.
- D8 `probably none` → unresolved; no financial candidate.
- D8 `none of these choices fits` → clarify whether no experience or an unlisted type.
- D8 an uncle owns bitcoin while the customer's holdings are unstated → ownership veto, not inferred crypto experience.
- D2 `I am 45 years old` → raw option model proposed the wrong age band; business boundary veto prevented acceptance.
- D9 six qualifying transactions → raw model once proposed zero; boundary veto blocked it.

TF-IDF does not reliably learn numeric comparisons merely by seeing a few numeral strings. Character features may improve spelling/word variants; they do not solve numerical ranges, negation, ownership, timeframe or conditional meaning by themselves. Preserve the numeric and business vetoes.

## Controlled improvement plan

1. Reproduce the environment and run the policy/UI tests. Inspect existing reports; do not rerun the known test just to call it fresh. Measure training/development errors and abstention reasons by question, intent gate, option gate and business veto.
2. Create versioned **development/training** additions for one or two priority questions first. Do not overwrite v1 train/calibration/test or rerun the dataset generator blindly: that script writes all three original splits and can replace edited data.
3. Add distinct, reviewed natural examples for every relevant option and all three intent buckets. Include short answers, synonyms, contractions, typos, messy punctuation and harmless extra chatter. Add paired contrasts whose meaning changes: `none`/`probably none`; `my shares`/`my uncle's shares`; earliest need/ideal goal; currently held/future plan; no trades/no experience; new contributions/current balance; guaranteed recovery/possible permanent loss. Uncertainty and confusion have null option IDs. Do not treat emotion, age or a risk word alone as a financial label.
4. Synthetic augmentation requires business-label review. Keep `source`, fictional status, IDs, question IDs and family IDs. Keep all paraphrases/near-duplicates from one source family in the same partition. Check normalized exact overlap and near-duplicate leakage; group-ID separation alone does not prove semantic independence. Never derive training examples from calibration/test replies.
5. Compare four variants where time permits: A current word-TF-IDF baseline; B better development data with the same features; C word-plus-character TF-IDF with original development data; D both changes. Keep LR and the runtime gates. Use a small predefined feature/regularization search on **training-derived development folds only**, rather than tuning on final test or conformal calibration results. If comparing gated outcomes during development, create internal training/calibration/validation partitions from development data; leave the final calibration set out of model selection. Report when those internal samples are too small for useful comparison.
6. For grouped validation, ensure folds contain the necessary intent/option classes. Current v1 family IDs are coarse and may leave an entire option absent from a training fold. Do not silently use row-random validation and call it independent; author multiple independent families per option or report unsupported fold comparisons.
7. If adding word/character features, isolate the experimental feature builder so the archived D12 baseline remains unchanged. Update lexical-support calculation and diagnostics correctly for the new representation; do not remove the out-of-vocabulary guard just because the pipeline no longer exposes one `tfidf` analyzer. Add meaningful gate/cache/provenance regressions for that change.
8. Freeze each selected model/configuration before recalibration. Retraining changes probabilities, so recalibrate rather than reuse old numerical thresholds. Keep alpha 0.10 for the primary comparison. **Do not relax alpha or confidence gates to make the demo advance more often.** If calibration becomes development material, reserve a new disjoint calibration set and document the loss of the old guarantee assumptions.
9. Treat v1's opened 204 cases as known regressions. Obtain a new independently authored/evaluated sealed set, distinct from training and calibration, for final comparison. Choose **one** experiment using development evidence, freeze its model/data/configuration and promotion criteria, then open the new sealed set once for the baseline-versus-selected-candidate comparison. Do not choose the winner among A–D from final-test scores, or repair/tune and call another run on that set independent. If the selected candidate fails, retain the baseline; later selection needs fresh independent validation. Ideally a separate person reviews labels before results are seen. If unavailable, report only development/known-regression results and say independent validation is pending. Publish uncertainty and small sample counts rather than a guaranteed improvement claim.
10. Exercise the real Streamlit default path with fictional free text, confirmations, corrections and final save consent. Preserve existing chats. Do not promote an experiment to default merely because it passes policy tests or stages more candidates.

## Report the right outcomes

For development comparisons of baseline versus each candidate, and the separate frozen final comparison of baseline versus the one selected candidate, report counts and denominators, both before and after business/control vetoes:

- Intent macro F1 and per-class precision/recall on properly separated evaluation data.
- Raw intent/option empirical conformal coverage, average set size, empty/multi-label/singleton rates.
- Financial candidate rate and abstention/clarification rate.
- Correct-option proposals among genuinely clear replies, and accuracy among staged candidates.
- **False Clarity:** a financial candidate proposed for a truly undecided/confused reply. Show the count, rate over all truly non-clear replies, and rate among staged candidates.
- **Wrong option:** a truly clear reply proposed in the wrong fixed financial category. Count this separately from false Clarity.
- Unsafe forced proposals: candidate despite an unresolved prediction set, veto, safety stop or missing required fact. This must remain zero in the policy checks; explicit visible selections are a different user path.
- Accidental confirmation/saving or stale playback after correction: must remain zero in the corresponding tests.
- Per-question breakdown, especially numeric questions and questions currently accepting no evaluation candidates.

An all-abstain model is not a semantic success. A less cautious model is not a success if it creates more wrong proposals. Choose based on a meaningful rise in correct clear-answer acceptance while keeping observed error counts/rates no worse on the **same separated comparison data**, with sample-size limitations stated. Conformal marginal coverage does not guarantee correct singletons or zero false Clarity, and post-veto acceptance has no automatic conformal guarantee.

Save reports to new exclusive paths under `reports/experiments/`; include model/data/config hashes, dependency versions, split status, evaluation label and error examples. Never overwrite earlier reports or relabel a reused set as unseen. The evaluator supports `--data-directory`, `--output` and `--evaluation-label`; reading test data with it is final evaluation, not a tuning loop.

## Deliverables and time limit

Produce a small experiment implementation, reviewed/versioned development data with provenance, appropriate tests, a reproducible command and a concise baseline/comparison report. State whether the evidence supports promotion; keep the existing default if improvement is not established. Explain what changed, which examples improved or failed, the trade-offs and what remains untested. Do not push or merge the new experiment unless I explicitly request it.

If only ten minutes are available, focus on one question and one controlled improvement; record actual elapsed time and finish with honest pilot results. Do not claim all 15 questions or real-customer performance have improved in that time. Start with useful investigation and implementation rather than asking me to restate this brief.
