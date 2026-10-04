# All-question ML evaluation, 4 October 2026

All 15 question interpreters use question-specific TF-IDF and Logistic Regression models, with separate split-conformal intent and fixed-option gates. Settings were fixed before test evaluation: alpha 0.10, word unigrams/bigrams, balanced class weights, random state 42. A recognized single word can pass the lexical guard; the guard adds no coverage guarantee. Confirmation and final save consent are separate application decisions.

The dataset contains 552 training, 660 calibration and 204 test examples. All examples are fictional, authored for this prototype. Each split covers 15 questions and all 72 fixed options. These results do not establish human comprehension or real-customer validity.

| Preserved execution | Data and test status | Raw resolver candidates | Engine candidates after controls and business veto | Engine false Clarity |
| --- | --- | --- | --- | --- |
| [First report](first-independent-heldout.json) and [status correction](first-independent-heldout.status.json) | Pre-finalization training source; two coincidental training matches against an external benchmark were subsequently replaced for provenance separation | 24, 23 correct, one wrong option | 21, all correct | 0 |
| [Administrative provenance recheck](administrative-provenance-recheck.json) | Final dataset; same previously opened test set; no model settings or test contents changed | 23, 21 correct, one false Clarity and one wrong option | 20, 19 correct | **1**: `SYNTH-V1-D8-TEST-018` |
| [Post-policy recheck](post-policy-reused-test.json) | Same final dataset and models; known test reused after a customer-ownership veto repair | 23, 21 correct, one false Clarity and one wrong option | 19, all correct | 0 on this reused test |

The administrative pre-fix error remains recorded: another person's bitcoin holdings were proposed as the customer's experience even though their own holdings were unspecified. The later deterministic veto blocks that proposal. It is a business-policy repair prompted by a known failure; its passing recheck is not independent validation. No prior result file was overwritten. The underlying model still makes that false-Clarity proposal.

The final engine staged 19 candidates from 204 replies (9.31%) and abstained on 185 (90.69%). All 19 staged candidates matched the authored expectations, and none was confirmed or recorded by the evaluator. No legacy fallback proposal occurred. On these single-turn probes, D5, D9, D10, D12 and D15 produced no accepted financial candidates. This is conservative prototype behavior with substantial free-text usability limits, not evidence that all-question natural language is ready for customers.

Raw intent prediction sets covered 197/204 labels (96.57%): 90 singleton sets, 86 correct singleton predictions, mean set size 1.657. Four truly nonclear replies received singleton Clarity (4/60, 6.67%); the option and policy gates removed their financial proposals in the final engine. Intent singleton accuracy was 86/90 (95.56%).

Raw option prediction sets covered 135/144 labels on genuinely clear financial replies (93.75%): 53 singleton sets, 49 correct, mean set size 2.340. On the 79 genuinely clear replies selected by a singleton Clarity intent, option coverage was 76/79 (96.20%), with 22 singleton sets and 21 correct. This selected subset has selection bias and no separate formal coverage guarantee. The raw wrong D9 option (`SYNTH-V1-D9-TEST-005`, six transactions proposed as zero) was blocked by the financial boundary check.

| Question | Intent calibration threshold | Option calibration threshold | Raw intent test coverage | Final staged candidates |
| --- | ---: | ---: | ---: | ---: |
| D1 | 0.7488 | 0.8818 | 100% | 1 |
| D2 | 0.7713 | 0.7986 | 92.86% | 1 |
| D3 | 0.7625 | 0.8552 | 100% | 1 |
| D4 | 0.7221 | 0.8296 | 100% | 2 |
| D5 | 0.7564 | 0.8721 | 95% | 0 |
| D6 | 0.7018 | 0.6807 | 90% | 3 |
| D7 | 0.7523 | 0.7390 | 91.67% | 1 |
| D8 | 0.7307 | 0.8678 | 95% | 1 |
| D9 | 0.7394 | 0.7589 | 100% | 0 |
| D10 | 0.7193 | 0.8091 | 100% | 0 |
| D11 | 0.7141 | 0.6840 | 90% | 3 |
| D12 | 0.7091 | 0.6829 | 100% | 0 |
| D13 | 0.7091 | 0.7220 | 90% | 2 |
| D14 | 0.7301 | 0.6681 | 100% | 4 |
| D15 | 0.7339 | 0.6760 | 100% | 0 |

Thresholds are nonconformity-score cutoffs, not probabilities that a proposed answer is correct. Split-conformal coverage requires exchangeable calibration and future inputs; these authored samples do not establish that assumption. Marginal coverage does not guarantee singleton precision or zero errors. Lexical filtering, semantic gating and business veto change acceptance, so post-veto accuracy is an empirical observation with no formal conformal coverage guarantee. Each JSON report retains source hashes, counts, detailed prediction sets and failure IDs.

The separate [known one-word requirements probe](known-one-word-requirements-probe.json) is not held-out validation: typed `none` at D8 produced a pending `D8_NONE` candidate through both ML gates and awaited confirmation. `shares`, `bonds` and `cash` abstained in that probe. The explicit UI choices remain a way for a user to state an allowed meaning themselves.

Final dataset SHA-256 values:

```text
train       96543fe7cc72d8a27403ead432fcf81971b0b93b28a5ed77f9e3770fbda83abf
calibration 3b63ebddf80f8b5758bcbef9cb8d662741e68de38660455dcc4649106d5792be
test        18d0dd2ba9fdbe7fa37b9b5cb27be2bc09be435f472341a12b1050c2dfe029fe
```

Fresh independent evaluation and peer comprehension testing remain required before a broader accuracy or usability claim. Single-turn metrics do not replace the application's separate tests for confirmation, corrections, care, consistency and save consent.
