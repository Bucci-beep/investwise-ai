# D5 runtime: safety and faithful confirmation

13 October 2026. This repair changes two modules in the manually launched D5 runtime. The model, prompts, schema, semantic validators, all fifteen questions and the normal application default are unchanged. There is no cascade or extra interpretation stage. The [technical handoff](d5-semantic-method-handoff.md#latest-runtime-update-safety-and-exact-confirmation) contains branch update and launch instructions.

## Changes

- **Safety:** reuse the main application's already-repaired first-person jump-intent prefix, including `want to`, `planning to` and `intend to`. Retain the runtime's existing height restrictions, quotation/negation guards and other safety handling. Do not overwrite the runtime with the main application's different controls file.
- **Confirmation:** quote the user's exact, contract-validated evidence for current funding and excluded sources. Deduplicate identical quotations in their original fact order. Keep an excluded source even when its category is also active. The existing receipt check, candidate, no-regular-source handling and explicit confirmation/correction remain in place.

The screenshot's saved model output already distinguished `just savings` (used) from `not investment withdrawals` (excluded), both within `D5_WITHDRAWALS`. The old renderer collapsed both into the category label and omitted the exclusion. This is a correction to the displayed interpretation, not retraining or a new semantic accuracy result.

The corrected confirmation for that same saved output is:

```text
I understand your current living-cost funding as:
"just savings"
Not counted as current funding, from your words:
"not investment withdrawals"
Have I understood correctly?
```

## Bounded offline checks

Only existing affected examples were used. No authentication, inference, new examples, training or calibration was performed. Networking was disabled in the replay process, and stored model outputs were passed through the real packaged interpreter and engine. This replay is a software check, not live free-text evidence or a fallback available in the demo.

| Existing evidence | Result |
|---|---|
| Ten existing height-danger/negative examples from the author's working-copy `tests/test_bounded_semantics.py`; their exact inputs are included in the JSON record | 6/10 matched before; 10/10 after. All eight urgent examples now stop before interpretation; bus usage and explicit negation remain unaffected. |
| Seven published clear examples: D5-STD-01–06 and 08 | Candidates and routing preserved, with confirmation and correction preserved in 7/7; no extra interpretation for those actions. |
| Five replies from the user's existing screenshot session | Both holding-only replies still ask about actual use. All three savings answers retain their original candidate; the broad category wording is removed from confirmation. |
| Nine affected clear replies across those two groups | Exact active/excluded fact quotations retained in 9/9; the remaining clear no-regular-source example retains its authored meaning. |

All **22 existing inputs** passed the final scoped software checks. The twelve saved-response replays preserved their action, bucket and candidate; no profile was saved. The [machine-readable record](d5-runtime-repair-2026-10-13.json) separates safety checks, saved-output replay and preservation. These counts are not a new model-performance score. The original eleven-example evidence and the earlier live conversation have not been rewritten.

## Preservation and use

The runtime manifest retains original source hashes and records the two repair hashes separately from the earlier credential-path portability change. The package remains 41 files: 38 identical to its original source, one portability modification and two runtime repairs. Existing uncommitted main-app and test-file changes were preserved byte for byte and are excluded from this publication. The JSON preservation snapshot describes the completed repair check, before these publication-only documentation updates.

Restart the manually launched D5 terminal to load the repaired runtime:

```bash
"$HOME/.venvs/investwise-d5/bin/python" demos/d5/demo.py chat
```

Run this from the repository root after pulling `feature/all-question-ml-prototype`. Existing installations need no dependency change; first-time users follow the [installation and individual sign-in instructions](../demos/d5/README.md). This command makes live interpretation requests when answers are submitted and still requires service access. The repair did not check provider availability or run the live demo. The normal application route is unchanged. Publication includes only these two runtime repairs, their manifest and documentation; no additional tests or model calls were run for publication.
