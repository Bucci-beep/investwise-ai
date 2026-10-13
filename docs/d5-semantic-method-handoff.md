# D5 semantic interpretation technical handoff

Updated 13 October 2026 · `feature/all-question-ml-prototype`

## Latest runtime update: safety and exact confirmation

The independent D5 terminal now includes two small software repairs:

- The packaged safety controls recognise the previously missed `want to`, `planning to` and `intend to` jump-intent forms, while retaining the existing negation and quotation guards.
- Confirmation quotes the user's exact validated active and excluded facts. For `No, just savings not investment withdrawals`, it displays `just savings` as current funding and `not investment withdrawals` as excluded. Both remain in the unchanged `D5_WITHDRAWALS` category; an exclusion is no longer hidden merely because it shares an active category.

The saved model response already contained that distinction. This fixes its presentation rather than training or changing the model. The same verified interpretation is reused; no extra model call, financial wording classifier or cascade is added.

The [runtime repair report](d5-runtime-repair-2026-10-13.md) and [recorded checks](d5-runtime-repair-2026-10-13.json) cover **22 existing affected inputs only**: ten safety examples, seven published clear replies and five replies from the existing screenshot session. All scoped checks passed offline; candidates, follow-ups, confirmation and correction were preserved. This publication runs no further tests or inference and does not establish current provider availability.

**Teammate update, from the repository root:**

```bash
git switch feature/all-question-ml-prototype
git pull --ff-only origin feature/all-question-ml-prototype
"$HOME/.venvs/investwise-d5/bin/python" demos/d5/demo.py chat
```

Close any running D5 terminal before launching again so it loads the repaired runtime. First-time users should follow the [Python 3.12 dependency installation and their own ChatGPT sign-in](../demos/d5/README.md#install-once) before `chat`. Existing installations need no dependency change. Each teammate needs their own eligible account and available usage; credentials remain outside Git. `check-11` is not required to launch.

Only the independent D5 runtime, its manifest and documentation are included in this update. Separate pending main-app safety/follow-up changes and their tests are not included. The model, prompts, schema, validators, all fifteen questions and the normal Streamlit default remain unchanged. The [runtime manifest](../demos/d5/runtime-manifest.json) records the two repair hashes; historical source-reference files and earlier evidence remain unchanged.

## Earlier semantic repair and live demonstration

**The D5 repair improved the eleven existing examples without changing the model.** Seven clear replies now yield seven correct candidates instead of five. A reply that only states income received now prompts a question about actual use instead of an unsupported MULTIPLE answer. This handoff contains the implemented method, exact source references and the saved before/after evidence.

The application default, all 15 questions, existing trained models and datasets remain unchanged. The repaired interpreter uses **GPT-6.1 Sol, low reasoning**. The branch now includes an [independent, manually launched terminal demonstration](../demos/d5/README.md) with its compatible runtime dependencies. It accepts new fictional replies through one direct interpreter; it does not enable a new route in the normal Streamlit app.

**A complete live D5 conversation has now succeeded through that terminal entry.** Existing case 09 prompted a focused actual-use question; case 01 produced a pension-only candidate which was confirmed; `/change` invalidated it; case 02 produced a fresh mixed-source candidate which was confirmed. Three model calls completed, with no service failures or retries. Confirmation and correction navigation made zero extra calls. The [report and actual transcript](d5-live-conversation-2026-10-12/REPORT.md) and [company demonstration instructions](../demos/d5/README.md#use-it-for-the-company-demonstration) are ready for teammates to use.

The earlier fresh-install eleven-case pass completed the first three cases, then stopped on provider overload at case 04. That interrupted pass remains unchanged in the [packaging record](d5-standalone-verification.md). The later conversation used three existing examples; it does not complete that eleven-case pass or establish unseen-text accuracy. Publishing this update adds no model calls, examples or tests.

## The business meaning the interpreter must preserve

D5 asks **“What currently pays for your everyday living costs?”** It identifies the customer's current funding categories, not their wealth, income amount, financial sufficiency or investment risk.

Read the whole reply as a relationship: **whose money, when, and does it actually pay current living costs?** Receiving wages or owning a pension fund does not by itself establish cost use. Conversely, a direct answer such as “My wages” can rely on the question's context; customers need not use scripted wording. No new short-answer case was added in this repair round.

Count categories, not payers or accounts. Two employers remain one salary category. Benefits and regular family support share one category. Classify the support received by the customer, not the donor's underlying earnings. Keep explicitly saved, future or otherwise excluded sources out of the active answer. Only an explicit statement supports “No regular source”.

## What changed and why it helps

| Change | Previous failure | Implemented repair and effect |
|---|---|---|
| Receipt versus use | “I receive wages and pension payments” became two active funding sources | The prompt explicitly separates receipt, ownership and availability from actual expenditure. Missing use information leaves the option unresolved. Case 09 changed at the raw interpretation stage. |
| Same-category representation | Two employers, or benefits plus family help, were interpreted correctly but rejected as duplicate facts | Each fact now represents a group with identical category, owner, time and cost-use relations. Several payers can share that group and its original quotation. Cases 03 and 04 now pass the contract without inventing payer names or relaxing duplicate checks. |
| Relevant follow-up | The interpreter identified uncertainty about wages but the engine asked a generic question | The same model response selects one controlled missing-information topic. The engine renders an authored question about that topic, without another model call or financial wording classifier. |
| Visible explanation | Confirmation repeated a category; Help did not distinguish pension holdings from payments | The earlier repair named active categories and exclusions; the 13 October runtime repair now quotes exact active/excluded evidence to retain finer distinctions. The existing Help route uses an authored explanation of pension payments used for costs versus a pension holding. |

The representation is a **category/relation group**, not a count of real-world source instances, despite retaining the existing `source_instance_id` field name for wire compatibility. Different ownership, time or use relations stay separate. Duplicate groups remain invalid even if their quotations differ. No post-generation merger repairs a rejected output.

The JSON structure and resolver interface remain unchanged. The decision contract and candidate provenance identify v2; the reused transport retains its v1 wire-schema name. The earlier proposal to add `source_anchor` was **not implemented**: grouped facts solved the two observed same-category rejections without adding a field.

The counterexample still works: “Both my wages and pension pay for everyday costs” remains MULTIPLE. This repair therefore distinguishes actual mixed funding from merely receiving two kinds of income in the observed examples.

## One interpretation and authored dialogue

Controls run before interpretation. For eligible D5 text, one call receives the current question, approved options, business instructions and original reply. It returns the bucket, proposed option, supported facts, exact quotations, completeness and missing-information topic. The program validates and reuses that immutable decision; confirmation does not request another interpretation.

| Outcome or topic | Behaviour |
|---|---|
| Clarity with complete supported funding | Propose the fixed option, quote the exact active/excluded evidence and request confirmation. |
| `current_funding_unknown` | Ask which sources, if any, actually pay costs now; permit another source or an unresolved answer. This includes receipt-only and holding-only replies. |
| Undecided with `source_use_unknown:<category>` | Ask neutrally whether the named category pays costs and explicitly allow remaining unsure. |
| `complete_answer_needed` | Request the complete current answer, including the correction, instead of reconstructing omitted facts. |
| Confusion with `question_meaning` | Give the fixed plain-English D5 explanation. Pure Help may be handled before the model. |
| `reply_unclear` | Acknowledge that the wording was not understood and invite rephrasing. |

For example, case 10 now receives: “It is fine to be unsure. Does money from salary or wages currently pay any of your living costs, or is that still unclear?” The following sentence asks for any other current sources so a subsequent reply can be interpreted as a complete answer. That subsequent answer was not tested in this round.

Every nonclear output contains no financial option or facts. Missing information alone is not customer indecision. The existing abstention mechanism also carries an understood information gap: cases 07 and 09 have `semantic_bucket: null`, `model_abstain: true` and a specific topic. Its existing `model_uncertain` diagnostic stays true; the authored wording distinguishes a missing fact from unrecognised language. This adds no fourth customer bucket.

The model retains `tools: []`, `tool_choice: "none"`, `store: false` and low reasoning. It has no execution, confirmation or save capability. The unchanged transport accepts completed passive output, with no automatic retry, substitute model or response repair. The engine retains controls, clarification limits, confirmation, correction and separate save consent. The successful standalone D5 conversation does not establish a full fifteen-question journey or exercise profile saving.

## The same eleven examples before and after

The comparison baseline is the isolated v1 GPT interpreter on these exact inputs—not the branch's default TF-IDF model or the different `feature/ml-improvement` evaluation bank.

| Case and exact reply | Before | After |
|---|---|---|
| 01 · “My wages stay in savings. My pension pays the bills.” | Correct pension candidate | Preserved; confirmation names excluded wages |
| 02 · “Both my wages and pension pay for everyday costs.” | Correct MULTIPLE candidate | Preserved; confirmation names both categories |
| 03 · “Pay from my two employers covers my living costs.” | Correct salary meaning, duplicate rejection | One salary group; correct candidate |
| 04 · “Benefits and regular help from my family pay my bills.” | Correct support meaning, duplicate rejection | One support group; correct candidate |
| 05 · “My brother sends me money every week for rent and food.” | Correct support candidate | Preserved |
| 06 · “I use my savings for bills. My pension starts next year.” | Correct withdrawals candidate | Preserved; future pension excluded |
| 07 · “I have a pension fund, but I have not started drawing from it.” | No candidate; generic complete-answer request | No candidate; asks actual current funding |
| 08 · “I have no regular source paying my living costs at the moment.” | Correct no-regular-source candidate | Preserved |
| 09 · “I receive wages and pension payments” | Unsupported MULTIPLE accepted | No option; asks actual use, allowing other sources or none |
| 10 · “My pension pays some bills; I cannot tell whether my wages also do.” | Correct Undecided; generic follow-up | Undecided retained; targeted wage-use question |
| 11 · “Does pension mean my fund or payments?” | Help missed the distinction | Help explains holding versus payments; zero model calls |

- Clear raw meanings: **7/7 → 7/7**. Clear correct accepted candidates: **5/7 → 7/7**. All five previously accepted correct answers remain, by case ID.
- Contract rejections: **2 → 0**. Unsupported accepted candidates: **1 → 0** in this set.
- After repair: seven financial candidates, two abstentions with relevant questions, one Undecided response and one Help response.
- Each run contains eleven records and ten actual model calls. No choice recovery, confirmation, save or UI interaction was counted as interpretation success.

These are two kinds of improvement: case 09 avoids an unsupported semantic inference; cases 03 and 04 recover already-correct meanings previously blocked by software. Follow-up relevance is a third, dialogue-level improvement. Abstention is not a completed financial answer, and Help is not a model success.

## Using the implementation references

Start with the [source and evidence manifest](d5-semantic-repair/manifest.json). The seven source-reference files preserve the exact tested bytes:

| Files under `docs/d5-semantic-repair/source-reference/` | Purpose |
|---|---|
| `config/d5-single-interpreter-v1-prompt.txt` and `...-v2-prompt.txt` | Compare the business instructions before and after. |
| `ai/d5_single_interpreter_contract.py` and `..._v2.py` | Compare group validation, controlled topics and contract identity; the public `resolve("D5", text)` interface is retained. |
| `ai/d5_single_interpreter_engine.py` and `..._v2.py` | Inspect the inherited state/receipt boundary and the small authored-wording/provenance adapter. |
| `ai/d5_single_interpreter_transport.py` | Inspect the unchanged bounded, passive, single-call transport. |

The [D5 question and option snapshot](d5-semantic-repair/source-reference/config/d5-question-options.json) contains the exact question and eight option definitions supplied in both saved runs. The manifest also records the tested business-spec file hash, its canonical hash used by the decisions, and the older target spec's file hash. This is a reference extract; the application's `config/business-decision-spec.json` is unchanged.

These files remain review references under `docs/`. At the handoff baseline, static dependency inspection found 37 local AI modules: 26 absent from the target branch, six different and five identical. The different shared modules included the conversation engine, pre-model controls, question specification, bounded answers, horizon mapper and all-question resolver. **Copying only the D5 files into the normal `ai/` directory would not reproduce the tested environment.**

The runnable demonstration addresses that gap with the dependency closure in `demos/d5/runtime/`, loaded only by its separate process. Its [runtime manifest](../demos/d5/runtime-manifest.json) identifies original and current hashes, including the two 13 October repairs. The credential provider has only two portability changes: use the current user's home directory and enforce credential storage outside the full checkout. The model, prompt, schema, validators and authored follow-ups remain unchanged. NumPy/scikit-learn are import dependencies of the existing shared types; no classifier is trained or invoked by the D5 demo.

Follow [the installation and launch instructions](../demos/d5/README.md) for Python 3.12 on macOS/Linux. Teammates use their own eligible account and external credentials. No credentials, caches or model artefacts are committed. Use the documented company sequence, or start `chat` and enter your own fictional wording; `check-11` is not a prerequisite. Follow-ups and corrections currently require complete funding statements. Confirmation ends the D5 example; change/new/exit remain available, and other questions and profile saving are disabled there. The main application's full journey and separate save-consent route remain unchanged.

## Evidence and reliability

Use the saved [before inputs](d5-semantic-repair/evidence/before/inputs.json), [after expectations](d5-semantic-repair/evidence/after/expectations.json), [comparison](d5-semantic-repair/evidence/after/comparison.json), [execution](d5-semantic-repair/evidence/after/execution.json) and [independent AI review](d5-semantic-repair/evidence/after/result-review.json). Both `records/` folders contain the exact original eleven records, including raw output, contract outcome, authored wording and safe backend-call metadata. Inputs and expectations are byte-identical across runs; labels were excluded from requests. The preserved expectations metadata describes the original baseline prompt; the repair used the separately versioned v2 prompt, with no label changes.

The repair result above was one targeted rerun of already-opened examples. It supports these repairs, not an unseen-text accuracy estimate, a guarantee of correctness, an optimal-model claim or whole-product readiness. The existing shared-quotation guard can still reject different relations in one category when they use the identical full evidence quote; the eleven cases do not establish general handling of that situation. The [standalone packaging verification](d5-standalone-verification.md) is recorded separately and uses only these same eleven inputs; no evaluation bank is expanded.

The business takeaway for the company meeting is: **use the existing language model to interpret source-to-use relationships, express them in the same categories as the questionnaire, and ask about the one material missing fact before confirming anything.**
