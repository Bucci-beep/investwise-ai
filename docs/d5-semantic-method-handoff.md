# D5 semantic interpretation technical handoff

12 October 2026 · `feature/all-question-ml-prototype`

**The D5 repair improved the eleven existing examples without changing the model.** Seven clear replies now yield seven correct candidates instead of five. A reply that only states income received now prompts a question about actual use instead of an unsupported MULTIPLE answer. This handoff contains the implemented method, exact source references and the saved before/after evidence.

The application default, all 15 questions, existing trained models and datasets remain unchanged. The repaired interpreter is an isolated experiment using **GPT-6.1 Sol, low reasoning**. This branch update publishes its implementation references and evidence; it does not enable a new app route.

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
| Visible explanation | Confirmation repeated a category; Help did not distinguish pension holdings from payments | Confirmation names active categories and exclusions. The existing Help route uses an authored explanation of pension payments used for costs versus a pension holding. |

The representation is a **category/relation group**, not a count of real-world source instances, despite retaining the existing `source_instance_id` field name for wire compatibility. Different ownership, time or use relations stay separate. Duplicate groups remain invalid even if their quotations differ. No post-generation merger repairs a rejected output.

The JSON structure and resolver interface remain unchanged. The decision contract and candidate provenance identify v2; the reused transport retains its v1 wire-schema name. The earlier proposal to add `source_anchor` was **not implemented**: grouped facts solved the two observed same-category rejections without adding a field.

The counterexample still works: “Both my wages and pension pay for everyday costs” remains MULTIPLE. This repair therefore distinguishes actual mixed funding from merely receiving two kinds of income in the observed examples.

## One interpretation and authored dialogue

Controls run before interpretation. For eligible D5 text, one call receives the current question, approved options, business instructions and original reply. It returns the bucket, proposed option, supported facts, exact quotations, completeness and missing-information topic. The program validates and reuses that immutable decision; confirmation does not request another interpretation.

| Outcome or topic | Behaviour |
|---|---|
| Clarity with complete supported funding | Propose the fixed option, explain the active/excluded categories and request confirmation. |
| `current_funding_unknown` | Ask which sources, if any, actually pay costs now; permit another source or an unresolved answer. This includes receipt-only and holding-only replies. |
| Undecided with `source_use_unknown:<category>` | Ask neutrally whether the named category pays costs and explicitly allow remaining unsure. |
| `complete_answer_needed` | Request the complete current answer, including the correction, instead of reconstructing omitted facts. |
| Confusion with `question_meaning` | Give the fixed plain-English D5 explanation. Pure Help may be handled before the model. |
| `reply_unclear` | Acknowledge that the wording was not understood and invite rephrasing. |

For example, case 10 now receives: “It is fine to be unsure. Does money from salary or wages currently pay any of your living costs, or is that still unclear?” The following sentence asks for any other current sources so a subsequent reply can be interpreted as a complete answer. That subsequent answer was not tested in this round.

Every nonclear output contains no financial option or facts. Missing information alone is not customer indecision. The existing abstention mechanism also carries an understood information gap: cases 07 and 09 have `semantic_bucket: null`, `model_abstain: true` and a specific topic. Its existing `model_uncertain` diagnostic stays true; the authored wording distinguishes a missing fact from unrecognised language. This adds no fourth customer bucket.

The model retains `tools: []`, `tool_choice: "none"`, `store: false` and low reasoning. It has no execution, confirmation or save capability. The unchanged transport accepts completed passive output, with no automatic retry, substitute model or response repair. The engine retains controls, clarification limits, confirmation, correction and separate save consent; this handoff does not claim a new full-journey verification.

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

These files are review references under `docs/`; they are not imported by the application. The target branch is older than the tested experiment. Static dependency inspection found 37 local AI modules: 26 absent, six different and five identical. The different shared modules include the conversation engine, pre-model controls, question specification, bounded answers, horizon mapper and all-question resolver. **Copying only these D5 files into `ai/` would not reproduce the tested environment.** The manifest identifies exact hashes and compatibility status so an engineer can plan a separate integration without overwriting shared controls.

Credentials, authentication caches, model artefacts and machine-specific launchers are not included. The existing OAuth setup is external and user-specific; authentication code also depends on a compatible POSIX runtime. No new login, inference or test run is needed to inspect this handoff. There is no app enable command in this change.

## Evidence and reliability

Use the saved [before inputs](d5-semantic-repair/evidence/before/inputs.json), [after expectations](d5-semantic-repair/evidence/after/expectations.json), [comparison](d5-semantic-repair/evidence/after/comparison.json), [execution](d5-semantic-repair/evidence/after/execution.json) and [independent AI review](d5-semantic-repair/evidence/after/result-review.json). Both `records/` folders contain the exact original eleven records, including raw output, contract outcome, authored wording and safe backend-call metadata. Inputs and expectations are byte-identical across runs; labels were excluded from requests. The preserved expectations metadata describes the original baseline prompt; the repair used the separately versioned v2 prompt, with no label changes.

This was one targeted rerun of already-opened examples. It supports these repairs, not an unseen-text accuracy estimate, a guarantee of correctness, an optimal-model claim or whole-product readiness. The existing shared-quotation guard can still reject different relations in one category when they use the identical full evidence quote; the eleven cases do not establish general handling of that situation. This publication adds no examples or testing.

The business takeaway for the company meeting is: **use the existing language model to interpret source-to-use relationships, express them in the same categories as the questionnaire, and ask about the one material missing fact before confirming anything.**
