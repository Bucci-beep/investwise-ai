# Independent D5 demo: packaging and the original eleven-case check

12 October 2026 · `feature/all-question-ml-prototype`

**Teammates can now install and manually start the repaired D5 terminal conversation.** Follow the [quick start](../demos/d5/README.md). This packages the existing interpreter and compatible dependencies; it does not enable a route in the normal app or change the semantic method.

**Later, separately authorised conversation:** the unchanged entry completed the sequence 09 → 01 → confirmation → correction → 02 → confirmation, using three live calls and no new examples. See the [successful conversation report and transcript](d5-live-conversation-2026-10-12/REPORT.md). The interrupted eleven-case run below remains a distinct historical result; it was not resumed or converted into a pass.

## What was delivered

- A separate `login` / `chat` entry, accepting fictional free text through one direct GPT-6.1 Sol low interpretation. Confirmation, correction, Help, pause/resume and stop use the existing engine. D5 ends before D6, and financial profile saving is disabled in this example.
- A private runtime containing the existing dependency closure and full fifteen-question specification. Forty of its 41 files are byte-identical copies. The credential provider has only two portability changes: resolve the current user's home directory and protect the whole checkout from credential storage. The [runtime manifest](../demos/d5/runtime-manifest.json) records both changes and file hashes.
- Eleven pinned Python dependencies and explicit launch instructions. The fresh installation and terminal start succeeded on **macOS, Python 3.12.14** without the source experiment on `PYTHONPATH`. Linux was not executed here; native Windows is unsupported by the existing POSIX credential lock.
- Credentials and diagnostic records remain outside Git. Each teammate needs their own eligible sign-in and exact-model access. No new sign-in was performed in this check; existing authorised credentials were reused.

The model, prompts, schema, validators and authored financial follow-ups were not changed during packaging. Default application source, all 15 questions and prior evidence remain unchanged. There is no retraining, additional LLM, new semantic rule, automatic retry, response repair or cached fallback.

## One live pass, no additional examples

The live check in this packaging round used the **original eleven inputs and unchanged expectations** through the same `D5Session.submit` method as the terminal conversation. The entry and its files were frozen before the run. Expectations were not sent to the interpreter. The run stopped at the fourth input; the complete-set scorer did not run.

| Original case | Result in this standalone run |
|---|---|
| 01 · Wages saved; pension pays bills | Correct raw meaning and accepted PENSION candidate; confirmation excludes wages. |
| 02 · Wages and pension both pay costs | Correct raw meaning and accepted MULTIPLE candidate; confirmation includes both. |
| 03 · Pay from two employers covers costs | Correct raw meaning and accepted SALARY candidate; one category/relation group. |
| 04 · Benefits and family support pay bills | No semantic output: provider reported `server_is_overloaded` / `service_unavailable_error`. |
| 05–11 | Not attempted after the failure. |

**Three completed cases, three correct raw meanings and three correct accepted candidates; four actual model requests.** The fourth response carried a provider error despite HTTP 200. It was not a validator rejection or an incorrect semantic answer. No retry, fallback or later input followed. There were no choice-recovery, confirmation or save actions in this check. The Help example was not reached.

The [execution receipt](d5-standalone-check-2026-10-12/execution.json) and [failure receipt](d5-standalone-check-2026-10-12/D5-STD-04-failure.json) preserve the reason. [Saved records and review](d5-standalone-check-2026-10-12/review.json) distinguish these three observations from an eleven-case pass. Provider service availability remains an external limitation for a live meeting.

## Preservation and limits

The [packaging record](d5-standalone-check-2026-10-12/packaging.json) records installed versions, runtime hashes and preservation checks. The [pre-run snapshot](d5-standalone-check-2026-10-12/started.json) and execution receipt show no frozen file changed during inference. Independent static review checked the terminal adapter; it did not run additional semantic cases. It corrected terminal presentation/resume handling before the frozen run, without changing the copied interpreter.

The previous complete eleven-case repair evidence remains unchanged: clear correct candidates increased from five to seven, two same-category contract rejections disappeared, and receipt-only text no longer produced an unsupported MULTIPLE candidate. See the [technical handoff and exact before/after table](d5-semantic-method-handoff.md). Those historical results are **not** a complete pass of this new portable entry.

This packaging round established a runnable independent entry and three live completed examples in its clean environment. The full eleven-case portable check remains incomplete. Interactive confirmation/correction was not exercised in that round; it subsequently succeeded in the separately recorded D5 conversation linked above. A full fifteen-question UI journey and teammate-specific first sign-in remain unverified here. No extra cases, test suite, app integration or default switch was added.
