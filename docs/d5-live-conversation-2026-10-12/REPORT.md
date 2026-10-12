# D5 live conversation completed

12 October 2026 (Europe/London). One persistent session through the unchanged `demos/d5/demo.py chat` entry, using only three of the existing eleven examples.

| Step | Existing input or control | Observed result |
|---|---|---|
| Follow-up | D5-STD-09: “I receive wages and pension payments” | Asked which sources actually pay housing, food and bills. No financial candidate and no false Undecided label. |
| Answer | D5-STD-01: “My wages stay in savings. My pension pays the bills.” | Correct PENSION candidate, with wages explicitly excluded. |
| Confirmation | `/confirm` | Pension answer confirmed in this session; zero interpretation calls. |
| Correction | `/change` | Previous D5 answer invalidated; zero interpretation calls. |
| Replacement | D5-STD-02: “Both my wages and pension pay for everyday costs.” | Correct new MULTIPLE candidate, including both sources; required fresh confirmation. |
| Confirmation | `/confirm` | Corrected answer confirmed in this session; zero interpretation calls. |
| Exit | `/exit` | Same process ended successfully. No financial profile saved. |

**Three live model requests, all completed. Two correct financial candidates and one appropriate missing-information follow-up.** There were no service failures, retries, cached answers, response repairs, choice selections or additional examples. Confirmation is reported separately from free-text interpretation.

The exact model remained GPT-6.1 Sol with low reasoning and interpretation-only permissions (`tools: []`, `tool_choice: none`, `store: false`). All 182 tracked files matched their pre-run hashes; the model instructions, all 15 questions and the default application were unchanged.

An independent AI review of the saved records found no concrete issue. The transcript records both confirmations and correction navigation, which the entry does not persist as separate state snapshots. The replacement record has an unconfirmed new candidate with cleared selection/confirmation fields; final confirmation is evidenced by the actual terminal output and the unchanged wrapper's confirmation condition.

This completes the requested D5 conversation. It uses already-opened wording and full replacement answers; it does not measure accuracy on unseen text or establish readiness of the normal 15-question application.

- [Actual terminal transcript](TRANSCRIPT.txt)
- [Structured terminal capture](terminal-transcript.json)
- [Results and record hashes](results.json)
- [Pre-run sequence and preservation snapshot](plan.json)

This publication uses the existing saved evidence; no additional inference, cases or tests were run. The three model records, results and pre-run snapshot retain their exact original bytes. Only the personal output-directory path is replaced in the two published transcripts. Credentials and raw transport traces are not included. See [publication provenance](publication.json).

For a live company demonstration or your own fictional wording, follow the [manual launch and conversation instructions](../../demos/d5/README.md#use-it-for-the-company-demonstration). Reading the saved transcript does not make a model call or replay an answer in the app.
