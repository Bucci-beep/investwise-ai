# Run the independent D5 demonstration

This is a manually started **terminal conversation** using the repaired D5 interpreter: GPT-6.1 Sol with low reasoning. You can type new fictional replies, read the interpretation or focused follow-up, confirm a candidate and replace an earlier answer. The normal Streamlit app, its default model route and its 15 questions are unchanged.

## Install once

Use **Python 3.12 on macOS or Linux**, a terminal and a local browser for sign-in. The existing credential implementation uses POSIX file locking; native Windows is not supported. Installation requires network access. The package pins match the source experiment; no model weights or additional LLM are downloaded.

From the repository root after pulling `feature/all-question-ml-prototype`:

```bash
python3.12 -m venv "$HOME/.venvs/investwise-d5"
"$HOME/.venvs/investwise-d5/bin/python" -m pip install -r demos/d5/requirements.txt
```

If `python3.12` is not found, install or select a Python 3.12 interpreter first. Use that interpreter only to create this separate environment; the commands afterwards use its explicit path. Do not change the normal app's environment.

## Sign in with your own account

```bash
"$HOME/.venvs/investwise-d5/bin/python" demos/d5/demo.py login
```

Complete **Continue with ChatGPT** in the system browser. The existing flow registers this app, checks identity and granted permission, and keeps credentials in an owner-only directory at `~/.config/investwise/d5-chatgpt-plan/`. The login command also checks that the account lists the exact `gpt-6.1-sol` model. Your account must be eligible and have access; somebody else's successful run does not grant access to your account. This uses the existing [official open-source sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in).

There is no shared credential in the repository, API-key setup or billing fallback. If this app already has a valid registration on your computer, you can start the conversation without signing in again. An expired token can be renewed by the existing credential provider; failed inference is never replayed automatically.

## Start a conversation

```bash
"$HOME/.venvs/investwise-d5/bin/python" demos/d5/demo.py chat
```

Enter a complete D5 answer on one line. The opening question is:

> What currently pays for your everyday living costs?

For example, use an existing demonstration reply: `My wages stay in savings. My pension pays the bills.` Read the proposed interpretation, then choose an explicit action:

| Command | Effect |
|---|---|
| `/help` | Explain D5 through the existing Help route. |
| `/confirm` | Confirm the pending D5 candidate in this session. No new interpretation call. |
| `/change` | Invalidate the previous D5 answer and request the complete replacement. |
| `/pause` / `/resume` | Pause and resume the current example through the existing controls. |
| `/stop` | Stop this example. Resume does not undo a stop. |
| `/new` | Explicitly start a separate fictional example with empty answer state. |
| `/exit` | Close the terminal demonstration. |

Ordinary replies such as the original pension terminology question go through the same control checks as the interpreter's conversation engine. Follow-ups ask for a complete current funding answer: do not rely on the model remembering an earlier source. Only the current reply, question/options and fixed instructions reach the model.

Confirmation ends this D5 example rather than opening D6. You can then change D5, start a separate example or exit. This isolated example cannot save a completed financial profile or bypass the main application's separate save-consent workflow. A pause, stop or safety response creates no financial answer. Any human handover shown is simulated; nobody is contacted.

The screen shows the number of interpretation calls for each action. Financial interpretation uses one direct model call; Help and deterministic controls do not constitute model successes. No choice menu is used as free-text recovery.

## Local records and access failures

Use fictional data. Each launch creates a private directory under `~/.local/state/investwise/d5-demo/` containing replies, raw model output, authored responses and safe backend receipts. These are local diagnostic records, not a saved financial profile. The terminal prints the directory location. Do not commit these directories or authentication files.

Optional `--profile-dir` and `--output-dir` arguments select external directories; neither can point inside the repository. The output directory must be new. A runtime or access failure stops the session: there is no automatic sign-in, inference retry, model replacement, output repair or cached-answer fallback. Read the safe receipt's stage/code, then resolve access deliberately before a later run.

The transport still uses empty tools, `tool_choice: none`, `store: false` and completed streamed responses. Its 30-second deadline and 1 MiB response limit are local limits, not a server token cap. See the [official preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).

## The original eleven examples

An explicit comparison command uses **only the same eleven published inputs**, through the same session submission method as the interactive entry:

```bash
"$HOME/.venvs/investwise-d5/bin/python" demos/d5/demo.py check-11
```

This command makes live model requests. Do not run it merely to view the existing results. It refuses changed input/expectation files, freezes the packaged code before execution, reads labels only after inference, and stops on a runtime/access failure without retry. The original Help example should make zero model calls. Automated option/routing matches still require a review of raw meanings and follow-up wording; they are not an independent semantic accuracy estimate.

The previous repair produced seven correct clear candidates, two missing-information responses, one Undecided response and one pre-model Help response. These are already-opened fictional examples, not an unseen evaluation bank. The [standalone verification record](../../docs/d5-standalone-verification.md) records the authorised packaging check separately.

## How the default stays unchanged

`runtime/` contains the existing tested dependency closure and full fifteen-question specification. A separate Python process loads that private `ai` package first and rejects imports mixed with the normal application's `ai` package. The normal factory and frontend never import this directory. Other-question resolution and profile saving are disabled in this example.

No model, prompt, schema, semantic validator, business definition or follow-up was changed while packaging. Only the credential provider's personal default path and repository containment path were made portable. Numerical packages are import dependencies of existing shared types; the demo does not train or call those classifiers. The [runtime manifest](runtime-manifest.json) records original and packaged hashes, portability differences and pinned versions.

For the semantic method and all eleven before/after cases, read the [technical handoff](../../docs/d5-semantic-method-handoff.md).
