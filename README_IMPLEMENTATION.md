# InvestWise AI v7 conversation shell

This package implements only the agreed combined scope.

## Retained architecture

The existing deterministic suitability/risk engine remains untouched.

The existing D12 NLP work remains untouched:

`TF-IDF + Logistic Regression -> conformal prediction -> dialogue policy -> deterministic horizon mapper -> confirmation`

The conversation shell wraps that D12 path. D13-D15 stay deterministic/fixed-choice until their free-text ML paths are separately evaluated.

## Files to copy

Copy these files into the existing repository:

- `ai/conversation_state.py`
- `ai/question_spec.py`
- `ai/pre_model_controls.py`
- `ai/d12_adapter.py`
- `ai/deterministic_conversation.py`
- `ai/conversation_engine.py`
- `tests/test_conversation_shell.py`

Do not replace the existing:
- `ai/text_classifier.py`
- `ai/conformal.py`
- `ai/dialogue_policy.py`
- `ai/interpretation.py`
- `ai/horizon_mapper.py`

## Business specification

Place the v7 `business-decision-spec.json` in the repository, for example:

`config/business-decision-spec.json`

The shell loads the question order, fixed options, labels, and completion roles from that file rather than duplicating 72 categories in Python.

## Wiring D12

Use the existing evaluated D12 interpreter and wrap it:

```python
from ai.d12_adapter import ExistingBoundedInterpreterAdapter
from ai.conversation_engine import ConversationEngine
from ai.question_spec import BusinessSpec

spec = BusinessSpec.load("config/business-decision-spec.json")

# existing_d12_interpreter is the BoundedQuestionInterpreter already created
d12 = ExistingBoundedInterpreterAdapter(existing_d12_interpreter)

engine = ConversationEngine(
    business_spec=spec,
    d12_resolver=d12,
)
```

The shell never commits a D12 option directly from the model. A D12 output becomes only a candidate and is recorded only after `confirm_current(True)`.

## Conversation sequence

1. Read current question from `engine.current_question`
2. Call `engine.submit_answer(text)`
3. If action is `confirm`, read back the candidate and call `engine.confirm_current(True/False)`
4. If action is `clarify`, ask one focused clarification
5. At three clarification turns, offer pause or finish incomplete
6. After all 15 are handled, call `engine.final_playback()`
7. Call `engine.confirm_final_accuracy(True)`
8. Separately call `engine.give_save_consent(True)`
9. Call `engine.save()`

## Important behaviour

- safety/control checks run before D12 ML
- no default to a middle answer
- D12 exact 3 years maps to Short
- D12 exact 10 years maps to Medium
- “all my savings ... next year” maps to a Short candidate plus a dependency hold
- D1 emergency purpose plus D12 Long creates a dependency hold rather than forcing Short
- D14 Greater plus D15 Unacceptable is preserved as two answers and requires acknowledgement
- correction invalidates final accuracy/save consent and affected D12/D13 confirmations where scope/dependency may have changed
- transient audit events are not written into the saved profile
- no investment recommendation is produced

## Run tests

```bash
pytest -q tests/test_conversation_shell.py
```
