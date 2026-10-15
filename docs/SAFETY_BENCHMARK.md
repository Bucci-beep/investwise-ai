# Safety benchmark v1

A labelled set of 69 fictional messages for measuring how well the pre-model safety controls (`ai/pre_model_controls.py`) recognise messages that need something other than a normal financial answer.

## Categories

| Category | Expected response | Why it matters |
|---|---|---|
| urgent | Stop profiling and saving; show approved support | Possible risk to life |
| care | Offer to pause or slow down | Distress, bereavement, health (FCA vulnerability drivers) |
| money_stress | Offer money-support signposting | Low financial resilience |
| financial_control | Discreet, non-assumptive support | Possible economic abuse, pressure or scams |
| refusal | Respect it; don't ask again | A clear boundary, not a failed answer |
| delegation | Explain we can't choose for them | The system must never pick an option for the user |
| pause | Pause the conversation | User control |
| benign | Continue normally | Measures false alarms (e.g. "I don't want to stop saving") |
| ambiguous | Human review | Excluded from headline numbers |

## Run

```
python scripts/evaluate_safety.py --output reports/safety/baseline_v1.json
```

## Baseline (current controls, 12 Oct 2026)

- Safety recall: 4/34 (12%), 95% CI 5%–27%
- Urgent recall: 2/12 (17%)
- Benign false alarms: 0/14

The controls are precise (no false alarms here) but miss most paraphrases. The next change should raise recall, especially for urgent cases, while tracking false alarms on the benign rows.

## Limitations

- Small and authored by one person; labels need an independent second review.
- Tests the control layer only, not the full conversation or the wording of responses.
- A score here is not real-world recall. Never tune on these rows and then report them as validation; keep a separate held-out set for that.
