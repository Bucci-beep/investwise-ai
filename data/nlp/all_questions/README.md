# Fictional D1–D15 ML development data

Every reply is fictional and newly authored. No real customer data is present. The approved business specification supplies labels; the existing business-example bank and legacy D12 CSVs were not used to generate these rows.

The fixed split roles are `train.csv` (fit/development), `calibration.csv` (conformal calibration only), and `test.csv` (reserved one-time evaluation). Do not tune on calibration or test outcomes. Blank CSV `option_id` means null.

Regenerate and validate with `python3 scripts/build_all_question_dataset.py`. The script embeds the authored expressions. `dataset_audit.json` records coverage and split checks; `provenance.json` records file hashes and limitations.

This is a small synthetic prototype dataset, not evidence of reliable performance with actual investors. A singleton conformal set is a candidate for user confirmation, not proof of intended meaning. Broader or empty sets must leave meaning unresolved.
