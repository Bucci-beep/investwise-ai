"""Read locked fictional test data only after model settings are frozen.

Reports raw prediction-set coverage separately from lexical and business gates.
Each execution creates a new report with exclusive creation: earlier results
remain available, including results obtained before any policy repair. Nothing
in this script selects thresholds, modifies datasets, confirms answers or saves
a customer profile. Synthetic results do not establish real-user validity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.all_question_ml import CONFORMAL_ALPHA, INTENT_LABELS, MIN_LEXICAL_COVERAGE, build_all_question_ml_resolver
from ai.conversation_engine import ConversationEngine
from ai.question_spec import BusinessSpec


def proportion(numerator, denominator):
    return numerator / denominator if denominator else None


def prediction_metrics(predictions, labels):
    """Marginal empirical coverage; no guarantee for accepted singletons."""
    n = len(labels)
    singleton = [i for i, result in enumerate(predictions) if len(result.prediction_set) == 1]
    singleton_correct = sum(next(iter(predictions[i].prediction_set)) == labels[i] for i in singleton)
    return {
        "n": n,
        "empirical_coverage": proportion(sum(label in result.prediction_set for label, result in zip(labels, predictions)), n),
        "average_set_size": proportion(sum(len(result.prediction_set) for result in predictions), n),
        "singleton_rate": proportion(len(singleton), n),
        "multi_class_rate": proportion(sum(len(result.prediction_set) > 1 for result in predictions), n),
        "empty_set_rate": proportion(sum(not result.prediction_set for result in predictions), n),
        "abstention_rate": proportion(n - len(singleton), n),
        "accepted_prediction_accuracy": proportion(singleton_correct, len(singleton)),
        "accepted_count": len(singleton),
        "accepted_correct_count": singleton_correct,
    }


def financial_metrics(rows, predictions):
    accepted = [i for i, option in enumerate(predictions) if option is not None]
    nonclear = [i for i, row in enumerate(rows) if row["bucket"] != "clarity"]
    false_clarity = [i for i in accepted if rows[i]["bucket"] != "clarity"]
    wrong_option = [i for i in accepted if rows[i]["bucket"] == "clarity" and predictions[i] != rows[i]["option_id"]]
    correct = sum(rows[i]["bucket"] == "clarity" and predictions[i] == rows[i]["option_id"] for i in accepted)
    return {
        "n": len(rows),
        "candidate_count": len(accepted),
        "candidate_rate": proportion(len(accepted), len(rows)),
        "abstention_rate": proportion(len(rows) - len(accepted), len(rows)),
        "accepted_financial_correctness": proportion(correct, len(accepted)),
        "accepted_correct_count": correct,
        "false_clarity_count": len(false_clarity),
        "false_clarity_rate_among_candidates": proportion(len(false_clarity), len(accepted)),
        "false_clarity_rate_among_true_nonclear": proportion(len(false_clarity), len(nonclear)),
        "true_nonclear_count": len(nonclear),
        "false_clarity_example_ids": [rows[i]["id"] for i in false_clarity],
        "wrong_option_count": len(wrong_option),
        "wrong_option_example_ids": [rows[i]["id"] for i in wrong_option],
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(data_directory, spec_path):
    spec = BusinessSpec.load(spec_path)
    # Fitting reads train and calibration only, before opening locked test.csv.
    resolver = build_all_question_ml_resolver(spec, data_directory)
    test_path = data_directory / "test.csv"
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("locked test data must contain only test rows")
    for row in rows:
        row["bucket"] = row["bucket"].lower()
        row["option_id"] = row["option_id"].strip() or None
        if row["question_id"] not in spec.question_order or row["bucket"] not in INTENT_LABELS:
            raise ValueError("test row contains an invalid question or semantic label")
        if row["option_id"] is not None:
            spec.option(row["question_id"], row["option_id"])
            if row["bucket"] != "clarity":
                raise ValueError("nonclear test row must not contain a financial option")

    intent_predictions = []
    option_predictions = []
    option_labels = []
    gated_options = []
    gated_option_labels = []
    base_candidates = []
    engine_candidates = []
    traces = []
    intercepted = []
    vetoed = []
    fallback = []
    recorded_without_confirmation = []

    for row in rows:
        qid, text = row["question_id"], row["text"]
        pair = resolver.predictors[qid]
        intent = pair.intent.predict(text)
        intent_predictions.append(intent)
        if row["bucket"] == "clarity" and row["option_id"] is not None:
            option = pair.options.predict(text)
            option_predictions.append(option)
            option_labels.append(row["option_id"])
            if intent.prediction_set == frozenset({"clarity"}):
                gated_options.append(option)
                gated_option_labels.append(row["option_id"])

        resolved = resolver.resolve(qid, text)
        base_candidates.append(resolved.candidate_option_id)
        # Each row is a fresh fictional single-turn conversation. These probes
        # do not claim to cover previous-answer conflicts or full save journeys.
        engine = ConversationEngine(spec, question_resolver=resolver)
        engine.state.current_index = spec.question_order.index(qid)
        response = engine.submit_answer(text)
        answer = engine.state.answers[qid]
        engine_candidate = answer.candidate_option_id
        engine_candidates.append(engine_candidate)
        attempted_ml = any(event.event_type == "ml_interpretation" for event in engine.state.audit)
        if resolved.candidate_option_id is not None and engine_candidate is None:
            (vetoed if attempted_ml else intercepted).append(row["id"])
        if engine_candidate is not None and engine_candidate != resolved.candidate_option_id:
            fallback.append(row["id"])
        if answer.selected_option_id is not None or response.get("recorded") is True or engine.state.saved_version is not None:
            recorded_without_confirmation.append(row["id"])
        traces.append({
            "id": row["id"], "question_id": qid, "text": text,
            "expected_bucket": row["bucket"], "expected_option_id": row["option_id"],
            "raw_intent_set": sorted(intent.prediction_set),
            "resolver_bucket": resolved.ordinary_bucket,
            "resolver_option_id": resolved.candidate_option_id,
            "resolver_intent_set": sorted(resolved.intent_set),
            "resolver_option_set": sorted(resolved.option_set),
            "resolver_reason": resolved.reason,
            "resolver_diagnostics": resolved.diagnostics,
            "engine_candidate_option_id": engine_candidate,
            "engine_action": response.get("action"), "engine_bucket": response.get("bucket"),
            "engine_reason": response.get("reason"), "engine_used_ml": attempted_ml,
            "engine_active_holds": sorted(engine.state.active_holds),
            "engine_recorded": response.get("recorded", False),
        })

    true_nonclear = [i for i, row in enumerate(rows) if row["bucket"] != "clarity"]
    intent_clear = [i for i, result in enumerate(intent_predictions) if result.prediction_set == frozenset({"clarity"})]
    false_intent_clear = [i for i in intent_clear if rows[i]["bucket"] != "clarity"]
    per_question = {}
    for qid in spec.question_order:
        indices = [i for i, row in enumerate(rows) if row["question_id"] == qid]
        qrows = [rows[i] for i in indices]
        per_question[qid] = {
            "intent_raw_conformal": prediction_metrics([intent_predictions[i] for i in indices], [r["bucket"] for r in qrows]),
            "resolver_financial": financial_metrics(qrows, [base_candidates[i] for i in indices]),
            "engine_financial_after_controls_and_business_veto": financial_metrics(qrows, [engine_candidates[i] for i in indices]),
            "intent_calibration_threshold": resolver.predictors[qid].intent.threshold,
            "option_calibration_threshold": resolver.predictors[qid].options.threshold,
        }
    source_paths = ["ai/all_question_ml.py", "ai/text_classifier.py", "ai/conformal.py",
                    "ai/conversation_engine.py", "ai/pre_model_controls.py", "ai/bounded_answers.py",
                    "ai/horizon_mapper.py", "ai/question_spec.py", "scripts/evaluate_all_question_ml.py"]
    counts = {}
    for split in ("train", "calibration", "test"):
        with (data_directory / f"{split}.csv").open(encoding="utf-8-sig", newline="") as handle:
            split_rows = list(csv.DictReader(handle))
        counts[split] = {"total": len(split_rows), "by_question": dict(Counter(r["question_id"] for r in split_rows)),
                         "by_bucket": dict(Counter(r["bucket"].lower() for r in split_rows))}

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "per-question TF-IDF + Logistic Regression + split conformal",
        "settings_frozen_before_test": {"alpha": CONFORMAL_ALPHA, "minimum_lexical_coverage": MIN_LEXICAL_COVERAGE,
                                         "random_state": 42, "word_ngrams": [1, 2], "class_weight": "balanced"},
        "provenance": {"resolver_source_hash": resolver.source_hash, "spec_version": spec.version,
                       "spec_sha256": file_hash(spec_path),
                       "dataset_sha256": {split: file_hash(data_directory / f"{split}.csv") for split in ("train", "calibration", "test")},
                       "implementation_sha256": {path: file_hash(ROOT / path) for path in source_paths},
                       "dataset_counts": counts},
        "interpretation_limits": [
            "Only fictional authored test examples were evaluated; no human comprehension or real-customer validation is established.",
            "Raw conformal coverage is empirical marginal prediction-set coverage, not singleton precision or a guarantee of no errors.",
            "A formal split-conformal coverage claim requires exchangeable calibration and future inputs; authored synthetic splits do not establish that assumption.",
            "Option metrics condition on truly clear examples with fixed financial labels. The selected-intent subset has selection bias and no independent coverage guarantee.",
            "Lexical filtering and business veto change acceptance behavior; post-veto acceptance has no formal conformal coverage guarantee.",
            "Fresh single-turn engine probes do not replace end-to-end confirmation, corrections, safety, consistency and save-gate tests.",
            "Settings were fixed before this evaluation; this script neither tunes models on test data nor modifies train/calibration/test files.",
        ],
        "intent_raw_conformal": prediction_metrics(intent_predictions, [r["bucket"] for r in rows]),
        "intent_false_clarity": {"count": len(false_intent_clear),
                                 "rate_among_intent_clarity_singletons": proportion(len(false_intent_clear), len(intent_clear)),
                                 "rate_among_true_nonclear": proportion(len(false_intent_clear), len(true_nonclear)),
                                 "example_ids": [rows[i]["id"] for i in false_intent_clear]},
        "options_raw_conformal_conditional_on_true_clarity": prediction_metrics(option_predictions, option_labels),
        "options_conditional_on_true_clarity_and_intent_clarity_singleton": prediction_metrics(gated_options, gated_option_labels),
        "resolver_financial": financial_metrics(rows, base_candidates),
        "engine_financial_after_controls_and_business_veto": financial_metrics(rows, engine_candidates),
        "engine_policy_comparison": {"before_ml_control_interception_ids": intercepted,
                                     "post_ml_business_veto_ids": vetoed,
                                     "unexpected_fallback_proposal_ids": fallback,
                                     "recorded_without_confirmation_ids": recorded_without_confirmation},
        "per_question": per_question,
        "row_results": traces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=ROOT / "data/nlp/all_questions")
    parser.add_argument("--spec", type=Path, default=ROOT / "config/business-decision-spec.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-label", default="initial_frozen_heldout")
    arguments = parser.parse_args()
    if arguments.output is not None and arguments.output.exists():
        parser.error("output already exists; earlier evaluation results must be preserved")
    report = evaluate(arguments.data_directory, arguments.spec)
    report["evaluation_label"] = arguments.evaluation_label
    output = arguments.output or ROOT / "reports/all_question_ml" / (
        datetime.now(timezone.utc).strftime("evaluation-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8] + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"report": str(output), "label": arguments.evaluation_label,
                      "test_rows": report["intent_raw_conformal"]["n"],
                      "base": report["resolver_financial"],
                      "engine": report["engine_financial_after_controls_and_business_veto"],
                      "policy": report["engine_policy_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
