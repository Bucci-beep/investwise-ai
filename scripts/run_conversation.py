from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ai.conversation_factory import ROOT, build_d12_interpreter
from ai.conversation_runtime import ConversationRuntime, RuntimeResponse
from ai.conversation_state import ConversationState


STRESS_CASES = ROOT / "data/stress/d12_cases.csv"


def _format_mapping(response: RuntimeResponse) -> str:
    if response.deterministic_mapping_status is None:
        return "not evaluated"
    durations = ", ".join(str(value) for value in response.deterministic_durations_years)
    return f"{response.deterministic_mapping_status} ({durations or 'no durations'})"


def print_trace(response: RuntimeResponse, state: ConversationState) -> None:
    question = state.get_question(response.question_id)
    print("\nRAW INPUT")
    print(response.user_text)
    print("\nCONTROL / GOVERNANCE")
    print("type:", response.control_type)
    print("matched:", response.control_matched)
    print("matched rule:", response.matched_rule or "none")
    print("model called:", "YES" if response.should_call_model else "NO")
    print("flags:", list(response.governance_flags) or "none")
    print("\nSTAGE A")
    if response.control_matched:
        print("SKIPPED")
    else:
        print("probabilities:", response.model_probabilities or "not run")
        print("conformal set:", list(response.conformal_prediction_set) or "not run/empty")
        print("model uncertainty:", response.model_uncertain)
        print("semantic undecided:", response.semantic_undecided)
    print("\nDETERMINISTIC MAPPING")
    print("mapping:", _format_mapping(response))
    print("mapped category:", response.proposed_value)
    print("\nACTION")
    print(response.action)
    print(response.message)
    print("\nSTATE")
    print("clarification count:", response.clarification_count)
    print("proposed value:", response.proposed_value)
    print("recorded value:", response.recorded_value)
    print("pending confirmation:", question.pending_confirmation)
    print("question status:", response.question_status)


def run_interactive(question_id: str) -> int:
    runtime = ConversationRuntime(build_d12_interpreter())
    state = ConversationState()
    state.start_question(question_id)
    print("D12 asks for the earliest realistic time you may need any of this money.")
    print("Type Ctrl-D to exit.")
    while not state.stopped:
        try:
            user_text = input("\nYou: ")
        except EOFError:
            print()
            break
        response = runtime.handle_reply(question_id, user_text, state)
        print_trace(response, state)
        if response.action == "recorded":
            break
    return 0


def _expected_value(row: dict[str, str]) -> str | None:
    return row["expected_value"].strip() or None


def run_stress_test(question_id: str, path: Path = STRESS_CASES) -> int:
    runtime = ConversationRuntime(build_d12_interpreter())
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    passed = 0
    unsafe_recordings = 0
    wrong_confirmation_actions = 0
    clarification_cases = 0
    undecided_cases = 0
    uncertainty_cases = 0
    wrong_control_activations = 0
    missed_controls = 0
    model_called_when_control_should_fire = 0
    control_fired_when_model_should_run = 0
    control_counts = {key: 0 for key in ("delegation", "advice_request", "refusal", "distress", "pause", "stop", "self_harm")}
    failures: list[tuple[str, list[str]]] = []

    for row in rows:
        state = ConversationState()
        response = runtime.handle_reply(question_id, row["text"], state)
        expected_action = row["expected_action"]
        expected_value = _expected_value(row)
        expected_recorded = row["expected_recorded_before_confirmation"].strip().lower() == "true"
        expected_control = row["expected_control"].strip()
        expected_model_called = row["expected_model_called"].strip().lower() == "true"
        actual_recorded = response.recorded_value is not None
        reasons: list[str] = []
        if response.action != expected_action:
            reasons.append("action")
        if response.proposed_value != expected_value:
            reasons.append("proposed_value")
        if actual_recorded != expected_recorded:
            reasons.append("recorded_before_confirmation")
        if response.control_type != expected_control:
            reasons.append("control")
            if expected_control == "none":
                wrong_control_activations += 1
            elif response.control_type == "none":
                missed_controls += 1
        if response.should_call_model != expected_model_called:
            reasons.append("model_called")
        if expected_control != "none" and response.should_call_model:
            model_called_when_control_should_fire += 1
        if expected_model_called and response.control_matched:
            control_fired_when_model_should_run += 1
        if actual_recorded:
            unsafe_recordings += 1
        if (response.action == "confirm") != (expected_action == "confirm"):
            wrong_confirmation_actions += 1
        if expected_action in {"clarify", "reoffer", "explain/restate", "incomplete"}:
            clarification_cases += 1
        if response.semantic_undecided:
            undecided_cases += 1
        if response.model_uncertain:
            uncertainty_cases += 1
        if expected_control in control_counts:
            control_counts[expected_control] += 1
        ok = not reasons and not actual_recorded
        if ok:
            passed += 1
        else:
            failures.append((row["case_id"], reasons or ["unsafe_recording"]))

        print(f"\n{row['case_id']}: {row['text']}")
        print(f"expected action={expected_action}; actual action={response.action}")
        print(f"expected value={expected_value}; actual proposed={response.proposed_value}")
        print(f"expected control={expected_control}; actual control={response.control_type}")
        print(f"expected model called={expected_model_called}; actual={response.should_call_model}")
        print(f"recorded before confirmation={response.recorded_value}")
        print(f"model uncertainty={response.model_uncertain}; semantic undecided={response.semantic_undecided}")
        print(f"deterministic mapping={response.deterministic_mapping_status}")
        print("PASS" if ok else f"FAIL ({', '.join(reasons)})")

    total = len(rows)
    print("\nSTRESS TEST SUMMARY")
    print("total cases:", total)
    print("passed:", passed)
    print("failed:", total - passed)
    print("unsafe recordings:", unsafe_recordings)
    print("wrong confirmation actions:", wrong_confirmation_actions)
    print("wrong control activations:", wrong_control_activations)
    print("missed controls:", missed_controls)
    print("model-called-when-control-should-fire:", model_called_when_control_should_fire)
    print("control-fired-when-model-should-run:", control_fired_when_model_should_run)
    print("delegation cases:", control_counts["delegation"])
    print("advice cases:", control_counts["advice_request"])
    print("refusal cases:", control_counts["refusal"])
    print("distress cases:", control_counts["distress"])
    print("pause/stop cases:", control_counts["pause"] + control_counts["stop"])
    print("self-harm cases:", control_counts["self_harm"])
    print("clarification cases:", clarification_cases)
    print("semantic-undecided cases:", undecided_cases)
    print("model-uncertainty cases:", uncertainty_cases)
    print("failures by case:", failures or "none")
    return 0 if passed == total and unsafe_recordings == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded InvestWise conversation runtime")
    parser.add_argument("--question", default="D12", choices=["D12"])
    parser.add_argument("--stress-test", action="store_true")
    args = parser.parse_args()
    if args.stress_test:
        return run_stress_test(args.question)
    return run_interactive(args.question)


if __name__ == "__main__":
    raise SystemExit(main())
