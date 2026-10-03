from __future__ import annotations

from ai.conversation_factory import build_conversation_engine
from ai.conversation_state import AnswerStatus
from ai.d12_adapter import D12Resolution


class CountingResolver:
    def __init__(self, resolution: D12Resolution):
        self.resolution = resolution
        self.calls = 0

    def resolve(self, text: str) -> D12Resolution:
        self.calls += 1
        return self.resolution


def confirm_option(engine, question_id: str, option_id: str) -> None:
    assert engine.state.current_question_id == question_id
    response = engine.submit_answer(option_id)
    assert response["action"] == "confirm"
    confirmation = engine.confirm_current(True)
    assert confirmation["confirmed_question_id"] == question_id


def advance_to(engine, target_question_id: str) -> None:
    while engine.state.current_question_id != target_question_id:
        question = engine.current_question
        assert question is not None

        if question.id == "D12":
            response = engine.submit_answer(
                "I may realistically need this money in 7 years"
            )
        else:
            response = engine.submit_answer(question.options[0].id)

        assert response["action"] == "confirm"
        engine.confirm_current(True)


def complete_all_questions(engine) -> None:
    while engine.state.current_question_id is not None:
        question = engine.current_question
        assert question is not None

        if question.id == "D12":
            response = engine.submit_answer(
                "I may realistically need this money in 7 years"
            )
        else:
            response = engine.submit_answer(question.options[0].id)

        assert response["action"] == "confirm"
        engine.confirm_current(True)


def test_d12_real_free_text_maps_seven_years_to_medium():
    engine = build_conversation_engine()
    advance_to(engine, "D12")

    response = engine.submit_answer(
        "I may realistically need this money in 7 years"
    )

    assert response["action"] == "confirm"
    assert response["candidate_option_id"] == "D12_MEDIUM"
    assert response["recorded"] is False

    engine.confirm_current(True)

    assert engine.state.answers["D12"].selected_option_id == "D12_MEDIUM"


def test_cross_boundary_explicit_duration_requires_clarification():
    engine = build_conversation_engine()
    advance_to(engine, "D12")

    response = engine.submit_answer(
        "I may need it between 2 and 7 years"
    )

    assert response["action"] == "clarify"
    assert response["bucket"] in {"Undecided", "Confusion"}
    assert engine.state.answers["D12"].selected_option_id is None


def test_all_savings_next_year_is_short_candidate_plus_dependency_hold():
    engine = build_conversation_engine()
    advance_to(engine, "D12")

    response = engine.submit_answer(
        "This is all my savings and I may need it next year"
    )

    assert response["action"] == "confirm"
    assert response["candidate_option_id"] == "D12_SHORT"
    assert "same_money_dependency" in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id is None


def test_safety_stop_prevents_d12_resolver_execution():
    engine = build_conversation_engine()
    advance_to(engine, "D12")

    resolver = CountingResolver(
        D12Resolution(
            candidate_option_id="D12_LONG",
            ordinary_bucket="Clarity",
        )
    )
    engine.d12_resolver = resolver

    response = engine.submit_answer("I want to kill myself")

    assert response["action"] == "safety_stop"
    assert response["simulated_handoff"] is True
    assert resolver.calls == 0
    assert engine.state.stopped_for_safety is True


def test_three_failed_clarifications_finish_incomplete():
    engine = build_conversation_engine()

    first = engine.submit_answer("this does not answer the question")
    second = engine.submit_answer("still not answering it")
    third = engine.submit_answer("I still cannot give one of those meanings")

    assert first["action"] == "clarify"
    assert second["action"] == "clarify"
    assert third["action"] == "finish_incomplete_or_pause"
    assert third["clarification_turns"] == 3
    assert third["profile_complete"] is False


def test_correction_refreshes_version_and_invalidates_d12_d13():
    engine = build_conversation_engine()
    complete_all_questions(engine)

    assert engine.confirm_final_accuracy(True)["accepted"] is True
    assert engine.give_save_consent(True)["accepted"] is True

    previous_version = engine.state.profile_version

    result = engine.correct_answer("D1")

    assert engine.state.profile_version > previous_version
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    assert "D12" in result["invalidated_questions"]
    assert "D13" in result["invalidated_questions"]
    assert engine.state.answers["D12"].status == AnswerStatus.PENDING
    assert engine.state.answers["D13"].status == AnswerStatus.PENDING


def test_emergency_purpose_plus_long_horizon_creates_dependency_hold():
    engine = build_conversation_engine()

    confirm_option(engine, "D1", "D1_EMERGENCIES")
    advance_to(engine, "D12")

    response = engine.submit_answer("I may realistically need it in 15 years")
    assert response["action"] == "confirm"
    assert response["candidate_option_id"] == "D12_LONG"

    engine.confirm_current(True)

    assert (
        "emergency_long_horizon_dependency"
        in engine.state.active_holds
    )


def test_greater_preference_and_unacceptable_comfort_need_acknowledgement():
    engine = build_conversation_engine()
    advance_to(engine, "D14")

    confirm_option(engine, "D14", "D14_GREATER")
    confirm_option(engine, "D15", "D15_UNACCEPTABLE")

    assert (
        "d14_d15_tension_unacknowledged"
        in engine.state.active_holds
    )

    engine.acknowledge_preference_comfort_tension()

    assert (
        "d14_d15_tension_unacknowledged"
        not in engine.state.active_holds
    )


def test_final_accuracy_save_consent_and_save_are_separate():
    engine = build_conversation_engine()
    complete_all_questions(engine)

    completion = engine.completion_report()
    assert completion["all_15_handled"] is True
    assert completion["eligible_for_final_accuracy_confirmation"] is True

    accuracy = engine.confirm_final_accuracy(True)
    assert accuracy["accepted"] is True

    premature_save = engine.save()
    assert premature_save["saved"] is False
    assert premature_save["reason"] == "save_consent_stale_or_missing"

    consent = engine.give_save_consent(True)
    assert consent["accepted"] is True

    saved = engine.save()
    assert saved["saved"] is True
    assert "audit" not in saved["profile"]
