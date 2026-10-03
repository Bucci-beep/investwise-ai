from __future__ import annotations

import json

import pytest

from ai.conversation_engine import ConversationEngine
from ai.d12_adapter import D12Resolution
from ai.question_spec import BusinessSpec


def build_minimal_spec() -> BusinessSpec:
    questions = []
    for i in range(1, 16):
        qid = f"D{i}"
        if qid == "D1":
            options = [
                {"id": "D1_WEALTH", "label": "Building wealth without a specific purchase", "meaning": "wealth"},
                {"id": "D1_EMERGENCIES", "label": "Emergencies or unexpected costs", "meaning": "emergency"},
            ]
        elif qid == "D12":
            options = [
                {"id": "D12_SHORT", "label": "Within 3 years, including exactly 3 years", "meaning": "short"},
                {"id": "D12_MEDIUM", "label": "More than 3 years, up to and including 10 years", "meaning": "medium"},
                {"id": "D12_LONG", "label": "More than 10 years", "meaning": "long"},
            ]
        elif qid == "D13":
            options = [
                {"id": "D13_ESSENTIALS_DIFFICULT", "label": "Essential costs would become difficult", "meaning": "difficult"},
                {"id": "D13_ADJUSTMENT", "label": "I would need to adjust plans", "meaning": "adjustment"},
                {"id": "D13_UNAFFECTED", "label": "My plans would be unaffected", "meaning": "unaffected"},
            ]
        elif qid == "D14":
            options = [
                {"id": "D14_LOWER", "label": "Lower fluctuation preference", "meaning": "lower"},
                {"id": "D14_BALANCED", "label": "Balanced trade-off", "meaning": "balanced"},
                {"id": "D14_GREATER", "label": "Greater growth preference", "meaning": "greater"},
            ]
        elif qid == "D15":
            options = [
                {"id": "D15_UNACCEPTABLE", "label": "Unacceptable", "meaning": "unacceptable"},
                {"id": "D15_WORRIED_ACCEPTABLE", "label": "Worried but acceptable", "meaning": "worried"},
                {"id": "D15_COMFORTABLE", "label": "Comfortable", "meaning": "comfortable"},
            ]
        else:
            options = [{"id": f"{qid}_A", "label": f"{qid} option A", "meaning": "A"}]

        role = "optional_context" if qid in {"D2", "D3", "D8", "D9", "D10"} else "required_meaning"
        questions.append(
            {
                "id": qid,
                "order": i,
                "topic": qid,
                "question": f"Question {qid}",
                "completion_role": role,
                "main_options": options,
            }
        )

    return BusinessSpec(
        {
            "version": "v7-review",
            "question_order": [f"D{i}" for i in range(1, 16)],
            "questions": questions,
        }
    )


class FakeD12Resolver:
    def __init__(self, resolution):
        self.resolution = resolution
        self.calls = 0

    def resolve(self, text: str):
        self.calls += 1
        return self.resolution


def advance_to(engine: ConversationEngine, target: str):
    while engine.state.current_question_id != target:
        qid = engine.state.current_question_id
        assert qid is not None
        option = engine.spec.question(qid).options[0]
        result = engine.submit_answer(option.id)
        assert result["action"] == "confirm"
        engine.confirm_current(True)


def complete_with_first_options(engine: ConversationEngine):
    while engine.state.current_question_id is not None:
        qid = engine.state.current_question_id
        option = engine.spec.question(qid).options[0]
        result = engine.submit_answer(option.id)
        assert result["action"] == "confirm"
        engine.confirm_current(True)


def test_candidate_is_not_recorded_until_confirmation():
    engine = ConversationEngine(build_minimal_spec())
    result = engine.submit_answer("D1_WEALTH")
    assert result["action"] == "confirm"
    assert engine.state.answers["D1"].selected_option_id is None
    assert engine.state.answers["D1"].candidate_option_id == "D1_WEALTH"


def test_d12_candidate_comes_from_existing_resolver():
    resolver = FakeD12Resolver(
        D12Resolution(candidate_option_id="D12_MEDIUM", ordinary_bucket="Clarity")
    )
    engine = ConversationEngine(build_minimal_spec(), d12_resolver=resolver)
    advance_to(engine, "D12")

    result = engine.submit_answer("I may realistically need it in 7 years")
    assert result["candidate_option_id"] == "D12_MEDIUM"
    assert resolver.calls == 1


def test_d12_all_savings_keeps_resolver_answer_plus_dependency_flag():
    resolver = FakeD12Resolver(
        D12Resolution(candidate_option_id="D12_SHORT", ordinary_bucket="Clarity")
    )
    engine = ConversationEngine(build_minimal_spec(), d12_resolver=resolver)
    advance_to(engine, "D12")

    result = engine.submit_answer("This is all my savings and I may need it next year")
    assert result["candidate_option_id"] == "D12_SHORT"
    assert resolver.calls == 1
    assert "same_money_dependency" in engine.state.active_holds


def test_safety_runs_before_d12_ml():
    resolver = FakeD12Resolver(
        D12Resolution(candidate_option_id="D12_LONG", ordinary_bucket="Clarity")
    )
    engine = ConversationEngine(build_minimal_spec(), d12_resolver=resolver)
    advance_to(engine, "D12")

    result = engine.submit_answer("I want to kill myself")
    assert result["action"] == "safety_stop"
    assert resolver.calls == 0


def test_user_undecided_never_defaults_to_middle():
    engine = ConversationEngine(build_minimal_spec())
    advance_to(engine, "D14")

    result = engine.submit_answer("not sure")
    assert result["action"] == "clarify"
    assert engine.state.answers["D14"].selected_option_id is None
    assert engine.state.answers["D14"].candidate_option_id is None


def test_three_clarification_limit():
    engine = ConversationEngine(build_minimal_spec())

    assert engine.submit_answer("something unrelated")["action"] == "clarify"
    assert engine.submit_answer("still unrelated")["action"] == "clarify"
    third = engine.submit_answer("still not one of the fixed options")

    assert third["action"] == "finish_incomplete_or_pause"
    assert third["clarification_turns"] == 3


def test_d12_model_uncertainty_reasks_instead_of_recording():
    resolver = FakeD12Resolver(
        D12Resolution(model_uncertain=True, reason="model_uncertainty")
    )
    engine = ConversationEngine(build_minimal_spec(), d12_resolver=resolver)
    advance_to(engine, "D12")

    result = engine.submit_answer("I might need it sometime")
    assert result["action"] == "clarify"
    assert result["model_uncertain"] is True
    assert engine.state.answers["D12"].selected_option_id is None


def test_correction_invalidates_accuracy_and_save_consent():
    engine = ConversationEngine(build_minimal_spec())
    complete_with_first_options(engine)

    assert engine.confirm_final_accuracy(True)["accepted"] is True
    assert engine.give_save_consent(True)["accepted"] is True

    old_version = engine.state.profile_version
    engine.correct_answer("D1")

    assert engine.state.profile_version > old_version
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    assert engine.state.answers["D12"].selected_option_id is None
    assert engine.state.answers["D13"].selected_option_id is None


def test_accuracy_and_save_consent_are_separate():
    engine = ConversationEngine(build_minimal_spec())
    complete_with_first_options(engine)

    accuracy = engine.confirm_final_accuracy(True)
    assert accuracy["accepted"] is True

    early_save = engine.save()
    assert early_save["saved"] is False
    assert early_save["reason"] == "save_consent_stale_or_missing"

    consent = engine.give_save_consent(True)
    assert consent["accepted"] is True
    assert engine.save()["saved"] is True


def test_transient_audit_is_not_persisted():
    engine = ConversationEngine(build_minimal_spec())
    complete_with_first_options(engine)
    engine.confirm_final_accuracy(True)
    engine.give_save_consent(True)

    saved = engine.save()
    assert saved["saved"] is True
    assert "audit" not in saved["profile"]


def test_emergency_plus_long_horizon_creates_dependency_hold():
    engine = ConversationEngine(build_minimal_spec())

    engine.submit_answer("D1_EMERGENCIES")
    engine.confirm_current(True)
    advance_to(engine, "D12")

    engine.submit_answer("D12_LONG")
    engine.confirm_current(True)

    assert "emergency_long_horizon_dependency" in engine.state.active_holds


def test_greater_preference_plus_unacceptable_requires_acknowledgement():
    engine = ConversationEngine(build_minimal_spec())
    advance_to(engine, "D14")

    engine.submit_answer("D14_GREATER")
    engine.confirm_current(True)

    engine.submit_answer("D15_UNACCEPTABLE")
    engine.confirm_current(True)

    assert "d14_d15_tension_unacknowledged" in engine.state.active_holds

    engine.acknowledge_preference_comfort_tension()
    assert "d14_d15_tension_unacknowledged" not in engine.state.active_holds
