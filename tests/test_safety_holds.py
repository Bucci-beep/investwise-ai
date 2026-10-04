"""Behavioural regressions for confirmation, care and material safety gates."""

import pytest

from ai.conversation_factory import build_legacy_comparison_engine as build_conversation_engine
from ai.d12_adapter import D12Resolution
from ai.horizon_mapper import HorizonMappingStatus, extract_supported_need_clause, map_explicit_horizon
from ai.pre_model_controls import check_pre_model_controls


def advance_to(engine, target):
    while engine.current_question is not None and engine.current_question.id != target:
        answer = engine.current_question.options[0].id
        result = engine.submit_answer(answer)
        assert result["action"] == "confirm", result
        engine.confirm_current(True)


def finish(engine, overrides=None, declines=()):
    overrides = overrides or {}
    while engine.current_question is not None:
        question = engine.current_question
        if question.id in declines:
            result = engine.submit_answer("Prefer not to answer")
            assert result["action"] == "continue"
            continue
        result = engine.submit_answer(overrides.get(question.id, question.options[0].id))
        assert result["action"] == "confirm", result
        engine.confirm_current(True)


@pytest.mark.parametrize(("horizon", "lock"), [("D12_SHORT", 2), ("D12_LONG", 20), ("The earliest need is in 2 years", 5)])
def test_need_before_access_preserves_need_and_blocks_completion(horizon, lock):
    engine = build_conversation_engine()
    finish(engine, {"D4": f"My fixed deposit is locked for {lock} years", "D12": horizon})
    selected = engine.state.answers["D12"].selected_option_id
    assert "need_before_access" in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is False
    assert engine.save()["saved"] is False
    assert engine.state.answers["D12"].selected_option_id == selected
    assert "may fall before" in engine.material_check_prompt()


def access_conflict_engine():
    engine = build_conversation_engine()
    finish(engine, {"D4": "My fixed deposit is locked for 5 years", "D12": "The earliest need is in 2 years"})
    return engine


def test_supported_access_funding_needs_fresh_versioned_confirmation():
    engine = access_conflict_engine()
    version = engine.state.profile_version
    answer = "My salary is available when the payment is due and will cover that payment without using this pot."
    result = engine.resolve_material_check(answer)
    assert result["action"] == "confirm_check"
    assert engine.pending_material_check["profile_version"] == version
    assert "funding_when_need_precedes_access" not in engine.state.answers["D12"].context_details
    assert engine.save()["saved"] is False
    result = engine.confirm_material_check(True)
    assert result["action"] == "material_check_resolved"
    assert engine.state.profile_version == version + 1
    assert engine.pending_material_check is None
    assert "need_before_access" not in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert engine.confirm_final_accuracy(True)["accepted"] is True
    assert engine.give_save_consent(True)["accepted"] is True
    saved = engine.save()
    details = saved["profile"]["confirmed_answers"]["D12"]["context_details"]
    assert "salary" in details["funding_when_need_precedes_access"]
    assert not any(key.startswith("material.") for key in details)
    assert "pending_option_id" not in details


@pytest.mark.parametrize("answer", [
    "My salary is available when the payment is due but cannot cover that payment.",
    "My salary is available when the payment is due and partly covers that payment.",
    "My salary is available when the payment is due and will cover that payment, but not without using this pot.",
    "Maybe my salary will cover that payment when the payment is due.",
])
def test_unsupported_funding_cannot_clear_access_hold(answer):
    engine = access_conflict_engine()
    result = engine.resolve_material_check(answer)
    assert result["action"] != "confirm_check"
    assert engine.pending_material_check is None
    assert "need_before_access" in engine.state.active_holds
    assert engine.save()["saved"] is False


def test_material_rejection_or_stale_confirmation_never_clears_hold():
    engine = access_conflict_engine()
    answer = "My salary is available when the payment is due and will cover that payment without using this pot."
    engine.resolve_material_check(answer)
    engine.confirm_material_check(False)
    assert "need_before_access" in engine.state.active_holds
    engine.resolve_material_check(answer)
    engine.correct_answer("D2")
    assert engine.pending_material_check is None
    assert engine.confirm_material_check(True)["recorded"] is False
    assert "need_before_access" in engine.state.active_holds


def test_material_cap_shares_d12_counter_and_excludes_help_care_accessibility():
    engine = access_conflict_engine()
    engine.state.answers["D12"].clarification_turns = 2
    assert engine.resolve_material_check("I need help")["action"] == "explain"
    assert engine.resolve_material_check("Please use short sentences")["action"] == "support_before_confirmation"
    engine.resume()
    assert engine.resolve_material_check("I am overwhelmed")["action"] == "care_pause"
    engine.resume()
    assert engine.state.answers["D12"].clarification_turns == 2
    assert engine.resolve_material_check("Not sure")["action"] == "finish_incomplete_or_pause"
    assert engine.state.answers["D12"].clarification_turns == 3
    assert engine.resolve_material_check("My salary is available when the payment is due and will cover that payment")["action"] == "finish_incomplete_or_pause"
    assert engine.state.answers["D12"].clarification_turns == 3


def test_same_pot_cannot_be_other_backup_even_if_context_declined():
    engine = build_conversation_engine()
    advance_to(engine, "D6")
    result = engine.submit_answer("I will use this same pot to cover all necessary living costs for the full year")
    assert result["action"] == "clarify"
    assert "same_pot_backup_conflict" in engine.state.active_holds
    engine.submit_answer("Prefer not to answer")
    finish(engine)
    assert engine.completion_report()["eligible_for_final_accuracy_confirmation"] is False
    result = engine.resolve_material_check("My separate savings are available during the full 12 months and cover all necessary costs without using this pot")
    assert result["action"] == "request_correction_target"
    assert result["question_id"] == "D6"
    assert "same_pot_backup_conflict" in engine.state.active_holds


def test_usable_backup_confirmation_clears_only_its_named_hold():
    engine = build_conversation_engine()
    advance_to(engine, "D6")
    result = engine.submit_answer("My pension deposit is locked for 5 years and covers all my costs")
    assert result["action"] == "clarify"
    assert "usable_backup_check" in engine.state.active_holds
    engine.submit_answer("D6_ALL")
    engine.confirm_current(True)
    finish(engine, {"D12": "This is all my savings and I may need it next year"})
    assert "same_money_dependency" in engine.state.active_holds
    result = engine.resolve_material_check("My separate savings are accessible during the full 12 months and cover all necessary costs without using this pot")
    assert result["action"] == "confirm_check"
    engine.confirm_material_check(True)
    assert "usable_backup_check" not in engine.state.active_holds
    assert "same_money_dependency" in engine.state.active_holds
    engine.clear_dependency_hold()
    assert not engine.state.active_holds
    engine.correct_answer("D2")
    assert "usable_backup_check" not in engine.state.active_holds
    engine.submit_answer("D2_AGE65_PLUS")
    engine.confirm_current(True)
    engine.correct_answer("D12")
    assert "usable_backup_check" in engine.state.active_holds


def test_generic_dependency_button_cannot_clear_material_access_conflict():
    engine = access_conflict_engine()
    engine.clear_dependency_hold()
    assert "need_before_access" in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is False


@pytest.mark.parametrize("horizon", ["D12_MEDIUM", "D12_LONG"])
def test_living_costs_longer_horizon_requires_dependency_check_not_short_default(horizon):
    engine = build_conversation_engine()
    finish(engine, {"D1": "D1_LIVING_COSTS", "D12": horizon})
    assert "living_costs_horizon_dependency" in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id == horizon
    assert engine.confirm_final_accuracy(True)["accepted"] is False


def test_accessibility_request_is_preserved_with_a_financial_candidate():
    engine = build_conversation_engine()
    result = engine.submit_answer("This pot is for emergencies. Please use short sentences")
    assert result["action"] == "support_before_confirmation"
    assert "short sentences" in result["help_prompt"]
    assert "accessibility" in engine.state.answers["D1"].independent_flags
    assert engine.state.answers["D1"].candidate_option_id == "D1_EMERGENCIES"
    assert engine.state.answers["D1"].clarification_turns == 0


@pytest.mark.parametrize(("overrides", "hold", "timing"), [
    ({"D12": "This is all my savings and I may need it next year"}, "same_money_dependency", "next year"),
    ({"D1": "D1_EMERGENCIES", "D12": "D12_LONG"}, "emergency_long_horizon_dependency", "in 15 years"),
    ({"D1": "D1_LIVING_COSTS", "D12": "D12_MEDIUM"}, "living_costs_horizon_dependency", "in 7 years"),
])
def test_generic_dependency_has_supported_free_text_confirmation_path(overrides, hold, timing):
    engine = build_conversation_engine()
    finish(engine, {**overrides, "D13": "D13_UNAFFECTED"})
    assert hold in engine.state.active_holds
    assert "20% loss" in engine.material_check_prompt()
    result = engine.resolve_material_check(f"The earliest realistic need for this same money is {timing}. In the 20% loss example I can still cover essentials and all my plans and budget stay unchanged.")
    assert result["action"] == "confirm_check", result
    assert result["check_id"] == hold
    assert "dependency_review" in result["context_details"]
    assert hold in engine.state.active_holds
    version = engine.state.profile_version
    assert engine.confirm_material_check(True)["action"] == "material_check_resolved"
    assert engine.state.profile_version == version + 1
    assert hold not in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is True
    assert engine.give_save_consent(True)["accepted"] is True
    saved = engine.save()
    assert saved["saved"] is True
    details = saved["profile"]["confirmed_answers"]["D12"]["context_details"]
    assert "dependency_review" in details
    assert not any(key.startswith(("material.", "access.")) for key in details)


def test_confirmed_dependency_can_retain_limited_capacity_without_forcing_another_group():
    engine = build_conversation_engine()
    finish(engine, {"D12": "This is all my savings and I may need it next year", "D13": "D13_ESSENTIALS_DIFFICULT"})
    result = engine.resolve_material_check("The earliest realistic need for this same money is next year. In the 20% loss example I could not pay my rent.")
    assert result["action"] == "confirm_check", result
    engine.confirm_material_check(True)
    assert engine.state.answers["D13"].selected_option_id == "D13_ESSENTIALS_DIFFICULT"
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert "same_money_dependency" not in engine.state.active_holds


@pytest.mark.parametrize(("reply", "target"), [
    ("The earliest realistic need is next year. In the 20% loss example essentials remain covered and all plans and budget stay unchanged.", "D12"),
    ("The earliest realistic need is in 15 years. In the 20% loss example I could not pay my rent.", "D13"),
])
def test_generic_dependency_changed_meaning_requires_correction(reply, target):
    engine = build_conversation_engine()
    finish(engine, {"D1": "D1_EMERGENCIES", "D12": "D12_LONG", "D13": "D13_UNAFFECTED"})
    result = engine.resolve_material_check(reply)
    assert result["action"] == "request_correction_target"
    assert result["question_id"] == target
    assert engine.state.answers["D12"].selected_option_id == "D12_LONG"
    assert engine.state.answers["D13"].selected_option_id == "D13_UNAFFECTED"
    assert "emergency_long_horizon_dependency" in engine.state.active_holds


def test_generic_dependency_does_not_accept_reassurance_or_clear_another_hold():
    engine = build_conversation_engine()
    finish(engine, {"D1": "D1_EMERGENCIES", "D12": "This is all my savings; the earliest realistic need is in 15 years", "D13": "D13_UNAFFECTED"})
    assert {"same_money_dependency", "emergency_long_horizon_dependency"} <= engine.state.active_holds
    assert engine.resolve_material_check("Everything is fine")["action"] == "clarify"
    result = engine.resolve_material_check("The earliest realistic need is in 15 years. In the 20% loss example essentials remain covered and all plans and budget stay unchanged.")
    assert result["check_id"] == "same_money_dependency"
    engine.confirm_material_check(True)
    assert "same_money_dependency" not in engine.state.active_holds
    assert "emergency_long_horizon_dependency" in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is False


def test_generic_dependency_acknowledgement_survives_age_but_not_scope_correction():
    engine = build_conversation_engine()
    finish(engine, {"D1": "D1_EMERGENCIES", "D12": "D12_LONG", "D13": "D13_UNAFFECTED"})
    engine.resolve_material_check("The earliest realistic need is in 15 years. In the 20% loss example essentials remain covered and all plans and budget stay unchanged.")
    engine.confirm_material_check(True)
    engine.correct_answer("D2")
    engine.submit_answer("D2_AGE65_PLUS")
    engine.confirm_current(True)
    assert "emergency_long_horizon_dependency" not in engine.state.active_holds
    engine.correct_answer("D12")
    finish(engine, {"D12": "D12_LONG"})
    assert "emergency_long_horizon_dependency" in engine.state.active_holds
    assert "dependency_review" not in engine.state.answers["D12"].context_details


def test_tension_acknowledgement_creates_new_version_and_invalidates_consent():
    engine = build_conversation_engine()
    finish(engine, {"D14": "D14_GREATER", "D15": "D15_UNACCEPTABLE"})
    version = engine.state.profile_version
    engine.state.final_accuracy_version = version
    engine.state.save_consent_version = version
    engine.acknowledge_preference_comfort_tension()
    assert engine.state.profile_version == version + 1
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    assert "d14_d15_tension_unacknowledged" not in engine.state.active_holds


def test_internal_access_and_material_markers_are_never_public_or_saved():
    engine = access_conflict_engine()
    result = engine.resolve_material_check("My salary is available when the payment is due and will cover that payment without using this pot.")
    assert result["action"] == "confirm_check"
    engine.confirm_material_check(True)
    playback = engine.final_playback()
    assert "access.available_after_years" in engine.state.answers["D4"].context_details
    assert "material.need_before_access.resolved" in engine.state.answers["D12"].context_details
    for row in playback["answers"]:
        assert not any(key.startswith(("material.", "access.", "understanding.")) for key in row["context_details"])
    engine.confirm_final_accuracy(True)
    engine.give_save_consent(True)
    for row in engine.save()["profile"]["confirmed_answers"].values():
        assert not any(key.startswith(("material.", "access.", "understanding.")) for key in row["context_details"])


def test_explicit_decline_is_clear_control_not_a_financial_candidate():
    engine = build_conversation_engine()
    result = engine.submit_answer("This pot is for retirement")
    assert result["action"] == "confirm"
    assert engine.state.answers["D1"].candidate_option_id == "D1_RETIREMENT"
    result = engine.submit_answer("I prefer not to answer")
    assert result["bucket"] == "Clarity"
    assert result["control_meaning"] == "declined"
    assert result["candidate_option_id"] is None
    record = engine.state.answers["D1"]
    assert record.status.value == "declined"
    assert record.candidate_option_id is None
    assert record.selected_option_id is None
    assert record.selected_option_label is None
    assert not record.context_details
    assert record.clarification_turns == 0
    assert engine.completion_report()["missing_required"]


def test_decline_clears_pending_currency_but_keeps_material_controls():
    engine = build_conversation_engine()
    advance_to(engine, "D3")
    assert engine.submit_answer("5000")["action"] == "clarify"
    assert "pending_option_id" in engine.state.answers["D3"].context_details
    result = engine.submit_answer("Prefer not to answer")
    assert result["bucket"] == "Clarity"
    assert engine.state.answers["D3"].context_details == {}
    advance_to(engine, "D6")
    engine.submit_answer("I will use this same pot to cover all costs for the full year")
    result = engine.submit_answer("Prefer not to answer")
    assert result["bucket"] == "Clarity"
    assert result["candidate_option_id"] is None
    assert "same_pot_backup_conflict" in engine.state.active_holds
    assert "same_pot_backup_conflict" in engine.state.answers["D6"].independent_flags


def test_declining_an_understanding_recheck_keeps_its_gate():
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    engine.submit_answer("D11_RECOVERY_ALWAYS")
    engine.confirm_current(True)
    result = engine.submit_answer("Prefer not to answer")
    assert result["bucket"] == "Clarity"
    assert engine.state.answers["D11"].selected_option_id is None
    assert "understanding_hold" in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is False


@pytest.mark.parametrize("text", [
    "Please use larger text",
    "Please use short sentences and one question at a time. I haven't answered yet.",
])
def test_accessibility_only_request_is_clear_control_with_no_financial_candidate(text):
    engine = build_conversation_engine()
    result = engine.submit_answer(text)
    assert result["action"] == "support_before_confirmation"
    assert result["bucket"] == "Clarity"
    assert result["control_meaning"] == "accessibility_request"
    assert result["candidate_option_id"] is None
    assert engine.state.answers["D1"].candidate_option_id is None
    assert engine.state.answers["D1"].selected_option_id is None
    assert engine.state.answers["D1"].clarification_turns == 0
    assert engine.current_question.id == "D1"


def test_accessibility_control_preserves_prior_candidate_without_classifying_new_financial_reply():
    engine = build_conversation_engine()
    engine.submit_answer("This pot is for retirement")
    result = engine.submit_answer("Please use larger text")
    assert result["bucket"] == "Clarity"
    assert result["candidate_option_id"] is None
    assert engine.state.answers["D1"].candidate_option_id == "D1_RETIREMENT"
    resumed = engine.resume()
    assert resumed["candidate_option_id"] == "D1_RETIREMENT"


def test_unsupported_financial_reply_with_help_remains_confusion():
    engine = build_conversation_engine()
    result = engine.submit_answer("I might do something with the money, but please explain")
    assert result["action"] == "support_before_confirmation"
    assert result["bucket"] == "Confusion"
    assert result.get("candidate_option_id") is None
    assert engine.state.answers["D1"].candidate_option_id is None
    assert engine.state.answers["D1"].clarification_turns == 0


def test_explicit_understood_but_undecided_horizon_stays_undecided_before_ml():
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    class NeverCalled:
        def resolve(self, text):
            raise AssertionError("Explicit indecision must be handled before the model")
    engine.d12_resolver = NeverCalled()
    result = engine.submit_answer("I understand earliest realistic need, but I cannot decide whether I may need it in two years or eight years")
    assert result["bucket"] == "Undecided"
    assert result["action"] == "clarify"
    assert engine.state.answers["D12"].candidate_option_id is None
    assert engine.state.answers["D12"].clarification_turns == 1


def test_accessibility_only_control_also_works_during_final_review():
    engine = build_conversation_engine()
    finish(engine)
    version = engine.state.profile_version
    result = engine.submit_answer("Please use larger text")
    assert result["action"] == "support_before_confirmation"
    assert result["bucket"] == "Clarity"
    assert result["candidate_option_id"] is None
    assert engine.state.profile_version == version
    assert engine.confirm_final_accuracy(True)["accepted"] is False
    assert engine.resume()["action"] == "final_review"


def test_explicit_break_with_clear_answer_pauses_before_confirmation_without_distress():
    engine = build_conversation_engine()
    result = engine.submit_answer("This pot is for retirement, but I need a break")
    assert result["action"] == "pace_pause"
    assert result["candidate_option_id"] == "D1_RETIREMENT"
    assert result["bucket"] == "Clarity"
    assert engine.state.paused
    record = engine.state.answers["D1"]
    assert record.selected_option_id is None
    assert "pause_or_stop" in record.independent_flags
    assert "safety_or_distress" not in record.independent_flags
    assert record.clarification_turns == 0
    assert engine.confirm_current(True)["action"] == "paused"
    assert engine.resume()["candidate_option_id"] == "D1_RETIREMENT"
    engine.confirm_current(True)
    assert record.selected_option_id == "D1_RETIREMENT"


def test_break_request_preserves_pending_candidate_and_exact_horizon_evidence():
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    answer = "The earliest realistic need is in 2 years"
    engine.submit_answer(answer)
    result = engine.submit_answer("I need a moment")
    assert result["action"] == "pace_pause"
    assert result["candidate_option_id"] == "D12_SHORT"
    assert engine.state.answers["D12"].raw_text == answer
    assert engine.state.answers["D12"].clarification_turns == 0
    assert "safety_or_distress" not in engine.state.answers["D12"].independent_flags


def test_grieving_and_explicit_break_preserve_both_care_and_control_flags():
    engine = build_conversation_engine()
    result = engine.submit_answer("I am grieving and I need a moment")
    assert result["action"] == "care_pause"
    flags = engine.state.answers["D1"].independent_flags
    assert {"safety_or_distress", "pause_or_stop"} <= flags
    assert engine.state.answers["D1"].candidate_option_id is None
    assert engine.state.answers["D1"].clarification_turns == 0


def test_accessibility_request_has_canonical_flag_and_help_flag_survives_candidate():
    engine = build_conversation_engine()
    engine.submit_answer("I need help")
    result = engine.submit_answer("This pot is for retirement. Please use larger text")
    assert result["action"] == "support_before_confirmation"
    assert {"accessibility", "accessibility_request", "help_request"} <= engine.state.answers["D1"].independent_flags
    assert engine.state.answers["D1"].candidate_option_id == "D1_RETIREMENT"


def test_d11_term_help_activates_understanding_check_without_consuming_attempt():
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    result = engine.submit_answer("What does permanent loss mean?")
    assert result["action"] == "explain"
    assert result["bucket"] == "Confusion"
    assert {"help_request", "understanding_hold"} <= engine.state.answers["D11"].independent_flags
    assert "understanding_hold" in engine.state.active_holds
    assert engine.state.answers["D11"].clarification_turns == 0
    assert engine.submit_answer("D11_LOSS_POSSIBLE")["action"] == "understanding_hold"
    result = engine.submit_answer("I could permanently lose some or all of the original money and recovery is not guaranteed")
    assert result["action"] == "confirm"
    engine.confirm_current(True)
    assert "understanding_hold" not in engine.state.active_holds


@pytest.mark.parametrize("backup", ["D6_NONE", "D6_SOME"])
def test_limited_backup_long_horizon_essential_difficulty_requires_actual_funding_check(backup):
    engine = build_conversation_engine()
    finish(engine, {"D5": "D5_NO_REGULAR", "D6": backup, "D12": "D12_LONG", "D13": "D13_ESSENTIALS_DIFFICULT", "D14": "D14_GREATER", "D15": "D15_WORRIED_ACCEPTABLE"})
    assert "essential_funding_timing" in engine.state.active_holds
    assert "actually usable" in engine.material_check_prompt()
    assert engine.confirm_final_accuracy(True)["accepted"] is False
    assert engine.state.answers["D12"].selected_option_id == "D12_LONG"
    assert engine.state.answers["D14"].selected_option_id == "D14_GREATER"
    result = engine.resolve_material_check("The earliest realistic need remains in 15 years. In the 20% loss example I could not pay rent.")
    assert result["action"] == "clarify"
    assert engine.pending_material_check is None
    assert "essential_funding_timing" in engine.state.active_holds


def essential_funding_engine():
    engine = build_conversation_engine()
    finish(engine, {"D5": "D5_NO_REGULAR", "D6": "D6_NONE", "D12": "D12_LONG", "D13": "D13_ESSENTIALS_DIFFICULT"})
    return engine


def test_supported_earlier_need_requires_d12_correction_and_short_limited_profile_can_complete():
    engine = essential_funding_engine()
    result = engine.resolve_material_check("I need this same money next year to pay my rent.")
    assert result["action"] == "request_correction_target"
    assert result["question_id"] == "D12"
    assert engine.state.answers["D12"].selected_option_id == "D12_LONG"
    engine.correct_answer("D12")
    result = engine.submit_answer("I need this same money next year to pay my rent.")
    assert result["action"] == "confirm"
    engine.confirm_current(True)
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert engine.state.answers["D13"].selected_option_id == "D13_ESSENTIALS_DIFFICULT"
    assert "essential_funding_timing" not in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is True


def test_all_year_usable_other_funding_requires_correct_backup_instead_of_clearing_hold():
    engine = essential_funding_engine()
    result = engine.resolve_material_check("My separate savings are accessible during the full 12 months and cover all necessary costs without using this pot. The earliest realistic need for this same money remains in 15 years.")
    assert result["action"] == "request_correction_target"
    assert result["question_id"] == "D6"
    assert engine.state.answers["D6"].selected_option_id == "D6_NONE"
    assert "essential_funding_timing" in engine.state.active_holds
    assert engine.pending_material_check is None


def test_essential_funding_cannot_accept_unknown_or_partial_funds_and_uses_d12_cap():
    engine = essential_funding_engine()
    engine.state.answers["D12"].clarification_turns = 2
    assert engine.resolve_material_check("I need help")["action"] == "explain"
    assert engine.state.answers["D12"].clarification_turns == 2
    result = engine.resolve_material_check("Not sure")
    assert result["action"] == "finish_incomplete_or_pause"
    assert engine.state.answers["D12"].clarification_turns == 3
    assert "essential_funding_timing" in engine.state.active_holds
    assert engine.save()["saved"] is False


@pytest.mark.parametrize("backup", ["D6_NONE", "D6_SOME"])
def test_no_backup_and_unaffected_twenty_percent_scenario_remain_separate(backup):
    engine = build_conversation_engine()
    finish(engine, {"D6": backup, "D12": "D12_LONG", "D13": "D13_UNAFFECTED"})
    assert "essential_funding_timing" not in engine.state.active_holds
    assert engine.state.answers["D6"].selected_option_id == backup
    assert engine.state.answers["D13"].selected_option_id == "D13_UNAFFECTED"
    assert engine.confirm_final_accuracy(True)["accepted"] is True


def test_actual_same_pot_rent_disclosure_remains_after_hypothetical_comfort():
    engine = build_conversation_engine()
    while engine.current_question.id != "D15":
        question = engine.current_question
        result = engine.submit_answer({"D12": "The earliest realistic need is in 15 years with no earlier need", "D13": "D13_UNAFFECTED"}.get(question.id, question.options[0].id))
        assert result["action"] == "confirm"
        engine.confirm_current(True)
    result = engine.submit_answer("I cannot imagine costs covered because in reality I need this money for rent.")
    assert result.get("candidate_option_id") is None
    assert "same_money_dependency" in engine.state.active_holds
    result = engine.submit_answer("Under this covered-cost example I would worry but could accept it.")
    assert result["candidate_option_id"] == "D15_WORRIED_ACCEPTABLE"
    engine.confirm_current(True)
    assert "same_money_dependency" in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id == "D12_LONG"
    assert engine.state.answers["D13"].selected_option_id == "D13_UNAFFECTED"
    assert engine.confirm_final_accuracy(True)["accepted"] is False
    assert engine.save()["saved"] is False


@pytest.mark.parametrize("text", [
    "Under this covered-cost example I would need this money for rent if my salary stopped.",
    "If I need this money for rent, I would be worried.",
    "In this hypothetical example my rent is paid from this pot.",
    "I do not need this money for rent.",
])
def test_hypothetical_or_negated_rent_clause_does_not_create_actual_dependency(text):
    engine = build_conversation_engine()
    advance_to(engine, "D15")
    engine.submit_answer(text)
    assert "same_money_dependency" not in engine.state.active_holds
    assert "same_money_dependency" not in engine.state.answers["D15"].independent_flags


def test_late_actual_dependency_invalidates_old_review_consent_and_stays_internal():
    engine = build_conversation_engine()
    finish(engine, {"D12": "D12_LONG", "D13": "D13_UNAFFECTED"})
    engine.confirm_final_accuracy(True)
    engine.give_save_consent(True)
    version = engine.state.profile_version
    engine.submit_answer("In reality I need this same money for essential costs.")
    assert engine.state.profile_version == version + 1
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    assert "same_money_dependency" in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id == "D12_LONG"
    assert engine.state.answers["D13"].selected_option_id == "D13_UNAFFECTED"
    assert engine.save()["saved"] is False
    assert not any(key.startswith("material.") for row in engine.final_playback()["answers"] for key in row["context_details"])


def test_named_instant_access_savings_full_tuition_payment_requires_confirmation():
    engine = access_conflict_engine()
    result = engine.resolve_material_check("Separate instant-access savings already cover the full tuition payment and will be usable when it is due in two years. Those savings are outside this fixed-term pot.")
    assert result["action"] == "confirm_check", result
    assert "separate instant-access savings" in result["context_details"]["funding_when_need_precedes_access"]
    assert "need_before_access" in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert engine.state.answers["D4"].selected_option_id == "D4_FIXED_TERM"
    assert engine.confirm_material_check(True)["action"] == "material_check_resolved"
    assert "need_before_access" not in engine.state.active_holds


@pytest.mark.parametrize("text", [
    "Separate instant-access savings cover only part of the tuition payment and will be usable when it is due in two years.",
    "Separate instant-access savings cover the full tuition payment but will be unavailable when it is due in two years.",
    "Separate instant-access savings do not cover the full tuition payment and will be usable when it is due in two years.",
    "Separate instant-access savings are usable when it is due in two years, but the tuition payment will not be fully covered.",
])
def test_qualified_tuition_payment_still_requires_full_usable_funding(text):
    engine = access_conflict_engine()
    result = engine.resolve_material_check(text)
    assert result["action"] != "confirm_check", result
    assert engine.pending_material_check is None
    assert "need_before_access" in engine.state.active_holds
    assert engine.save()["saved"] is False


@pytest.mark.parametrize("text", ["2-12 years", "2–12 years", "3-5 years", "three—five years"])
def test_cross_boundary_dash_range_never_maps_to_last_endpoint(text):
    mapped = map_explicit_horizon(text)
    assert mapped.status == HorizonMappingStatus.AMBIGUOUS
    assert mapped.category is None
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    result = engine.submit_answer(f"The earliest need is in {text}")
    assert result["action"] == "clarify"
    assert engine.state.answers["D12"].candidate_option_id is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("three years plus one day", "D12_MEDIUM"), ("ten years and one day", "D12_LONG"), ("two weeks", "D12_SHORT")],
)
def test_smaller_time_units_are_not_ignored(text, expected):
    assert map_explicit_horizon(text).category == expected
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    result = engine.submit_answer(f"The earliest need is in {text}")
    # The semantic gate may abstain; it must never propose the boundary below.
    assert result.get("candidate_option_id") in {None, expected}


def test_maturity_date_is_not_the_earliest_need():
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    result = engine.submit_answer("My fixed deposit matures in six years")
    assert result["action"] == "clarify"
    assert result["reason"] == "availability_or_goal_is_not_earliest_need"
    assert engine.state.answers["D12"].candidate_option_id is None


@pytest.mark.parametrize("belief", ["D11_RECOVERY_ALWAYS", "D11_CAPITAL_PROTECTED"])
def test_confirmed_misconception_is_not_a_supported_understanding_pass(belief):
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    assert engine.submit_answer(belief)["action"] == "confirm"
    result = engine.confirm_current(True)
    assert result["action"] == "understanding_hold"
    assert "Recovery is not guaranteed" in result["explanation"]
    assert "own words" in result["followup"]
    assert engine.current_question.id == "D11"
    assert "understanding_hold" in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)["accepted"] is False
    assert engine.give_save_consent(True)["accepted"] is False
    assert engine.save()["saved"] is False


@pytest.mark.parametrize("reply", ["yes", "D11_LOSS_POSSIBLE", "Some or all of the original money could be permanently lost"])
def test_yes_or_supplied_selection_cannot_bypass_own_word_recheck(reply):
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    engine.submit_answer("D11_RECOVERY_ALWAYS")
    engine.confirm_current(True)
    result = engine.submit_answer(reply)
    assert result["action"] == "understanding_hold"
    assert "understanding_hold" in engine.state.active_holds
    assert engine.state.answers["D11"].candidate_option_id is None


def test_supported_teachback_needs_fresh_confirmation_before_hold_clears():
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    engine.submit_answer("D11_CAPITAL_PROTECTED")
    engine.confirm_current(True)
    result = engine.submit_answer("I might permanently lose some or all of the original money, and recovery is not guaranteed.")
    assert result["action"] == "confirm"
    assert result["candidate_option_id"] == "D11_LOSS_POSSIBLE"
    assert "understanding_hold" in engine.state.active_holds
    assert engine.state.answers["D11"].selected_option_id is None
    engine.confirm_current(True)
    assert "understanding_hold" not in engine.state.active_holds
    assert engine.current_question.id == "D12"


def test_rejected_teachback_does_not_remove_the_understanding_requirement():
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    engine.submit_answer("D11_RECOVERY_ALWAYS")
    engine.confirm_current(True)
    engine.submit_answer("Some or all of the original money may be permanently lost, and recovery is not guaranteed.")
    engine.confirm_current(False)
    result = engine.submit_answer("D11_LOSS_POSSIBLE")
    assert result["action"] == "understanding_hold"
    assert engine.state.answers["D11"].candidate_option_id is None


def test_age_correction_cannot_clear_an_unresolved_dependency():
    engine = build_conversation_engine()
    finish(engine, {"D12": "This is all my savings and I may need it next year"})
    assert "same_money_dependency" in engine.state.active_holds
    result = engine.correct_answer("D2")
    assert result["invalidated_questions"] == ["D2"]
    assert "same_money_dependency" in engine.state.active_holds
    engine.submit_answer("D2_AGE65_PLUS")
    engine.confirm_current(True)
    assert engine.save()["saved"] is False


def test_dependency_disclosure_on_a_context_question_is_not_ignored():
    engine = build_conversation_engine()
    result = engine.submit_answer("All my savings are for retirement")
    assert result["action"] == "confirm"
    assert result["candidate_option_id"] == "D1_RETIREMENT"
    assert "same_money_dependency" in engine.state.active_holds


def test_ordinary_distress_offers_pacing_without_inventing_a_financial_answer():
    engine = build_conversation_engine()
    result = engine.submit_answer("I am overwhelmed")
    assert result["action"] == "care_pause"
    assert "take this at your pace" in result["pacing_text"]
    assert engine.state.answers["D1"].clarification_turns == 0
    assert engine.state.answers["D1"].candidate_option_id is None
    assert engine.state.paused is True
    assert engine.resume()["action"] == "resume"
    assert "care_pacing" not in engine.state.active_holds


def test_clear_answer_plus_distress_retains_candidate_but_waits_for_pacing_choice():
    engine = build_conversation_engine()
    result = engine.submit_answer("This money is for retirement. My wife died")
    assert result["action"] == "care_pause"
    assert result["candidate_option_id"] == "D1_RETIREMENT"
    assert engine.confirm_current(True)["action"] == "paused"
    resumed = engine.resume()
    assert resumed["candidate_option_id"] == "D1_RETIREMENT"
    engine.confirm_current(True)
    assert engine.state.answers["D1"].selected_option_id == "D1_RETIREMENT"


def test_clear_answer_plus_help_gets_support_before_confirmation():
    engine = build_conversation_engine()
    result = engine.submit_answer("This money is for retirement, please explain this question")
    assert result["action"] == "support_before_confirmation"
    assert result["candidate_option_id"] == "D1_RETIREMENT"
    assert engine.state.answers["D1"].clarification_turns == 0
    assert engine.confirm_current(True)["action"] == "paused"
    assert engine.resume()["candidate_option_id"] == "D1_RETIREMENT"


def test_stop_during_care_ends_run_instead_of_resuming():
    engine = build_conversation_engine()
    engine.submit_answer("I am overwhelmed")
    assert engine.submit_answer("stop")["action"] == "stop"
    assert engine.resume()["action"] == "stop"
    assert engine.submit_answer("D1_RETIREMENT")["action"] == "stop"
    assert engine.save()["saved"] is False


def test_clarification_cap_is_enforced_in_the_engine():
    engine = build_conversation_engine()
    for _ in range(3):
        engine.submit_answer("not sure")
    result = engine.submit_answer("D1_RETIREMENT")
    assert result["action"] == "finish_incomplete_or_pause"
    assert engine.state.answers["D1"].clarification_turns == 3
    assert engine.state.answers["D1"].candidate_option_id is None


def test_conditional_context_declines_allow_completion_without_a_material_hold():
    engine = build_conversation_engine()
    finish(engine, declines={"D4", "D5", "D6", "D7"})
    assert engine.completion_report()["eligible_for_final_accuracy_confirmation"] is True


def test_scope_and_loss_premise_corrections_invalidate_dependent_meanings():
    for corrected in ["D1", "D11"]:
        engine = build_conversation_engine()
        finish(engine)
        result = engine.correct_answer(corrected)
        assert {"D12", "D13", "D14", "D15"}.issubset(result["invalidated_questions"])
        assert engine.confirm_final_accuracy(True)["accepted"] is False


def test_rejecting_currency_candidate_removes_pending_money_context():
    engine = build_conversation_engine()
    advance_to(engine, "D3")
    assert engine.submit_answer("GBP 2000")["action"] == "confirm"
    engine.confirm_current(False)
    result = engine.submit_answer("EUR")
    assert result["action"] == "clarify"
    assert engine.state.answers["D3"].candidate_option_id is None


def test_final_save_requires_current_gates_even_if_consent_was_previously_given():
    engine = build_conversation_engine()
    finish(engine)
    assert engine.confirm_final_accuracy(True)["accepted"] is True
    assert engine.give_save_consent(True)["accepted"] is True
    engine.submit_answer("I am overwhelmed")
    assert engine.give_save_consent(True)["accepted"] is False
    assert engine.save()["saved"] is False


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("I would like fifteen years of investment, but I realistically need part of this same money to pay living costs next year.", "D12_SHORT"),
        ("The earliest realistic need is in four years, although I ideally want to invest for fifteen years.", "D12_MEDIUM"),
        ("Correction: I need £1,000 of this same included pot for rent next year, and no part earlier. Replace 15 years.", "D12_SHORT"),
        ("Actually I need this same money next year instead of fifteen years.", "D12_SHORT"),
    ],
)
def test_explicit_actual_need_is_distinguished_from_ideal_or_replaced_horizon(reply, expected):
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    result = engine.submit_answer(reply)
    assert result["action"] == "confirm", result
    assert result["candidate_option_id"] == expected
    assert engine.state.answers["D12"].raw_text == reply
    assert engine.state.answers["D12"].selected_option_id is None


@pytest.mark.parametrize(
    "reply",
    [
        "I want to invest for fifteen years, but I may need it next year if my pension is not enough.",
        "The earliest need is in two years or after twelve years; I cannot determine which.",
        "I need it in four years, but I need part of it in fifteen years too.",
    ],
)
def test_competing_or_conditional_actual_needs_are_not_reduced_to_the_smallest_number(reply):
    assert extract_supported_need_clause(reply) is None
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    result = engine.submit_answer(reply)
    assert result["action"] == "clarify", result
    assert engine.state.answers["D12"].candidate_option_id is None


@pytest.mark.parametrize("reply", [
    "I am about to take my own life. I need help staying safe now.",
    "I'm going to take my own life tonight.",
    "Im gonna take my life now.",
    "I will take my own life.",
    "I plan to take my own life.",
    "I'm planning to take my life.",
    "I intend to take my own life.",
    "I do not have an emergency fund. I am about to take my own life.",
])
def test_first_person_take_life_danger_stops_before_d12_and_cannot_resume(reply):
    engine = build_conversation_engine()
    advance_to(engine, "D12")

    class NeverResolve:
        def resolve(self, text):
            pytest.fail("Urgent safety controls must run before financial interpretation")

    engine.d12_resolver = NeverResolve()
    engine.submit_answer("pause")
    result = engine.submit_answer(reply)
    assert result["action"] == "safety_stop"
    assert engine.state.stopped_for_safety is True
    assert engine.state.answers["D12"].candidate_option_id is None
    assert engine.state.answers["D12"].clarification_turns == 0
    assert engine.confirm_current(True)["action"] == "safety_stop"
    assert engine.resume()["action"] == "safety_stop"
    assert engine.save()["saved"] is False


@pytest.mark.parametrize("reply", [
    "I am not going to take my own life.",
    "I will not take my own life.",
    "I do not want to take my life.",
    "I am about to take my own life insurance documents to the bank.",
])
def test_negated_take_life_intent_and_life_insurance_are_not_urgent(reply):
    assert check_pre_model_controls(reply).action != "safety_stop"


@pytest.mark.parametrize(("qid", "reply", "expected"), [
    ("D1", "This money is for retirement. Please pause here.", "D1_RETIREMENT"),
    ("D2", "I am 17 and need this conversation paused.", "D2_UNDER18"),
    ("D6", "No other income or money could cover any of my necessary costs for the full year. I want to pause now.", "D6_NONE"),
    ("D8", "I personally held cryptoassets; stop here please.", "D8_CRYPTO"),
    ("D9", "I instructed three investment purchases or sales in the last twelve months; pause please.", "D9_COUNT1_5"),
    ("D11", "For market investments I believe the original capital cannot fall. I need you to pause.", "D11_CAPITAL_PROTECTED"),
    ("D12", "The earliest realistic need for this money is in eleven years, with no earlier need for any part. Please pause.", "D12_LONG"),
])
def test_mixed_explicit_pause_retains_supported_candidate_without_confirmation(qid, reply, expected):
    engine = build_conversation_engine()
    advance_to(engine, qid)
    version = engine.state.profile_version
    result = engine.submit_answer(reply)
    record = engine.state.answers[qid]
    assert result["action"] == "pace_pause"
    assert result["candidate_option_id"] == expected
    assert result["bucket"] == "Clarity"
    assert "confirmation_text" not in result
    assert record.raw_text == reply
    assert record.selected_option_id is None
    assert record.clarification_turns == 0
    assert "pause_or_stop" in record.independent_flags
    assert engine.state.profile_version == version
    assert engine.confirm_current(True)["action"] == "paused"
    assert engine.resume()["candidate_option_id"] == expected
    if qid == "D11":
        assert "understanding_hold" in record.independent_flags
        assert "understanding_hold" in engine.state.active_holds
        assert engine.confirm_current(True)["action"] == "understanding_hold"
    else:
        engine.confirm_current(True)
        assert record.selected_option_id == expected


@pytest.mark.parametrize("reply", [
    "I want to pause now.", "pause please", "stop here please", "I need this conversation paused.",
])
def test_pure_pause_variants_preserve_existing_candidate_without_d12_execution(reply):
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    answer = "The earliest realistic need is in two years"
    assert engine.submit_answer(answer)["candidate_option_id"] == "D12_SHORT"

    class NeverResolve:
        def resolve(self, text):
            pytest.fail("A pure pause request must not invoke financial interpretation")

    engine.d12_resolver = NeverResolve()
    assert engine.submit_answer(reply)["action"] == "pause"
    assert engine.state.answers["D12"].raw_text == answer
    assert engine.confirm_current(True)["action"] == "paused"
    assert engine.resume()["candidate_option_id"] == "D12_SHORT"


@pytest.mark.parametrize("reply", [
    "I do not want to pause now.", "I do not need this conversation paused.",
    "I may need this money when I stop working.",
])
def test_negated_pause_requests_and_stopping_work_do_not_pause(reply):
    assert check_pre_model_controls(reply).action not in {"pause", "pace_answer", "stop"}


def test_unrelated_negation_does_not_suppress_a_later_pause():
    engine = build_conversation_engine()
    result = engine.submit_answer("I do not want to pause now. This money is for retirement. Actually, please pause here.")
    assert result["action"] == "pace_pause"
    assert engine.state.answers["D1"].clarification_turns == 0
    assert engine.confirm_current(True)["action"] == "paused"


@pytest.mark.parametrize("reply", [
    "My fixed deposit matures in six years. Please explain this question.",
    "My fixed deposit matures in six years. Please use short sentences.",
])
def test_support_for_access_date_does_not_consume_a_d12_factual_attempt(reply):
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    result = engine.submit_answer(reply)
    assert result["action"] == "support_before_confirmation"
    assert engine.state.answers["D12"].candidate_option_id is None
    assert engine.state.answers["D12"].clarification_turns == 0


@pytest.mark.parametrize(("reply", "expected_action"), [
    ("I am overwhelmed", "care_pause"),
    ("My fixed deposit matures in six years. Please explain this question.", "support_before_confirmation"),
    ("I need a break", "pace_pause"),
])
def test_d12_support_controls_do_not_count_when_no_resolver_is_configured(reply, expected_action):
    engine = build_conversation_engine()
    advance_to(engine, "D12")
    engine.d12_resolver = None
    result = engine.submit_answer(reply)
    assert result["action"] == expected_action
    assert engine.state.answers["D12"].clarification_turns == 0


def test_d12_support_does_not_count_when_resolver_abstains_without_a_bucket():
    engine = build_conversation_engine()
    advance_to(engine, "D12")

    class EmptyResolution:
        def resolve(self, text):
            return D12Resolution()

    engine.d12_resolver = EmptyResolution()
    result = engine.submit_answer("I need support. Please explain this question.")
    assert result["action"] == "support_before_confirmation"
    assert engine.state.answers["D12"].clarification_turns == 0


def test_unconfirmed_misconception_hold_is_provisional_but_confirmed_hold_needs_teachback():
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    engine.submit_answer("D11_RECOVERY_ALWAYS")
    assert "understanding_hold" in engine.state.answers["D11"].independent_flags
    assert "understanding_hold" in engine.state.active_holds
    engine.confirm_current(False)
    assert "understanding_hold" not in engine.state.active_holds
    engine.submit_answer("D11_CAPITAL_PROTECTED")
    engine.confirm_current(True)
    assert "understanding_hold" in engine.state.active_holds
    assert engine.submit_answer("D11_LOSS_POSSIBLE")["action"] == "understanding_hold"


@pytest.mark.parametrize(("qid", "reply", "expected"), [
    ("D2", "Sorry, correction: I turned sixty-five yesterday.", "D2_AGE65_PLUS"),
    ("D2", "I keep saying sixty-four out of habit, sorry, correction: I turned sixty-five yesterday. That is my age now.", "D2_AGE65_PLUS"),
    ("D11", "I used to think waiting fixed it, but that is not my answer now: with market investments, some or all of the money I started with could be gone for good. Getting it back is not guaranteed, however long I wait.", "D11_LOSS_POSSIBLE"),
])
def test_explicit_correction_markers_preserve_supported_revised_candidate_and_flag(qid, reply, expected):
    engine = build_conversation_engine()
    advance_to(engine, qid)
    result = engine.submit_answer(reply)
    assert result["action"] == "confirm"
    assert result["candidate_option_id"] == expected
    assert "correction" in engine.state.answers[qid].independent_flags
    assert engine.state.answers[qid].selected_option_id is None


def test_help_during_misconception_teachback_does_not_consume_a_factual_attempt():
    engine = build_conversation_engine()
    advance_to(engine, "D11")
    engine.submit_answer("D11_CAPITAL_PROTECTED")
    engine.confirm_current(True)
    record = engine.state.answers["D11"]
    attempts = record.clarification_turns
    result = engine.submit_answer("For market investments I believe the original capital cannot fall. Please explain this question.")
    assert result["action"] == "support_before_confirmation"
    assert "understanding_hold" in engine.state.active_holds
    assert record.candidate_option_id is None
    assert record.clarification_turns == attempts


def cash_account_access_engine():
    engine = build_conversation_engine()
    finish(engine, {"D4": "My fixed deposit is locked for four years", "D6": "D6_SOME", "D12": "The earliest need is in two years"})
    return engine


def test_current_separate_cash_account_can_stage_payment_funding_without_changing_backup():
    engine = cash_account_access_engine()
    version = engine.state.profile_version
    reply = "I have now identified a separate cash account that I can use on the payment date, and it covers that payment in full."
    result = engine.resolve_material_check(reply)
    assert result["action"] == "confirm_check"
    assert result["check_id"] == "need_before_access"
    assert "separate cash account" in result["context_details"]["funding_when_need_precedes_access"]
    assert engine.state.profile_version == version
    assert "need_before_access" in engine.state.active_holds
    assert engine.state.answers["D6"].selected_option_id == "D6_SOME"
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert engine.save()["saved"] is False
    assert engine.confirm_material_check(True)["action"] == "material_check_resolved"
    assert "need_before_access" not in engine.state.active_holds
    assert engine.state.answers["D6"].selected_option_id == "D6_SOME"
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"


@pytest.mark.parametrize("reply", [
    "I have not identified a separate cash account that I can use on the payment date, and it covers that payment in full.",
    "I have now identified a separate cash account that I cannot use on the payment date, and it covers that payment in full.",
    "I have now identified a separate cash account that I can use only after the payment date, and it covers that payment in full.",
    "I have now identified a separate cash account that I can use on the payment date, and it covers that payment in part.",
    "I have now identified a separate cash account that I can use on the payment date, and it does not fully cover that payment.",
    "I plan to open a separate cash account that I can use on the payment date, and it covers that payment in full.",
    "I have now identified a separate cash account that I can use on the payment date, and it will cover that payment in full once it is funded.",
    "I have now identified a separate cash account that I can use on the payment date, and it covers that payment in full, but a relative promises to fund it later.",
    "I have now identified a separate cash account that I can use on the payment date, and it covers that payment in full, but only half is actually funded.",
])
def test_cash_account_negations_partial_coverage_and_future_promises_cannot_clear_access_hold(reply):
    engine = cash_account_access_engine()
    result = engine.resolve_material_check(reply)
    assert result["action"] != "confirm_check"
    assert engine.pending_material_check is None
    assert "need_before_access" in engine.state.active_holds
    assert engine.state.answers["D6"].selected_option_id == "D6_SOME"
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert engine.save()["saved"] is False
