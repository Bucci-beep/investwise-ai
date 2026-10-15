"""Focused Streamlit checks for the chat presentation and engine gates."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ai.conversation_factory import build_legacy_comparison_engine as build_conversation_engine
from ai.conversation_engine import ConversationEngine
from ai.conversation_state import AnswerStatus


APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def open_app(engine=None):
    app = AppTest.from_file(str(APP))
    app.session_state["iw_engine"] = engine if engine is not None else build_conversation_engine()
    app.run(timeout=30)
    assert not app.exception
    return app


def send(app, text):
    app.chat_input(key="iw_chat_input").set_value(text).run(timeout=30)
    assert not app.exception
    return app


def click(app, label):
    next(button for button in app.button if button.label == label).click().run(timeout=30)
    assert not app.exception
    return app


def completed_engine():
    engine = build_conversation_engine()
    while engine.current_question is not None:
        option = engine.current_question.options[0]
        result = engine.submit_answer(option.id)
        assert result["action"] == "confirm"
        engine.confirm_current(True)
    assert engine.completion_report()["eligible_for_final_accuracy_confirmation"]
    return engine


def test_rule_refresh_keeps_prior_answers_and_requires_confirmation_for_d8_none():
    engine = build_conversation_engine()
    while engine.state.current_question_id != "D8":
        assert engine.submit_answer(engine.current_question.options[0].id)["action"] == "confirm"
        engine.confirm_current(True)
    previous_answers = {qid: record.selected_option_id for qid, record in engine.state.answers.items()}
    state, model, store = engine.state, engine.d12_resolver, engine.store
    class PriorLoadedEngine(ConversationEngine):
        pass
    engine.__class__ = PriorLoadedEngine
    app = open_app(engine)
    current = app.session_state["iw_engine"]
    assert type(current) is ConversationEngine
    assert current.state is state
    assert current.d12_resolver is model
    assert current.store is store
    assert previous_answers == {qid: record.selected_option_id for qid, record in current.state.answers.items()}
    send(app, "none")
    record = current.state.answers["D8"]
    assert app.session_state["iw_last_response"]["action"] == "confirm"
    assert record.candidate_option_id == "D8_NONE"
    assert record.selected_option_id is None
    assert current.state.current_question_id == "D8"
    send(app, "yes")
    assert record.selected_option_id == "D8_NONE"
    assert current.state.current_question_id == "D9"


def test_explicit_own_word_summary_accuracy_is_separate_from_save_consent():
    app = open_app(completed_engine())
    send(app, "Yes, this exact summary is accurate.")
    engine = app.session_state["iw_engine"]
    assert engine.state.final_accuracy_version == engine.state.profile_version
    assert engine.state.save_consent_version is None
    assert engine.state.saved_version is None
    send(app, "No, leave it unsaved.")
    assert app.session_state["iw_unsaved"]
    assert engine.state.saved_version is None


def test_qualified_summary_agreement_cannot_grant_accuracy_or_save():
    app = open_app(completed_engine())
    send(app, "Yes, this exact summary is accurate, except the age may be wrong.")
    engine = app.session_state["iw_engine"]
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None


def test_named_own_word_correction_refreshes_summary_and_revokes_consent():
    app = open_app(completed_engine())
    send(app, "Yes, this exact summary is accurate.")
    send(app, "Change D2.")
    engine = app.session_state["iw_engine"]
    assert engine.state.current_question_id == "D2"
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    send(app, "I am sixty-four years old now.")
    send(app, "Yes, that is right.")
    assert engine.state.answers['D2'].selected_option_id == 'D2_AGE50_64'
    latest_summary = [m['text'] for m in app.session_state['iw_messages'] if 'Please review all the answers' in m['text']][-1]
    assert '50–64' in latest_summary or '50-64' in latest_summary
    assert engine.state.final_accuracy_version is None


def engine_at_d12():
    engine = build_conversation_engine()
    while engine.state.current_question_id != "D12":
        option = engine.current_question.options[0]
        assert engine.submit_answer(option.id)["action"] == "confirm"
        engine.confirm_current(True)
    return engine


def test_actual_dependency_typed_at_review_invalidates_accuracy_in_ui():
    app = open_app(completed_engine())
    send(app, "yes")
    engine = app.session_state["iw_engine"]
    assert engine.state.final_accuracy_version == engine.state.profile_version
    send(app, "In reality I need this same money for rent.")
    assert "same_money_dependency" in engine.state.active_holds
    assert engine.state.final_accuracy_version != engine.state.profile_version
    assert engine.save()["saved"] is False
    assert any("new detail" in message["text"] for message in app.session_state["iw_messages"])


def test_accessibility_request_at_review_pauses_before_accuracy_confirmation():
    app = open_app(completed_engine())
    send(app, "Please use larger text.")
    engine = app.session_state["iw_engine"]
    assert engine.state.paused
    assert app.session_state["iw_large_text"]
    assert engine.state.final_accuracy_version != engine.state.profile_version


def engine_at_d11():
    engine = build_conversation_engine()
    while engine.state.current_question_id != "D11":
        option = engine.current_question.options[0]
        assert engine.submit_answer(option.id)["action"] == "confirm"
        engine.confirm_current(True)
    return engine


def test_free_text_is_primary_and_transcript_persists_help_exchange():
    app = open_app()
    engine = app.session_state["iw_engine"]
    assert app.chat_input(key="iw_chat_input")
    assert engine.state.current_question_id == "D1"
    assert "What is the money we are discussing for?" in app.chat_message[-1].markdown[0].value

    send(app, "I need help")
    assert app.session_state["iw_last_response"]["action"] == "explain"
    assert engine.state.answers["D1"].clarification_turns == 0
    assert engine.state.answers["D1"].selected_option_id is None

    click(app, "Explain the question")
    roles = [message.name for message in app.chat_message]
    assert roles[-4:] == ["user", "assistant", "user", "assistant"]
    assert "I need help" in app.chat_message[-4].markdown[0].value
    assert "use this money for" in app.chat_message[-1].markdown[0].value
    assert engine.state.answers["D1"].clarification_turns == 0


def test_optional_fixed_choice_is_only_a_candidate_until_typed_yes():
    app = open_app()
    engine = app.session_state["iw_engine"]

    app.button(key="iw_option_D1_D1_WEALTH").click().run(timeout=30)
    assert not app.exception
    record = engine.state.answers["D1"]
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D1_WEALTH"
    assert record.selected_option_id is None
    assert engine.state.current_question_id == "D1"
    assert "Let me check" in app.chat_message[-1].markdown[0].value

    send(app, "yes")
    assert record.status == AnswerStatus.CONFIRMED
    assert record.selected_option_id == "D1_WEALTH"
    assert engine.state.current_question_id == "D2"
    assert [message.name for message in app.chat_message[-3:]] == [
        "user", "assistant", "assistant"
    ]


def test_typed_no_rejects_candidate_and_keeps_question_open():
    app = open_app()
    engine = app.session_state["iw_engine"]
    app.button(key="iw_option_D1_D1_WEALTH").click().run(timeout=30)

    send(app, "It's a little bit different")
    record = engine.state.answers["D1"]
    assert record.status == AnswerStatus.PENDING
    assert record.candidate_option_id is None
    assert record.selected_option_id is None
    assert engine.state.current_question_id == "D1"

    send(app, "Retirement")
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D1_RETIREMENT"
    assert record.selected_option_id is None


def test_d12_natural_free_text_uses_existing_candidate_confirmation_path():
    engine = engine_at_d12()
    app = open_app(engine)

    send(app, "I may realistically need this money in 7 years")
    record = engine.state.answers["D12"]
    assert app.session_state["iw_last_response"]["action"] == "confirm"
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D12_MEDIUM"
    assert record.selected_option_id is None
    assert record.source == "d12_ml"

    send(app, "yes")
    assert record.status == AnswerStatus.CONFIRMED
    assert record.selected_option_id == "D12_MEDIUM"
    assert engine.state.current_question_id == "D13"


def test_confirmed_pot_detail_and_currency_survive_final_playback():
    engine = build_conversation_engine()
    first = engine.submit_answer("My rainy-day savings pot is for emergencies")
    assert first["action"] == "confirm"
    assert "rainy-day savings pot" in first["context_details"]["purpose_detail"]
    engine.confirm_current(True)
    assert engine.submit_answer(engine.current_question.options[0].id)["action"] == "confirm"
    engine.confirm_current(True)

    amount = engine.submit_answer("About £10,000 GBP in that pot")
    assert amount["action"] == "confirm"
    assert amount["context_details"]["currency"] == "GBP"
    engine.confirm_current(True)
    while engine.current_question is not None:
        assert engine.submit_answer(engine.current_question.options[0].id)["action"] == "confirm"
        engine.confirm_current(True)

    app = open_app(engine)
    summary = next(
        message.markdown[0].value
        for message in app.chat_message
        if "**Your answers**" in message.markdown[0].value
    )
    assert "rainy-day savings pot" in summary
    assert "currency: GBP" in summary
    assert "does not establish the maximum loss you could afford" in summary
    assert "D1 ·" in summary and "D15 ·" in summary
    assert summary.count("(confirmed):**") == 15
    assert engine.final_playback()["answers"][2]["context_details"]["currency"] == "GBP"


def test_currency_followup_preserves_range_without_counting_help_as_an_attempt():
    engine = build_conversation_engine()
    for _ in range(2):
        assert engine.submit_answer(engine.current_question.options[0].id)["action"] == "confirm"
        engine.confirm_current(True)
    app = open_app(engine)

    send(app, "5k")
    record = engine.state.answers["D3"]
    assert record.status == AnswerStatus.UNDECIDED
    assert record.clarification_turns == 1
    assert record.selected_option_id is None
    assert "Which currency" in app.chat_message[-1].markdown[0].value

    send(app, "I need help")
    assert app.session_state["iw_last_response"]["action"] == "explain"
    assert record.clarification_turns == 1
    click(app, "Explain the question")
    assert record.clarification_turns == 1

    send(app, "GBP")
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D3_FROM5000_LT20000"
    assert record.selected_option_id is None
    assert record.context_details["currency"] == "GBP"
    assert record.clarification_turns == 1
    send(app, "yes")
    assert record.status == AnswerStatus.CONFIRMED
    assert engine.state.current_question_id == "D4"


def test_saved_context_excludes_distress_and_exact_amount():
    engine = build_conversation_engine()
    result = engine.submit_answer("I want to buy a home, but I am panicking")
    assert result["action"] == "care_pause"
    assert engine.resume()["action"] == "resume"
    engine.confirm_current(True)
    assert engine.submit_answer(engine.current_question.options[0].id)["action"] == "confirm"
    engine.confirm_current(True)
    result = engine.submit_answer("About £10,000 GBP in that pot")
    assert result["action"] == "confirm"
    engine.confirm_current(True)
    while engine.current_question is not None:
        assert engine.submit_answer(engine.current_question.options[0].id)["action"] == "confirm"
        engine.confirm_current(True)

    app = open_app(engine)
    send(app, "yes")  # Accuracy only.
    assert engine.store.saved_profiles == []
    send(app, "yes")  # Separate save consent.
    assert len(engine.store.saved_profiles) == 1
    saved = json.dumps(engine.store.saved_profiles[0], ensure_ascii=False).lower()
    assert "panicking" not in saved
    assert "£10,000" not in saved
    assert '"currency": "gbp"' in saved
    assert "i want to buy a home" in saved


def test_pause_and_typed_resume_preserve_candidate_and_place():
    app = open_app()
    engine = app.session_state["iw_engine"]
    app.button(key="iw_option_D1_D1_WEALTH").click().run(timeout=30)
    record = engine.state.answers["D1"]

    send(app, "pause")
    assert engine.state.paused
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D1_WEALTH"
    assert engine.state.current_question_id == "D1"
    assert engine.state.final_accuracy_version is None

    send(app, "resume")
    assert not engine.state.paused
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D1_WEALTH"
    assert engine.state.current_question_id == "D1"
    assert any(button.label == "Yes, that’s right" for button in app.button)


def test_distress_care_choice_preserves_candidate_until_user_continues():
    app = open_app()
    engine = app.session_state["iw_engine"]
    send(app, "I want to buy a home, but I am panicking")

    record = engine.state.answers["D1"]
    assert app.session_state["iw_last_response"]["action"] == "care_pause"
    assert engine.state.paused
    assert "care_pacing" in engine.state.active_holds
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D1_HOME"
    assert record.selected_option_id is None
    assert record.clarification_turns == 0
    assert "take this at your pace" in app.chat_message[-1].markdown[0].value

    click(app, "Explain the current question")
    assert engine.state.paused
    assert record.selected_option_id is None
    click(app, "Continue when ready")
    assert not engine.state.paused
    assert "care_pacing" not in engine.state.active_holds
    assert record.status == AnswerStatus.CANDIDATE
    click(app, "Yes, that’s right")
    assert record.status == AnswerStatus.CONFIRMED
    assert engine.state.current_question_id == "D2"


def test_user_can_stop_from_care_state_without_fake_handoff():
    app = open_app()
    engine = app.session_state["iw_engine"]
    send(app, "I am panicking")
    click(app, "Stop this conversation")

    assert "user_stop" in engine.state.active_holds
    assert engine.state.answers["D1"].selected_option_id is None
    assert engine.state.saved_version is None
    assert len(app.chat_input) == 0
    assert "No further financial questions" in app.chat_message[-1].markdown[0].value


def test_clear_answer_with_help_waits_for_explanation_then_confirmation():
    app = open_app()
    engine = app.session_state["iw_engine"]
    send(app, "I want to buy a home, please explain the choices")

    record = engine.state.answers["D1"]
    assert app.session_state["iw_last_response"]["action"] == "support_before_confirmation"
    assert engine.state.paused
    assert "support_request" in engine.state.active_holds
    assert record.candidate_option_id == "D1_HOME"
    assert record.selected_option_id is None

    click(app, "Explain the current question")
    assert engine.state.paused
    click(app, "Continue when ready")
    assert not engine.state.paused
    assert record.status == AnswerStatus.CANDIDATE
    assert record.selected_option_id is None
    send(app, "yes")
    assert record.selected_option_id == "D1_HOME"
    assert engine.state.current_question_id == "D2"


def test_d11_misconception_requires_fresh_own_words_before_progress():
    engine = engine_at_d11()
    app = open_app(engine)
    app.button(key="iw_option_D11_D11_RECOVERY_ALWAYS").click().run(timeout=30)
    assert not app.exception
    assert engine.state.answers["D11"].status == AnswerStatus.CANDIDATE
    send(app, "yes")

    record = engine.state.answers["D11"]
    assert app.session_state["iw_last_response"]["action"] == "understanding_hold"
    assert "understanding_hold" in engine.state.active_holds
    assert record.selected_option_id is None
    assert engine.state.current_question_id == "D11"
    assert "In your own words" in app.chat_message[-1].markdown[0].value
    assert not any(button.key and button.key.startswith("iw_option_D11") for button in app.button)

    send(app, "Some or all of the original money could be permanently lost and recovery is not guaranteed")
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D11_LOSS_POSSIBLE"
    assert record.selected_option_id is None
    send(app, "yes")
    assert record.selected_option_id == "D11_LOSS_POSSIBLE"
    assert "understanding_hold" not in engine.state.active_holds
    assert engine.state.current_question_id == "D12"


def test_final_accuracy_and_save_consent_are_separate_chat_turns():
    engine = completed_engine()
    app = open_app(engine)
    assert len(engine.final_playback()["answers"]) == 15
    assert engine.state.saved_version is None

    send(app, "yes")
    version = engine.state.profile_version
    assert engine.state.final_accuracy_version == version
    assert engine.state.save_consent_version is None
    assert engine.state.saved_version is None
    assert engine.store.saved_profiles == []
    assert "Confirming accuracy alone has not saved it" in app.chat_message[-1].markdown[0].value

    send(app, "yes")
    assert engine.state.save_consent_version == version
    assert engine.state.saved_version == version
    assert len(engine.store.saved_profiles) == 1
    assert "saved in this demo session" in app.chat_message[-1].markdown[0].value


def test_final_review_correction_invalidates_previous_summary():
    engine = completed_engine()
    old_version = engine.state.profile_version
    app = open_app(engine)

    send(app, "It's a little bit different")
    assert app.session_state["iw_needs_correction"]
    assert engine.state.final_accuracy_version is None
    assert engine.state.saved_version is None

    send(app, "question 12")
    assert engine.state.current_question_id == "D12"
    assert engine.state.answers["D12"].status == AnswerStatus.PENDING
    assert engine.state.answers["D12"].selected_option_id is None
    assert engine.state.profile_version > old_version
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None


@pytest.mark.parametrize("reply", ["It's a little different", "change my answer"])
def test_typed_correction_after_accuracy_cancels_save_stage(reply):
    engine = completed_engine()
    app = open_app(engine)
    send(app, "yes")
    assert engine.state.final_accuracy_version == engine.state.profile_version

    send(app, reply)
    assert app.session_state["iw_needs_correction"]
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    assert engine.state.saved_version is None

    send(app, "question 12")
    assert engine.state.current_question_id == "D12"
    assert engine.state.answers["D12"].selected_option_id is None
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None


@pytest.mark.parametrize("text", ["goodbye world", "im gonna jump off the building"])
@pytest.mark.parametrize("at_review", [False, True])
def test_urgent_safety_language_stops_chat_and_saving(text, at_review):
    app = open_app(completed_engine() if at_review else None)
    engine = app.session_state["iw_engine"]
    if at_review:
        send(app, "yes")
    send(app, text)

    assert engine.state.stopped_for_safety
    assert app.session_state["iw_last_response"]["action"] == "safety_stop"
    if not at_review:
        assert engine.state.answers["D1"].selected_option_id is None
    assert engine.state.saved_version is None
    assert engine.give_save_consent(True)["accepted"] is False
    assert engine.save()["saved"] is False
    assert engine.resume()["action"] == "safety_stop"
    assert len(app.chat_input) == 0
    assert "No live agent has been contacted" in app.chat_message[-1].markdown[0].value


def test_three_clarification_attempts_remain_capped_after_pause_and_resume():
    app = open_app()
    engine = app.session_state["iw_engine"]
    for text in ("The sky is blue", "Today is Saturday", "I like oranges"):
        send(app, text)

    record = engine.state.answers["D1"]
    assert record.clarification_turns == 3
    assert app.session_state["iw_last_response"]["action"] == "finish_incomplete_or_pause"
    assert record.selected_option_id is None
    assert len(app.chat_input) == 0

    click(app, "Pause conversation")
    assert engine.state.paused
    click(app, "Resume conversation")
    assert not engine.state.paused
    assert record.clarification_turns == 3
    assert record.selected_option_id is None
    assert len(app.chat_input) == 0
    assert all(button.label != "Help" for button in app.button)


def test_optional_question_can_be_declined_after_clarification_cap():
    engine = build_conversation_engine()
    assert engine.submit_answer("D1_WEALTH")["action"] == "confirm"
    engine.confirm_current(True)
    assert engine.state.current_question_id == "D2"
    app = open_app(engine)

    for text in ("The sky is blue", "Today is Saturday", "I like oranges"):
        send(app, text)
    record = engine.state.answers["D2"]
    assert record.clarification_turns == 3
    assert record.selected_option_id is None
    assert len(app.chat_input) == 0

    click(app, "Leave this contextual answer unanswered")
    assert record.status == AnswerStatus.DECLINED
    assert record.selected_option_id is None
    assert engine.state.current_question_id == "D3"


def access_conflict_engine():
    engine = build_conversation_engine()
    replies = {
        "D1": "D1_PURCHASE", "D3": "About 10,000 GBP in this pot",
        "D4": "All this money is in a fixed-term deposit. I cannot access it for five years.",
        "D12": "The earliest realistic need is in two years, for a tuition payment.",
        "D13": "D13_ADJUSTMENT", "D14": "D14_BALANCED",
        "D15": "D15_WORRIED_ACCEPTABLE",
    }
    while engine.current_question:
        question = engine.current_question
        result = engine.submit_answer(replies.get(question.id, question.options[0].id))
        assert result["action"] == "confirm", result
        engine.confirm_current(True)
    assert "need_before_access" in engine.state.active_holds
    return engine


def test_material_check_requires_fact_confirmation_then_new_accuracy_and_consent():
    engine = access_conflict_engine()
    app = open_app(engine)
    assert "how would it be met" in app.chat_message[-1].markdown[0].value
    assert not engine.completion_report()["eligible_for_final_accuracy_confirmation"]
    old_version = engine.state.profile_version

    send(app, "My separate savings are available when the payment is due and cover the payment")
    assert app.session_state["iw_last_response"]["action"] == "confirm_check"
    assert "need_before_access" in engine.state.active_holds
    assert engine.state.answers["D12"].selected_option_id == "D12_SHORT"
    assert engine.state.final_accuracy_version is None

    send(app, "yes")
    assert "need_before_access" not in engine.state.active_holds
    assert engine.state.profile_version > old_version
    assert engine.state.final_accuracy_version is None
    summary = next(message.markdown[0].value for message in reversed(app.chat_message)
                   if "**Your answers**" in message.markdown[0].value)
    assert "funding before access" in summary
    assert "separate savings" in summary
    assert "access timing" in summary
    assert "access.available_after_years" not in summary
    assert engine.store.saved_profiles == []

    send(app, "yes")
    assert engine.state.final_accuracy_version == engine.state.profile_version
    assert engine.store.saved_profiles == []
    send(app, "yes")
    assert len(engine.store.saved_profiles) == 1
    saved = json.dumps(engine.store.saved_profiles[0])
    assert "funding_when_need_precedes_access" in saved
    assert "material.need_before_access.resolved" not in saved
    assert "access.available_after_years" not in saved


def test_final_material_clarification_cap_survives_pause_and_resume():
    engine = access_conflict_engine()
    app = open_app(engine)
    for reply in ("Maybe my savings", "I don't know when they are available", "Probably my salary"):
        send(app, reply)
    assert engine.state.answers["D12"].clarification_turns == 3
    assert app.session_state["iw_last_response"]["action"] == "finish_incomplete_or_pause"
    assert len(app.chat_input) == 0
    click(app, "Pause conversation")
    click(app, "Resume conversation")
    assert len(app.chat_input) == 0
    assert "need_before_access" in engine.state.active_holds
    assert engine.store.saved_profiles == []
    click(app, "Finish incomplete")
    assert app.session_state["iw_finished_incomplete"]
    assert len(app.chat_input) == 0


def test_generic_dependency_free_text_is_reviewed_in_latest_summary():
    engine = build_conversation_engine()
    replies = {"D1": "D1_EMERGENCIES", "D12": "The earliest realistic need is in fourteen years", "D13": "D13_UNAFFECTED"}
    while engine.current_question:
        question = engine.current_question
        result = engine.submit_answer(replies.get(question.id, question.options[0].id))
        assert result["action"] == "confirm", result
        engine.confirm_current(True)
    assert "emergency_long_horizon_dependency" in engine.state.active_holds
    app = open_app(engine)
    assert "earliest point" in app.chat_message[-1].markdown[0].value
    send(app, "The earliest realistic need for this same money is in fourteen years. A 20% loss would leave essential costs paid and would not change my budget or other plans.")
    assert app.session_state["iw_last_response"]["action"] == "confirm_check"
    assert engine.store.saved_profiles == []
    send(app, "yes")
    assert "emergency_long_horizon_dependency" not in engine.state.active_holds
    assert engine.completion_report()["eligible_for_final_accuracy_confirmation"]
    summary = next(message.markdown[0].value for message in reversed(app.chat_message)
                   if "**Your answers**" in message.markdown[0].value)
    assert "dependency check" in summary
    assert engine.state.final_accuracy_version is None


def test_larger_text_request_changes_presentation_without_financial_answer():
    app = open_app()
    engine = app.session_state["iw_engine"]
    send(app, "Please use larger text")
    assert app.session_state.iw_large_text
    assert engine.state.paused
    assert engine.state.answers["D1"].selected_option_id is None
    assert engine.state.answers["D1"].clarification_turns == 0
    assert app.session_state["iw_last_response"]["bucket"] == "Clarity"
    assert any("font-size: 1.25rem" in item.value for item in app.markdown)
    assert "increased the chat text size" in app.chat_message[-1].markdown[0].value


def test_accessibility_request_during_confirmation_preserves_candidate():
    app = open_app()
    engine = app.session_state["iw_engine"]
    send(app, "I want to buy a home")
    record = engine.state.answers["D1"]
    assert record.candidate_option_id == "D1_HOME"
    send(app, "Please use larger text")
    assert engine.state.paused
    assert record.candidate_option_id == "D1_HOME"
    assert record.selected_option_id is None
    click(app, "Continue when ready")
    assert record.candidate_option_id == "D1_HOME"
    send(app, "yes")
    assert record.selected_option_id == "D1_HOME"
    assert engine.state.current_question_id == "D2"


@pytest.mark.parametrize(("qid", "reply", "definition"), [
    ("D8", "What is an ETF?", "fund traded on an exchange"),
    ("D7", "What does required repayment mean?", "payment you must make now"),
    ("D10", "What does new outside money mean?", "fresh money you added"),
])
def test_term_questions_receive_a_neutral_definition_before_an_answer(qid, reply, definition):
    engine = build_conversation_engine()
    engine.state.current_index = engine.state.question_order.index(qid)
    app = open_app(engine)
    send(app, reply)
    assert definition in app.chat_message[-1].markdown[0].value
    assert engine.state.answers[qid].clarification_turns == 0
    assert engine.state.answers[qid].selected_option_id is None


def test_clear_amount_plus_break_pauses_before_confirmation():
    engine = build_conversation_engine()
    engine.state.current_index = engine.state.question_order.index("D10")
    app = open_app(engine)
    send(app, "I put 1,000 GBP of new outside money into investments in the past 12 months. I need a break.")
    record = engine.state.answers["D10"]
    assert engine.state.paused
    assert record.candidate_option_id == "D10_FROM1000_LT5000"
    assert record.selected_option_id is None
    assert record.clarification_turns == 0
    assert "pause_or_stop" in record.independent_flags
    assert "safety_or_distress" not in record.independent_flags
    send(app, "yes")
    assert record.selected_option_id is None
    send(app, "continue")
    assert "currency: GBP" in app.chat_message[-1].markdown[0].value
    send(app, "yes")
    assert record.selected_option_id == "D10_FROM1000_LT5000"
