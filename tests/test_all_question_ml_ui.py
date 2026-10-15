"""Real default-model UI diagnostics, separate from sealed holdout evaluation.

These checks use the installed TF-IDF/LR/conformal models and visible user
controls. They do not mock predictions, tune thresholds, inspect test.csv, or
assert statistical accuracy. A correct safe abstention is allowed for Q8 none;
an incorrect finance candidate is never allowed.
"""

from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from ai.all_question_ml import AllQuestionMLResolver
from ai.conformal import SplitConformalClassifier
from ai.conversation_factory import build_conversation_engine
from ai.conversation_state import AnswerStatus

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "app.py"

# Deliberate customer selections for a fictional flow. The optional current
# amount is declined rather than invented. No text/ML result is forced through
# when a visible positive-amount shortcut requires a separate currency fact.
VISIBLE_CHOICES = {
    "D1": "D1_WEALTH",
    "D2": "D2_AGE35_49",
    "D3": None,
    "D4": "D4_EASY_ACCESS",
    "D5": "D5_SALARY",
    "D6": "D6_ALL",
    "D7": "D7_NO_REQUIRED",
    "D8": "D8_NONE",
    "D9": "D9_ZERO",
    "D10": "D10_ZERO",
    "D11": "D11_LOSS_POSSIBLE",
    "D12": "D12_MEDIUM",
    "D13": "D13_UNAFFECTED",
    "D14": "D14_BALANCED",
    "D15": "D15_WORRIED_ACCEPTABLE",
}


def open_default_app():
    # No iw_engine injection: this exercises the production default factory.
    app = AppTest.from_file(str(APP)).run(timeout=90)
    assert not app.exception
    assert app.session_state["iw_engine"].interpretation_mode == "ml"
    return app


def engine(app):
    return app.session_state["iw_engine"]


def send(app, text):
    app.chat_input(key="iw_chat_input").set_value(text).run(timeout=30)
    assert not app.exception


def click_label(app, label):
    buttons = [button for button in app.button if button.label == label]
    assert len(buttons) == 1, (label, [button.label for button in app.button])
    buttons[0].click().run(timeout=30)
    assert not app.exception


def choose_visible(app, qid, option_id):
    assert engine(app).state.current_question_id == qid
    if option_id is None:
        click_label(app, "Skip")
        record = engine(app).state.answers[qid]
        assert record.status == AnswerStatus.DECLINED
        assert record.selected_option_id is None
        return
    app.button(key=f"iw_option_{qid}_{option_id}").click().run(timeout=30)
    assert not app.exception
    record = engine(app).state.answers[qid]
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == option_id
    assert record.selected_option_id is None
    assert record.source == "explicit_user_choice"
    assert engine(app).state.current_question_id == qid
    click_label(app, "Yes, that’s right")
    assert engine(app).state.answers[qid].selected_option_id == option_id


def advance_visibly(app, target=None):
    while engine(app).current_question is not None:
        qid = engine(app).state.current_question_id
        if qid == target:
            break
        choose_visible(app, qid, VISIBLE_CHOICES[qid])


def latest_ml_event(app, qid):
    matches = [event for event in engine(app).state.audit
               if event.event_type == "ml_interpretation" and event.question_id == qid]
    assert matches, f"No real ML interpretation was audited for {qid}"
    event = matches[-1]
    assert event.payload["source"] == "all_question_ml"
    return event


def assert_no_rule_proposal_for_last_free_text(app, qid):
    record = engine(app).state.answers[qid]
    if record.candidate_option_id is not None:
        assert record.source == "all_question_ml"
    response = app.session_state["iw_last_response"]
    assert response["ml_evidence"]["source"] == "all_question_ml"
    assert response["ml_evidence"]["intent_set"] == latest_ml_event(app, qid).payload["intent_set"]
    return response


def test_default_factory_has_fitted_real_models_for_all_fifteen_questions():
    current = build_conversation_engine()
    assert current.interpretation_mode == "ml"
    assert isinstance(current.question_resolver, AllQuestionMLResolver)
    assert current.d12_resolver is None
    expected_questions = {f"D{number}" for number in range(1, 16)}
    assert set(current.question_resolver.predictors) == expected_questions
    assert set(current.question_resolver.allowed_options) == expected_questions
    assert current.question_resolver.source_hash
    for qid, pair in current.question_resolver.predictors.items():
        assert isinstance(pair.intent, SplitConformalClassifier)
        assert isinstance(pair.options, SplitConformalClassifier)
        assert set(pair.intent.classifier.classes) == {"clarity", "undecided", "confusion"}
        assert set(pair.options.classifier.classes) == {
            option.id for option in current.spec.question(qid).options
        }
        for predictor in (pair.intent, pair.options):
            assert 0 <= predictor.threshold <= 1
            pipeline = predictor.classifier._pipeline
            assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
            assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
            assert pipeline.named_steps["tfidf"].vocabulary_
            assert pipeline.named_steps["classifier"].coef_.size > 0


def test_currency_only_followup_completes_retained_amount_without_ml():
    app = open_default_app()
    advance_visibly(app, "D3")

    option_id = "D3_FROM5000_LT20000"
    app.button(key=f"iw_option_D3_{option_id}").click().run(timeout=30)
    assert not app.exception
    assert engine(app).state.answers["D3"].context_details["pending_option_id"] == option_id

    send(app, "pounds sterling")
    record = engine(app).state.answers["D3"]
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == option_id
    assert record.context_details["currency"] == "GBP"
    assert record.source == "bounded_material_completion"
    assert app.session_state["iw_last_response"]["bucket"] == "Clarity"


def test_amount_and_currency_are_accepted_as_bounded_facts_before_ml():
    app = open_default_app()
    advance_visibly(app, "D3")

    send(app, "About £30,000 GBP.")
    record = engine(app).state.answers["D3"]
    assert record.status == AnswerStatus.CANDIDATE
    assert record.candidate_option_id == "D3_FROM20000_LT50000"
    assert record.context_details["currency"] == "GBP"
    assert record.source == "bounded_material_completion"
    assert app.session_state["iw_last_response"]["bucket"] == "Clarity"


def test_default_first_free_text_turn_uses_actual_ml_even_for_an_exact_label():
    app = open_default_app()
    qid = engine(app).state.current_question_id
    assert qid == "D1"
    # An exact label used to bypass ML in the archived hybrid. Typed text in
    # the default app must now produce ML evidence or abstain through ML.
    send(app, engine(app).spec.option("D1", "D1_HOME").label)
    response = assert_no_rule_proposal_for_last_free_text(app, "D1")
    record = engine(app).state.answers["D1"]
    assert record.selected_option_id is None
    assert engine(app).state.current_question_id == "D1"
    assert record.candidate_option_id in {None, "D1_HOME"}, response
    if record.candidate_option_id is not None:
        assert response["ml_evidence"]["intent_set"] == ["clarity"]
        assert response["ml_evidence"]["option_set"] == ["D1_HOME"]


def test_typing_an_option_id_is_not_a_visible_selection_or_a_rule_fallback():
    app = open_default_app()
    send(app, "D1_HOME")
    response = assert_no_rule_proposal_for_last_free_text(app, "D1")
    assert engine(app).state.answers["D1"].selected_option_id is None
    assert engine(app).state.answers["D1"].source != "explicit_user_choice"
    assert response.get("recorded", False) is False


def test_q8_none_is_correct_pending_ml_candidate_or_safe_abstention():
    app = open_default_app()
    advance_visibly(app, "D8")
    before = {qid: engine(app).state.answers[qid].selected_option_id
              for qid in engine(app).state.question_order[:7]}
    send(app, "none")
    response = assert_no_rule_proposal_for_last_free_text(app, "D8")
    record = engine(app).state.answers["D8"]
    assert record.candidate_option_id in {None, "D8_NONE"}, response
    assert record.selected_option_id is None
    assert engine(app).state.current_question_id == "D8"
    assert before == {qid: engine(app).state.answers[qid].selected_option_id for qid in before}
    if record.candidate_option_id == "D8_NONE":
        assert response["action"] == "confirm"
        assert record.source == "all_question_ml"
        click_label(app, "Yes, that’s right")
        assert engine(app).state.answers["D8"].selected_option_id == "D8_NONE"
        assert engine(app).state.current_question_id == "D9"
    else:
        assert record.status != AnswerStatus.CANDIDATE
        assert response["action"] in {"clarify", "finish_incomplete_or_pause"}
        assert not any(button.label == "Yes, that’s right" for button in app.button)


@pytest.mark.parametrize("reply", ["probably none", "none of these choices fits"])
def test_q8_uncertainty_or_none_of_choices_never_creates_false_clarity(reply):
    app = open_default_app()
    advance_visibly(app, "D8")
    send(app, reply)
    response = assert_no_rule_proposal_for_last_free_text(app, "D8")
    record = engine(app).state.answers["D8"]
    assert record.candidate_option_id is None, response
    assert record.selected_option_id is None
    assert record.status != AnswerStatus.CANDIDATE
    assert engine(app).state.current_question_id == "D8"
    assert not any(button.label == "Yes, that’s right" for button in app.button)


def test_all_visible_choices_finish_with_separate_save_and_latest_correction():
    app = open_default_app()
    advance_visibly(app)
    current = engine(app)
    assert current.state.current_question_id is None
    assert current.completion_report()["all_15_handled"]
    assert current.completion_report()["eligible_for_final_accuracy_confirmation"]
    assert current.state.answers["D3"].status == AnswerStatus.DECLINED
    assert current.state.final_accuracy_version is None
    assert current.state.save_consent_version is None
    assert current.state.saved_version is None
    assert current.store.saved_profiles == []
    # These were actual visible customer choices, not free-text model fallback.
    assert not any(event.event_type == "ml_interpretation" for event in current.state.audit)
    assert not any(button.label == "Yes, save this version" for button in app.button)

    click_label(app, "Yes, the summary is accurate")
    current = engine(app)
    assert current.state.final_accuracy_version == current.state.profile_version
    assert current.state.save_consent_version is None
    assert current.state.saved_version is None
    assert current.store.saved_profiles == []
    send(app, "It looks right.")
    assert engine(app).state.save_consent_version is None
    assert engine(app).store.saved_profiles == []
    click_label(app, "Yes, save this version")
    current = engine(app)
    first_version = current.state.profile_version
    assert current.state.saved_version == first_version
    assert len(current.store.saved_profiles) == 1
    assert current.store.saved_profiles[0]["confirmed_answers"]["D2"]["option_id"] == "D2_AGE35_49"

    # The saved view hides chat input but retains its explicit correction list.
    app.selectbox[0].set_value("D2").run(timeout=30)
    assert not app.exception
    click_label(app, "Change this answer")
    current = engine(app)
    assert current.state.current_question_id == "D2"
    assert current.state.profile_version > first_version
    assert current.state.final_accuracy_version is None
    assert current.state.save_consent_version is None
    assert current.state.saved_version is None
    assert len(current.store.saved_profiles) == 1
    choose_visible(app, "D2", "D2_AGE65_PLUS")
    current = engine(app)
    assert current.state.current_question_id is None
    assert current.state.answers["D2"].selected_option_id == "D2_AGE65_PLUS"
    assert current.state.final_accuracy_version is None
    assert current.state.save_consent_version is None
    assert len(current.store.saved_profiles) == 1
    summaries = [message["text"] for message in app.session_state["iw_messages"]
                 if "Please review all the answers" in message["text"]]
    assert current.spec.option("D2", "D2_AGE65_PLUS").label in summaries[-1]
    assert current.spec.option("D2", "D2_AGE35_49").label not in summaries[-1]
    assert not any(button.label == "Yes, save this version" for button in app.button)

    click_label(app, "Yes, the summary is accurate")
    assert engine(app).state.save_consent_version is None
    assert len(engine(app).store.saved_profiles) == 1
    click_label(app, "Yes, save this version")
    current = engine(app)
    assert current.state.saved_version == current.state.profile_version
    assert len(current.store.saved_profiles) == 2
    assert current.store.saved_profiles[-1]["confirmed_answers"]["D2"]["option_id"] == "D2_AGE65_PLUS"
    assert current.store.saved_profiles[-1]["profile_version"] == current.state.profile_version
