"""Regression checks for facts, negation, scope and saved data minimisation."""
from ai.bounded_answers import interpret_bounded_answer, supported_detail
from ai.conversation_factory import build_legacy_comparison_engine as build_conversation_engine
from ai.conversation_state import AnswerStatus
from ai.question_spec import BusinessSpec
from ai.pre_model_controls import check_pre_model_controls
from pathlib import Path
import pytest

SPEC = BusinessSpec.load(Path(__file__).parents[1] / "config/business-decision-spec.json")


@pytest.mark.parametrize("text", ["none", "None!", "none at all", "no", "nope", "nothing", "never", "no investments"])
def test_d8_whole_reply_none_means_no_personal_investment_holdings(text):
    result = interpret_bounded_answer(SPEC, "D8", text)
    assert (result.bucket, result.option_id) == ("Clarity", "D8_NONE")


@pytest.mark.parametrize(("text", "bucket"), [
    ("probably none", "Undecided"),
    ("none of these", "Confusion"),
    ("None of my friends have held shares", "Confusion"),
    ("My wife has never held investments", "Confusion"),
    ("My colleague never owned investments", "Confusion"),
    ("I own none now", "Confusion"),
    ("No investment trades in the past 12 months", "Confusion"),
    ("I have never held investments, unless my pension counts", "Confusion"),
    ("I have never personally held shares", "Confusion"),
    ("I have not owned bonds", "Confusion"),
    ("I only held bonds; I never owned shares", "Confusion"),
])
def test_d8_none_neighbours_do_not_create_false_clarity(text, bucket):
    result = interpret_bounded_answer(SPEC, "D8", text)
    assert result.bucket == bucket
    assert result.option_id is None


@pytest.mark.parametrize(("text", "expected"), [
    ("none except bonds", "D8_BONDS"),
    ("I have no investing knowledge, but I personally held an ETF", "D8_FUNDS_ETFS"),
    ("I personally held bonds. Correction: none", "D8_NONE"),
])
def test_d8_exceptions_knowledge_and_corrections_keep_supported_meaning(text, expected):
    assert interpret_bounded_answer(SPEC, "D8", text).option_id == expected


@pytest.mark.parametrize(("qid", "text"), [
    ("D1", "All of this pot is solely for buying a house. All of this pot is solely for retirement."),
    ("D4", "Every penny is in a fixed-term deposit. Every penny is in my current account. I am not describing separate parts."),
    ("D5", "Salary is my only regular source. Pension income is my only regular source."),
    ("D2", "I worked in a cafe for nine years."),
    ("D2", "My current age is 49. My current age is also 50. Neither is a correction."),
    ("D6", "Other resources cover all costs for twelve months. I have no other usable resources to cover those costs."),
    ("D8", "I watch investing videos and my cousin owns shares."),
    ("D8", "I have never personally held investments. I personally held bonds last year. Both claims stand."),
    ("D9", "I sent money to my landlord six times this year."),
    ("D10", "My investments are currently worth GBP 8,000."),
])
def test_fresh_subject_scope_and_exclusive_conflicts_never_propose_a_financial_meaning(qid, text):
    assert interpret_bounded_answer(SPEC, qid, text).option_id is None


@pytest.mark.parametrize(("qid", "text", "expected"), [
    ("D2", "I am 50 years old.", "D2_AGE50_64"),
    ("D2", "I am exactly 18.", "D2_AGE18_34"),
    ("D5", "Benefits and regular support from my parents pay rent. No wages.", "D5_BENEFITS_SUPPORT"),
    ("D6", "Without this pot, my wages cover every essential and required repayment for twelve months. Nothing falls short.", "D6_ALL"),
    ("D6", "If the reserve were unavailable for all twelve months, no other resource could pay any necessary costs or required repayments.", "D6_NONE"),
    ("D6", "Without this pot for the whole year, my wages cover some necessary costs but not all.", "D6_SOME"),
    ("D5", "My living costs are paid by withdrawing from this named reserve each month.", "D5_WITHDRAWALS"),
    ("D7", "Required repayments are up to date and comfortably within budget. Paying them is not difficult.", "D7_KEEPING_UP_EASILY"),
    ("D8", "I personally held corporate bonds last year.", "D8_BONDS"),
    ("D9", "Twelve monthly fund purchases happened under my recurring order, one each month. Bank transfers were separate and I am not counting those again.", "D9_COUNT11_PLUS"),
    ("D10", "GBP 8,000 of new outside money was invested in the past 12 months.", "D10_FROM5000_LT10000"),
    ("D11", "Some or all of the money could be gone for good, and getting it back is not guaranteed.", "D11_LOSS_POSSIBLE"),
    ("D11", "A market investment might permanently lose some or all of what I put in, and I cannot rely on getting it back.", "D11_LOSS_POSSIBLE"),
    ("D13", "Essentials remain paid using the same funding. My budget and other plans would stay the same.", "D13_UNAFFECTED"),
    ("D14", "Growth potential matters most; I accept bigger ups and downs and possible permanent loss. Greater growth is not guaranteed.", "D14_GREATER"),
])
def test_neighbouring_supported_facts_keep_their_meaning(qid, text, expected):
    assert interpret_bounded_answer(SPEC, qid, text).option_id == expected


@pytest.mark.parametrize("text", [
    "im gonna jump off the building",
    "I'm going to jump off the roof.",
    "I am about to jump from the bridge.",
    "I will jump out of the window.",
])
def test_explicit_first_person_height_danger_stops_before_financial_mapping(text):
    assert check_pre_model_controls(text).action == "safety_stop"


@pytest.mark.parametrize("text", [
    "I'm gonna jump on the bus to get to work.",
    "I will not jump off the building.",
])
def test_other_or_explicitly_negated_jumping_is_not_an_urgent_intent_match(text):
    assert check_pre_model_controls(text).action != "safety_stop"


@pytest.mark.parametrize(("qid", "text"), [
    ("D11", "I could permanently lose all the capital, but the original amount cannot fall."),
    ("D4", "I used to hold shares."),
    ("D5", "I will receive a salary after I start a job next month."),
    ("D8", "I will buy shares."),
    ("D15", "I am not comfortable."),
    ("D13", "I wouldn't struggle to pay rent, but I haven't worked out my other plans."),
])
def test_unsupported_or_conflicting_facts_do_not_create_candidates(qid, text):
    assert interpret_bounded_answer(SPEC, qid, text).option_id is None


def test_scope_correction_invalidates_old_pot_context_and_save_permissions():
    engine = build_conversation_engine()
    for qid in engine.state.question_order:
        record = engine.state.answers[qid]
        record.status = AnswerStatus.CONFIRMED
        record.selected_option_id = SPEC.question(qid).options[0].id
    engine.state.current_index = 15
    engine.state.final_accuracy_version = engine.state.profile_version
    engine.state.save_consent_version = engine.state.profile_version
    response = engine.correct_answer("D1")
    assert {"D1", "D3", "D4", "D5", "D6", "D7", "D12", "D13", "D14", "D15"} <= set(response["invalidated_questions"])
    assert engine.state.final_accuracy_version is None
    assert engine.state.save_consent_version is None
    assert not engine.save()["saved"]


def test_context_detail_excludes_amount_and_distress():
    detail = supported_detail("This ten thousand pounds pot is for a house, but I feel panicking")
    assert "ten thousand" not in detail.lower()
    assert "panicking" not in detail.lower()
    assert "house" in detail.lower()


@pytest.mark.parametrize(("text", "expected"), [
    ("A 20% loss would leave essential costs paid and would not change my budget or other plans.", "D13_UNAFFECTED"),
    ("I could still pay the essential costs, and my budget and other plans wouldn't change.", "D13_UNAFFECTED"),
    ("I would not cover essential costs after a 20% loss, although other plans would not change.", "D13_ESSENTIALS_DIFFICULT"),
    ("I can still cover rent; my budget won't change, but I would postpone the holiday.", "D13_ADJUSTMENT"),
])
def test_practical_effect_negation_does_not_reverse_financial_meaning(text, expected):
    assert interpret_bounded_answer(SPEC, "D13", text).option_id == expected


@pytest.mark.parametrize(("qid", "text"), [
    ("D3", "Can you book a flight?"),
    ("D6", "Which phone charger should I buy?"),
    ("D9", "Tell me tomorrow’s weather."),
    ("D10", "Translate hello into French."),
    ("D11", "Tell me about dinosaurs."),
    ("D13", "Recommend a restaurant."),
    ("D15", "Help me plan a holiday."),
])
def test_unrelated_requests_are_confusion_without_a_financial_candidate(qid, text):
    result = interpret_bounded_answer(SPEC, qid, text)
    assert result.bucket == "Confusion"
    assert result.option_id is None
    assert result.reason == "off_topic"


@pytest.mark.parametrize(("qid", "text"), [
    ("D11", "I am still not sure what permanent loss means."),
    ("D11", "I do not understand what permanently lost means"),
    ("D3", "I need a bit of help with this one."),
    ("D2", "Are you asking my actual age or when I plan to retire?"),
    ("D6", "Can you help me with this question?"),
    ("D6", "What counts as necessary living costs? Do you mean holidays too?"),
    ("D9", "Can you help me with the count?"),
    ("D10", "Can you help me answer this money question?"),
    ("D15", "Do you mean my bills are covered in this example? I do not understand how this differs from the earlier question."),
])
def test_explanation_requests_do_not_consume_financial_clarification_trials(qid, text):
    engine = build_conversation_engine()
    engine.state.current_index = engine.state.question_order.index(qid)
    result = engine.submit_answer(text)
    assert result["action"] == "explain"
    assert result["bucket"] == "Confusion"
    assert engine.state.answers[qid].clarification_turns == 0
    assert engine.state.answers[qid].selected_option_id is None
    assert "help_request" in engine.state.answers[qid].independent_flags


@pytest.mark.parametrize(("qid", "text", "expected"), [
    ("D1", "It is for the future, really.", "Undecided"),
    ("D2", "I am retired.", "Confusion"),
    ("D5", "I get money here and there.", "Undecided"),
    ("D7", "I owe about £30,000.", "Confusion"),
    ("D8", "My dad has investments and I help him.", "Undecided"),
    ("D9", "My pension manager must have made lots of trades; I did not instruct or know them.", "Confusion"),
    ("D6", "My backup is the same pot A, so I should be covered.", "Confusion"),
])
def test_uncertain_financial_facts_and_wrong_question_premises_remain_distinct(qid, text, expected):
    result = interpret_bounded_answer(SPEC, qid, text)
    assert result.bucket == expected
    assert result.option_id is None


def test_backup_explicitly_excluding_same_pot_is_not_treated_as_same_pot_dependency():
    result = interpret_bounded_answer(SPEC, "D6", "if pot off limits a whole year wages n other accessible cash cover ALL essentials repayments too none of backup is this pot")
    assert result.bucket == "Clarity"
    assert result.option_id == "D6_ALL"
    assert "same_pot_backup_conflict" not in result.flags


@pytest.mark.parametrize("text", [
    "This pot is physical cash kept at home. Please ask one thing at a time.",
    "This pot is for a home. My wife has died and I need a moment.",
    "This 10000 AUD pot is for a home. Please keep your replies to one short sentence.",
])
def test_saved_context_filters_support_and_life_event_clauses(text):
    detail = supported_detail(text).lower()
    assert "home" in detail
    assert "please" not in detail
    assert "died" not in detail
    assert "10000" not in detail
