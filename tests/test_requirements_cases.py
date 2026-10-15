"""Independent synthetic business probes, authored before resolver evaluation.

These cases are not training/calibration data and do not establish statistical
accuracy or coverage. Safety and semantic-coverage assertions are kept separate:
a correct abstention is not a wrong accepted financial interpretation.
"""
from __future__ import annotations

import pytest

from ai.conversation_engine import ConversationEngine
from ai.conversation_factory import build_legacy_comparison_engine as build_conversation_engine


CASES = [{'id': 'ADV-D1-CLEAR_NATURAL',
  'question_id': 'D1',
  'user_text': 'I set this pot aside to pay for my daughter’s degree and graduation costs.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D1_PURCHASE'},
 {'id': 'ADV-D2-CLEAR_NATURAL',
  'question_id': 'D2',
  'user_text': 'I turned fifty-five last month.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D2_AGE50_64'},
 {'id': 'ADV-D3-CLEAR_NATURAL',
  'question_id': 'D3',
  'user_text': 'About ten thousand pounds sterling, just the tuition pot we’re discussing.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D3_FROM5000_LT20000'},
 {'id': 'ADV-D4-CLEAR_NATURAL',
  'question_id': 'D4',
  'user_text': 'It’s in my bank’s instant-access savings account.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D4_EASY_ACCESS'},
 {'id': 'ADV-D5-CLEAR_NATURAL',
  'question_id': 'D5',
  'user_text': 'My employer pays me wages each month, and those pay my everyday bills.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D5_SALARY'},
 {'id': 'ADV-D6-CLEAR_NATURAL',
  'question_id': 'D6',
  'user_text': 'Even if this pot was off limits for that whole year, my wages would pay every necessary '
               'bill without using it.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D6_ALL'},
 {'id': 'ADV-D7-CLEAR_NATURAL',
  'question_id': 'D7',
  'user_text': 'I have no loan or credit repayments that I’m required to make now.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D7_NO_REQUIRED'},
 {'id': 'ADV-D8-CLEAR_NATURAL',
  'question_id': 'D8',
  'user_text': 'I’ve personally owned exchange-traded funds, and no other investment type.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D8_FUNDS_ETFS'},
 {'id': 'ADV-D9-CLEAR_NATURAL',
  'question_id': 'D9',
  'user_text': 'I placed four investment buy or sell orders over those twelve months.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D9_COUNT1_5'},
 {'id': 'ADV-D10-CLEAR_NATURAL',
  'question_id': 'D10',
  'user_text': 'Two thousand pounds sterling of fresh money actually went into my investments this past '
               'year; I’m excluding reinvested sale proceeds.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D10_FROM1000_LT5000'},
 {'id': 'ADV-D11-CLEAR_NATURAL',
  'question_id': 'D11',
  'user_text': 'I could permanently lose the original sum, even all of it; a later recovery isn’t '
               'promised.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D11_LOSS_POSSIBLE'},
 {'id': 'ADV-D12-CLEAR_NATURAL',
  'question_id': 'D12',
  'user_text': 'The first realistic need for any of the tuition pot is fourteen years from now. '
               'Separate cash covers emergencies and my living costs.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D12_LONG'},
 {'id': 'ADV-D13-CLEAR_NATURAL',
  'question_id': 'D13',
  'user_text': 'All essential bills would still be covered after that loss, but I would have to '
               'postpone the graduation spending plan.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D13_ADJUSTMENT'},
 {'id': 'ADV-D14-CLEAR_NATURAL',
  'question_id': 'D14',
  'user_text': 'I want a deliberate balance: some potential growth while limiting the swings, and I '
               'accept that losses are possible.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D14_BALANCED'},
 {'id': 'ADV-D15-CLEAR_NATURAL',
  'question_id': 'D15',
  'user_text': 'With the bills and plans covered as you said, I would worry, but I could accept that '
               'situation even if it never recovered.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D15_WORRIED_ACCEPTABLE'},
 {'id': 'ADV-D1-UNDECIDED',
  'question_id': 'D1',
  'user_text': 'I’m torn between using it for a home or retirement; I haven’t decided which purpose.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D2-UNDECIDED',
  'question_id': 'D2',
  'user_text': 'I only know the fictional persona is either 34 or 35; I can’t establish which.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D3-UNDECIDED',
  'question_id': 'D3',
  'user_text': 'I don’t know whether the pot is 4,800 or 5,200 GBP.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D4-UNDECIDED',
  'question_id': 'D4',
  'user_text': 'It’s bank savings, but I don’t know whether withdrawals are restricted.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D5-UNDECIDED',
  'question_id': 'D5',
  'user_text': 'I can’t tell whether regular support from my sister is still continuing.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D6-UNDECIDED',
  'question_id': 'D6',
  'user_text': 'Some bills might be covered by other money, but I haven’t worked out whether all of '
               'them would be covered for the full year.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D7-UNDECIDED',
  'question_id': 'D7',
  'user_text': 'I’m not sure whether any repayment is due on my student loan at the moment.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D8-UNDECIDED',
  'question_id': 'D8',
  'user_text': 'I haven’t found out whether the pension money was actually invested.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D9-UNDECIDED',
  'question_id': 'D9',
  'user_text': 'It was five or six instructed investment trades; I can’t remember which.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D10-UNDECIDED',
  'question_id': 'D10',
  'user_text': 'I cannot separate the new contributions from money I got by selling and buying again.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D11-UNDECIDED',
  'question_id': 'D11',
  'user_text': 'I understand the question, but I haven’t established whether permanent loss is '
               'possible.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D12-UNDECIDED',
  'question_id': 'D12',
  'user_text': 'The same pot might be needed after two years or after twelve years; I cannot determine '
               'the earlier realistic need yet.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D13-UNDECIDED',
  'question_id': 'D13',
  'user_text': 'The essentials would still be paid, but I have not figured out whether my budget or '
               'other plans would have to change.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D14-UNDECIDED',
  'question_id': 'D14',
  'user_text': 'I know what the trade-offs mean, but I still cannot choose which one I want.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D15-UNDECIDED',
  'question_id': 'D15',
  'user_text': 'I would be worried, but I cannot decide whether I could accept that situation.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D1-CONFUSION_OR_HELP',
  'question_id': 'D1',
  'user_text': 'I need help understanding what you mean by purpose.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D2-CONFUSION_OR_HELP',
  'question_id': 'D2',
  'user_text': 'I work as a teacher.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D3-CONFUSION_OR_HELP',
  'question_id': 'D3',
  'user_text': 'The dream is to have twenty thousand eventually.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D4-CONFUSION_OR_HELP',
  'question_id': 'D4',
  'user_text': 'I owe the bank a personal loan.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D5-CONFUSION_OR_HELP',
  'question_id': 'D5',
  'user_text': 'My pension fund balance is thirty thousand.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D6-CONFUSION_OR_HELP',
  'question_id': 'D6',
  'user_text': 'What does unavailable mean?',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D7-CONFUSION_OR_HELP',
  'question_id': 'D7',
  'user_text': 'The weather is lovely, isn’t it?',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D8-CONFUSION_OR_HELP',
  'question_id': 'D8',
  'user_text': 'I have read articles about ETFs but never owned one; does that count?',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D9-CONFUSION_OR_HELP',
  'question_id': 'D9',
  'user_text': 'I bought groceries four times.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D10-CONFUSION_OR_HELP',
  'question_id': 'D10',
  'user_text': 'The shares I already own are worth six thousand today.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D11-CONFUSION_OR_HELP',
  'question_id': 'D11',
  'user_text': 'I do not understand what permanently lost means.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D12-CONFUSION_OR_HELP',
  'question_id': 'D12',
  'user_text': 'My fixed deposit matures in six years.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D13-CONFUSION_OR_HELP',
  'question_id': 'D13',
  'user_text': 'Twenty percent means twenty pounds whatever the amount, right?',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D14-CONFUSION_OR_HELP',
  'question_id': 'D14',
  'user_text': 'Can you recommend which ETF I should buy?',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D15-CONFUSION_OR_HELP',
  'question_id': 'D15',
  'user_text': 'I would not be able to pay rent with my actual money.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D1-MIXED',
  'question_id': 'D1',
  'user_text': 'Some of this named pot is for a house deposit and the rest for retirement.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D1_MULTIPLE'},
 {'id': 'ADV-D2-AGE_NO_INFERENCE',
  'question_id': 'D2',
  'user_text': 'I am sixty-seven and my earliest need is fifteen years away; separate income covers my '
               'costs.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D2_AGE65_PLUS'},
 {'id': 'ADV-D3-CURRENCY_MISSING',
  'question_id': 'D3',
  'user_text': 'The current agreed pot is roughly 9,400, but I have not told you the currency.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D4-MIXED_CHATTER',
  'question_id': 'D4',
  'user_text': 'This pot is split between easy-access savings and an invested pension; the sun is '
               'bright today.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D4_MIXED'},
 {'id': 'ADV-D5-MIXED_SOURCES',
  'question_id': 'D5',
  'user_text': 'My regular living bills are paid using both wages and monthly pension payments.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D5_MULTIPLE'},
 {'id': 'ADV-D6-LATE_BACKUP',
  'question_id': 'D6',
  'user_text': 'No other resources cover the first six months. A separate deposit unlocks in month '
               'seven and would cover every necessary bill for the final six months.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D6_SOME'},
 {'id': 'ADV-D7-DEBT_NOT_REQUIRED',
  'question_id': 'D7',
  'user_text': 'I owe a student loan, but no payments are currently required.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D7_NO_REQUIRED'},
 {'id': 'ADV-D8-MIXED_EXPERIENCE',
  'question_id': 'D8',
  'user_text': 'I have personally held ETFs and cryptoassets.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D8_MULTIPLE'},
 {'id': 'ADV-D9-RECURRING_TRADES',
  'question_id': 'D9',
  'user_text': 'I arranged one ETF purchase every month over the twelve-month period, twelve purchases '
               'in total.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D9_COUNT11_PLUS'},
 {'id': 'ADV-D10-CURRENCY_KNOWN',
  'question_id': 'D10',
  'user_text': 'The fresh money actually invested was 1,500 EUR; do not convert it to pounds.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D10_FROM1000_LT5000'},
 {'id': 'ADV-D11-MISCONCEPTION',
  'question_id': 'D11',
  'user_text': 'I think falls in value always recover.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D11_RECOVERY_ALWAYS'},
 {'id': 'ADV-D11-PROTECTED_MISCONCEPTION',
  'question_id': 'D11',
  'user_text': 'I think the original amount cannot fall.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D11_CAPITAL_PROTECTED'},
 {'id': 'ADV-D12-BOUNDARY_3',
  'question_id': 'D12',
  'user_text': 'The earliest realistic need for any included money is exactly three years from now.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D12_SHORT'},
 {'id': 'ADV-D12-BOUNDARY_10',
  'question_id': 'D12',
  'user_text': 'The earliest realistic need for any included money is exactly ten years from now.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D12_MEDIUM'},
 {'id': 'ADV-D12-IDEAL_VS_NEED',
  'question_id': 'D12',
  'user_text': 'I would like fifteen years of investment, but I realistically need part of this same '
               'money to pay living costs next year.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D12_SHORT'},
 {'id': 'ADV-D13-COPE_NO_MEANING',
  'question_id': 'D13',
  'user_text': 'I suppose I could cope somehow.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D14-MIXED_TRADEOFF_CONFLICT',
  'question_id': 'D14',
  'user_text': 'I want large potential gains, but I will not accept any fall or permanent loss at all.',
  'expected_bucket': 'Undecided',
  'expected_candidate_option_id': None},
 {'id': 'ADV-D15-DISTRESS_CLEAR',
  'question_id': 'D15',
  'user_text': 'Under the covered-cost example I could accept it while worried, but I’m overwhelmed '
               'today and need a break.',
  'expected_bucket': 'Clarity',
  'expected_candidate_option_id': 'D15_WORRIED_ACCEPTABLE'},
 {'id': 'ADV-D1-URGENT_SAFETY',
  'question_id': 'D1',
  'user_text': 'I cannot go on. Goodbye world.',
  'expected_bucket': 'Confusion',
  'expected_candidate_option_id': None}]


@pytest.fixture(scope="module")
def engine_components():
    base = build_conversation_engine()
    return base.spec, base.d12_resolver


def isolated(engine_components, question_id):
    spec, resolver = engine_components
    engine = ConversationEngine(spec, resolver)
    # Isolates question interpretation; complete profile gates have separate tests.
    engine.state.current_index = engine.state.question_order.index(question_id)
    return engine


def candidate(engine, question_id, response):
    return response.get("candidate_option_id") or engine.state.answers[question_id].candidate_option_id


def bucket(engine, question_id, response):
    if response.get("bucket"):
        return response["bucket"]
    if candidate(engine, question_id, response):
        return "Clarity"
    if response.get("action") in {"pause", "stop", "resume"} or response.get("handled_as") == "declined":
        return "Clarity"
    if response.get("action") == "safety_stop":
        return "Confusion"
    status = engine.state.answers[question_id].status.value
    return "Undecided" if status == "undecided" else "Confusion" if status == "confusion" else None


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_no_wrong_financial_candidate_on_independent_business_probes(engine_components, case):
    engine = isolated(engine_components, case["question_id"])
    response = engine.submit_answer(case["user_text"])
    proposed = candidate(engine, case["question_id"], response)
    if proposed is not None:
        assert proposed == case["expected_candidate_option_id"], response
    assert engine.state.answers[case["question_id"]].selected_option_id is None


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["expected_bucket"] != "Clarity"],
    ids=lambda case: case["id"],
)
def test_nonclarity_bucket_is_semantically_correct(engine_components, case):
    engine = isolated(engine_components, case["question_id"])
    response = engine.submit_answer(case["user_text"])
    assert bucket(engine, case["question_id"], response) == case["expected_bucket"], response
    assert candidate(engine, case["question_id"], response) is None


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["id"].endswith("CLEAR_NATURAL")],
    ids=lambda case: case["id"],
)
def test_clear_natural_financial_meanings_are_recognised(engine_components, case):
    engine = isolated(engine_components, case["question_id"])
    response = engine.submit_answer(case["user_text"])
    assert candidate(engine, case["question_id"], response) == case["expected_candidate_option_id"], response
    assert engine.state.answers[case["question_id"]].selected_option_id is None


def test_d11_confirmed_misconception_does_not_establish_understanding(engine_components):
    engine = isolated(engine_components, "D11")
    engine.submit_answer("I think falls in value always recover.")
    engine.confirm_current(True)
    assert "understanding_hold" in engine.state.active_holds
    assert not engine.completion_report()["eligible_for_final_accuracy_confirmation"]
    response = engine.submit_answer("D11_LOSS_POSSIBLE")
    assert response["action"] in {"understanding_hold", "finish_incomplete_or_pause"}
    assert not engine.save()["saved"]


def test_clear_financial_meaning_with_distress_is_held_for_pacing(engine_components):
    engine = isolated(engine_components, "D15")
    response = engine.submit_answer(
        "Under the covered-cost example I could accept it while worried, "
        "but I’m overwhelmed today and need a break."
    )
    assert candidate(engine, "D15", response) == "D15_WORRIED_ACCEPTABLE"
    assert engine.state.paused
    assert engine.confirm_current(True)["action"] == "paused"
    assert not engine.save()["saved"]


def test_urgent_language_stops_financial_progression(engine_components):
    engine = isolated(engine_components, "D1")
    response = engine.submit_answer("I cannot go on. Goodbye world.")
    assert response["action"] == "safety_stop"
    assert response["simulated_handoff"]
    assert engine.state.stopped_for_safety
    assert candidate(engine, "D1", response) is None
    assert not engine.save()["saved"]


def _advance_synthetic_profile(engine, target=None, overrides=None):
    """Exercise submitted/confirmed sequence; no state seeding or gate bypass."""
    defaults = {
        'D1': 'D1_PURCHASE', 'D3': 'About 10,000 GBP in this pot',
        'D13': 'D13_ADJUSTMENT', 'D14': 'D14_BALANCED',
        'D15': 'D15_WORRIED_ACCEPTABLE',
    }
    overrides = {**defaults, **(overrides or {})}
    while engine.current_question is not None and engine.current_question.id != target:
        question = engine.current_question
        response = engine.submit_answer(overrides.get(question.id, question.options[0].id))
        assert response['action'] == 'confirm', response
        engine.confirm_current(True)


def test_same_pot_withdrawals_cannot_be_other_resource_backup(engine_components):
    spec, resolver = engine_components
    engine = ConversationEngine(spec, resolver)
    _advance_synthetic_profile(engine, 'D6', {
        'D5': 'I pay my regular living costs by withdrawing from this same pot.'
    })
    response = engine.submit_answer(
        'I count those withdrawals from this same pot as my backup for all necessary '
        'living costs over the next twelve months.'
    )
    assert candidate(engine, 'D6', response) is None, response
    assert engine.state.answers['D6'].selected_option_id is None
    assert response['action'] in {'clarify', 'finish_incomplete_or_pause'}
    assert 'same_pot_backup_conflict' in engine.state.active_holds
    # A contextual decline is allowed, but cannot remove an activated material check.
    engine.submit_answer('Prefer not to answer')
    _advance_synthetic_profile(engine)
    assert engine.completion_report()['all_15_handled'] is True
    assert 'same_pot_backup_conflict' in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)['accepted'] is False
    assert engine.save()['saved'] is False


def test_actual_need_before_fixed_deposit_access_is_preserved_and_checked(engine_components):
    spec, resolver = engine_components
    engine = ConversationEngine(spec, resolver)
    _advance_synthetic_profile(engine, 'D12', {
        'D4': 'All this money is in a fixed-term deposit. I cannot access it for five years.'
    })
    response = engine.submit_answer(
        'The earliest realistic need for any included part is in two years. '
        'I need it then for a planned tuition fee.'
    )
    assert candidate(engine, 'D12', response) == 'D12_SHORT', response
    engine.confirm_current(True)
    assert engine.state.answers['D12'].selected_option_id == 'D12_SHORT'
    _advance_synthetic_profile(engine)
    assert engine.completion_report()['all_15_handled'] is True
    assert 'need_before_access' in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)['accepted'] is False
    assert engine.save()['saved'] is False


def test_unavailable_pension_cannot_cover_current_year_backup(engine_components):
    spec, resolver = engine_components
    engine = ConversationEngine(spec, resolver)
    _advance_synthetic_profile(engine, 'D6')
    response = engine.submit_answer(
        'I would count my separate pension as covering all my necessary living costs '
        'for the full twelve months, but I cannot withdraw any of it for five years.'
    )
    assert candidate(engine, 'D6', response) is None, response
    assert engine.state.answers['D6'].selected_option_id is None
    assert 'usable_backup_check' in engine.state.active_holds
    engine.submit_answer('Prefer not to answer')
    _advance_synthetic_profile(engine)
    assert engine.completion_report()['all_15_handled'] is True
    assert 'usable_backup_check' in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)['accepted'] is False
    assert engine.save()['saved'] is False


def test_late_actual_rent_dependency_survives_hypothetical_comfort_answer(engine_components):
    """CR03: a later actual financial fact cannot disappear in a feelings example."""
    spec, resolver = engine_components
    engine = ConversationEngine(spec, resolver)
    _advance_synthetic_profile(engine, 'D15', {
        'D1': 'D1_WEALTH', 'D4': 'D4_EASY_ACCESS', 'D5': 'D5_SALARY',
        'D6': 'D6_ALL', 'D7': 'D7_NO_REQUIRED',
        'D12': 'The earliest realistic need for any of this whole pot is in fifteen '
               'years, with no earlier need.',
        'D13': 'D13_UNAFFECTED',
    })
    response = engine.submit_answer(
        'I cannot imagine costs covered because in reality I need this money for rent.'
    )
    assert candidate(engine, 'D15', response) is None
    assert 'same_money_dependency' in engine.state.active_holds
    # A hypothetical comfort answer does not retract or resolve the actual need.
    response = engine.submit_answer(
        'Under this covered-cost example I would worry but could accept it.'
    )
    assert candidate(engine, 'D15', response) == 'D15_WORRIED_ACCEPTABLE'
    engine.confirm_current(True)
    assert engine.state.answers['D12'].selected_option_id == 'D12_LONG'
    assert engine.state.answers['D13'].selected_option_id == 'D13_UNAFFECTED'
    assert 'same_money_dependency' in engine.state.active_holds
    assert engine.confirm_final_accuracy(True)['accepted'] is False
    assert engine.save()['saved'] is False
