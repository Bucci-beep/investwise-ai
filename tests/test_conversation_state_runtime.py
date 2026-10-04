from __future__ import annotations

import pytest

from ai.conversation_factory import build_d12_interpreter
from ai.conversation_runtime import ConversationRuntime
from ai.conversation_state import ConversationState, QuestionStatus


@pytest.fixture(scope="module")
def runtime() -> ConversationRuntime:
    return ConversationRuntime(build_d12_interpreter())


def test_value_is_not_recorded_before_confirmation():
    state = ConversationState()
    state.set_proposed_value("D12", "D12_MEDIUM")
    question = state.get_question("D12")
    assert question.proposed_value == "D12_MEDIUM"
    assert question.recorded_value is None
    assert question.pending_confirmation is True


def test_confirmation_records_exactly_the_proposed_value():
    state = ConversationState()
    state.set_proposed_value("D12", "D12_LONG")
    assert state.confirm_value("D12") == "D12_LONG"
    assert state.get_question("D12").recorded_value == "D12_LONG"


def test_rejection_clears_pending_proposal():
    state = ConversationState()
    state.set_proposed_value("D12", "D12_SHORT")
    state.reject_confirmation("D12")
    question = state.get_question("D12")
    assert question.proposed_value is None
    assert question.recorded_value is None
    assert question.pending_confirmation is False
    assert question.status == QuestionStatus.IN_PROGRESS


def test_clarification_count_and_maximum():
    state = ConversationState(max_clarifications_per_question=3)
    assert state.increment_clarification("D12") == 1
    assert state.increment_clarification("D12") == 2
    assert state.get_question("D12").status == QuestionStatus.IN_PROGRESS
    assert state.increment_clarification("D12") == 3
    assert state.get_question("D12").status == QuestionStatus.INCOMPLETE


def test_pause_and_resume_preserve_confirmation_state():
    state = ConversationState()
    state.set_proposed_value("D12", "D12_MEDIUM")
    state.pause()
    assert state.paused is True
    assert state.get_question("D12").status == QuestionStatus.PAUSED
    state.resume()
    assert state.paused is False
    assert state.get_question("D12").status == QuestionStatus.AWAITING_CONFIRMATION


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 year", "D12_SHORT"),
        ("exactly 3 years", "D12_SHORT"),
        ("4 years", "D12_MEDIUM"),
        ("exactly 10 years", "D12_MEDIUM"),
        ("11 years", "D12_LONG"),
    ],
)
def test_explicit_boundaries_flow_through_runtime(runtime, text, expected):
    state = ConversationState()
    response = runtime.handle_reply("D12", text, state)
    assert response.action == "confirm"
    assert response.proposed_value == expected
    assert response.recorded_value is None
    assert response.deterministic_mapping_status == "mapped"


def test_runtime_confirmation_flow(runtime):
    state = ConversationState()
    proposal = runtime.handle_reply("D12", "about 7 years", state)
    assert proposal.action == "confirm"
    assert proposal.recorded_value is None
    recorded = runtime.handle_reply("D12", "yes", state)
    assert recorded.action == "recorded"
    assert recorded.recorded_value == proposal.proposed_value == "D12_MEDIUM"


def test_runtime_rejection_never_records(runtime):
    state = ConversationState()
    runtime.handle_reply("D12", "about 7 years", state)
    rejected = runtime.handle_reply("D12", "no", state)
    assert rejected.action == "reask"
    assert rejected.proposed_value is None
    assert rejected.recorded_value is None


@pytest.mark.parametrize("text", ["3 or 4 years", "10 or 11 years"])
def test_ambiguous_duration_does_not_produce_value(runtime, text):
    response = runtime.handle_reply("D12", text, ConversationState())
    assert response.action == "clarify"
    assert response.proposed_value is None
    assert response.recorded_value is None
    assert response.deterministic_mapping_status == "ambiguous"


def test_semantic_undecided_does_not_produce_category(runtime):
    response = runtime.handle_reply("D12", "honestly I have no idea", ConversationState())
    assert response.action == "reoffer"
    assert response.semantic_undecided is True
    assert response.proposed_value is None


def test_confusion_does_not_produce_category(runtime):
    response = runtime.handle_reply("D12", "I do not understand the question", ConversationState())
    assert response.action == "explain/restate"
    assert response.proposed_value is None


def test_model_uncertainty_does_not_produce_category(runtime):
    response = runtime.handle_reply(
        "D12", "I might need some next year but I am not certain", ConversationState()
    )
    assert response.action == "clarify"
    assert response.model_uncertain is True
    assert response.proposed_value is None
    assert response.recorded_value is None


def test_three_runtime_clarifications_mark_incomplete(runtime):
    state = ConversationState()
    runtime.handle_reply("D12", "I do not understand the question", state)
    runtime.handle_reply("D12", "what does earliest mean", state)
    third = runtime.handle_reply("D12", "banana telescope pension", state)
    assert third.action == "incomplete"
    assert third.clarification_count == 3
    assert third.question_status == "incomplete"


def test_summary_is_serialisable():
    state = ConversationState()
    state.set_proposed_value("D12", "D12_SHORT")
    summary = state.summary()
    assert summary["questions"]["D12"]["status"] == "awaiting_confirmation"


class SpyInterpreter:
    question_id = "D12"

    def __init__(self, delegate=None):
        self.delegate = delegate
        self.calls = 0

    def interpret(self, text):
        self.calls += 1
        if self.delegate is None:
            raise AssertionError("The interpreter must not be called for this control")
        return self.delegate.interpret(text)


@pytest.mark.parametrize(
    ("text", "control_type", "action"),
    [
        ("you decide", "delegation", "reoffer_options"),
        ("pick for me", "delegation", "reoffer_options"),
        ("which option is best for me", "advice_request", "explain_scope_and_reoffer"),
        ("what do you recommend", "advice_request", "explain_scope_and_reoffer"),
        ("I'd rather not answer", "refusal", "accept_refusal"),
        ("skip this question", "refusal", "accept_refusal"),
    ],
)
def test_non_model_controls_directly_bypass_interpreter(text, control_type, action):
    spy = SpyInterpreter()
    response = ConversationRuntime(spy).handle_reply("D12", text, ConversationState())
    assert spy.calls == 0
    assert response.control_type == control_type
    assert response.control_matched is True
    assert response.should_call_model is False
    assert response.action == action
    assert response.proposed_value is None
    assert response.recorded_value is None


def test_refusal_marks_question_incomplete_and_flagged():
    spy = SpyInterpreter()
    state = ConversationState()
    response = ConversationRuntime(spy).handle_reply("D12", "I prefer not to say", state)
    assert spy.calls == 0
    assert response.question_status == "incomplete"
    assert "refusal" in response.governance_flags


@pytest.mark.parametrize(
    ("text", "control_type", "action", "paused", "stopped"),
    [
        ("I'm overwhelmed right now", "distress", "pause_for_distress", True, False),
        ("pause", "pause", "pause", True, False),
        ("not now", "pause", "pause", True, False),
        ("stop", "stop", "stop", False, True),
        ("I want to kill myself", "self_harm", "safety_stop", False, True),
    ],
)
def test_state_controls_bypass_interpreter(text, control_type, action, paused, stopped):
    spy = SpyInterpreter()
    state = ConversationState()
    response = ConversationRuntime(spy).handle_reply("D12", text, state)
    assert spy.calls == 0
    assert response.control_type == control_type
    assert response.action == action
    assert state.paused is paused
    assert state.stopped is stopped
    assert response.recorded_value is None


@pytest.mark.parametrize(
    ("text", "expected_action"),
    [
        ("I don't want you to decide for me", "explain/restate"),
        ("I wouldn't ask you to choose", "explain/restate"),
        ("I am not refusing, I just don't know", "reoffer"),
    ],
)
def test_negated_controls_reach_interpreter(text, expected_action):
    spy = SpyInterpreter(build_d12_interpreter())
    response = ConversationRuntime(spy).handle_reply("D12", text, ConversationState())
    assert spy.calls == 1
    assert response.control_matched is False
    assert response.control_type == "none"
    assert response.should_call_model is True
    assert response.action == expected_action


@pytest.mark.parametrize(
    ("text", "expected_action"),
    [
        ("about seven years", "confirm"),
        ("honestly I have no idea", "reoffer"),
        ("I do not understand the question", "explain/restate"),
    ],
)
def test_normal_d12_paths_still_call_interpreter(text, expected_action):
    spy = SpyInterpreter(build_d12_interpreter())
    response = ConversationRuntime(spy).handle_reply("D12", text, ConversationState())
    assert spy.calls == 1
    assert response.should_call_model is True
    assert response.action == expected_action


def test_control_clears_pending_proposal_without_recording():
    spy = SpyInterpreter()
    state = ConversationState()
    state.set_proposed_value("D12", "D12_MEDIUM")
    response = ConversationRuntime(spy).handle_reply("D12", "you decide", state)
    question = state.get_question("D12")
    assert spy.calls == 0
    assert response.recorded_value is None
    assert question.proposed_value is None
    assert question.recorded_value is None
    assert question.pending_confirmation is False
