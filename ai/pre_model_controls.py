from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ControlType(str, Enum):
    NONE = "none"
    DELEGATION = "delegation"
    ADVICE_REQUEST = "advice_request"
    REFUSAL = "refusal"
    DISTRESS = "distress"
    PAUSE = "pause"
    STOP = "stop"
    SELF_HARM = "self_harm"
    CORRECTION = "correction"
    UNDECIDED = "undecided"  # compatibility for the wider fixed-choice engine


@dataclass(frozen=True)
class ControlDecision:
    matched: bool
    control_type: ControlType = ControlType.NONE
    action: str | None = None
    flags: tuple[str, ...] = ()
    matched_rule: str | None = None
    message: str | None = None
    reason: str | None = None
    should_call_model: bool = True
    terminal: bool = False

    @property
    def flag(self) -> str | None:
        """Compatibility accessor for the existing D1-D15 engine."""
        return self.flags[0] if self.flags else None


ControlResult = ControlDecision


def _full(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(?:{pattern})\s*[?.!]?\s*$", re.I)


_NEGATED_DELEGATION = _full(
    r"i\s+(?:do\s+not|don't|would\s+not|wouldn't)\s+"
    r"(?:want|ask)\s+you\s+to\s+(?:decide|choose|pick)(?:\s+for\s+me)?"
)
_NEGATED_REFUSAL = _full(
    r"i\s+am\s+not\s+refusing(?:\s*,)?\s+i\s+just\s+(?:do\s+not|don't)\s+know"
)
_SELF_HARM = re.compile(r"\b(kill myself|end my life|suicide|hurt myself|harm myself)\b", re.I)
_DISTRESS_RULES = (
    ("lost_job_cannot_deal", re.compile(r"\blost my job\b.*\b(?:can't|cannot|can not) deal with this(?: right now)?\b", re.I)),
    ("overwhelmed_now", _full(r"i(?:'m| am) overwhelmed(?: right now)?")),
    ("panicking_about_money", _full(r"i(?:'m| am) panicking about money")),
    ("bereavement_cannot_think", re.compile(r"\b(?:had a bereavement|bereavement)\b.*\b(?:can't|cannot|can not) think about this\b", re.I)),
    ("cannot_cope_questions", _full(r"i (?:can't|cannot|can not) cope with (?:these|the) questions(?: right now)?")),
)
_STOP_RULES = (("stop", _full(r"stop")), ("end_this", _full(r"end this")), ("quit", _full(r"quit")))
_PAUSE_RULES = (
    ("pause", _full(r"pause")),
    ("not_now", _full(r"not now")),
    ("come_back_later", _full(r"(?:let(?:'s| us) )?come back later")),
)
_REFUSAL_RULES = (
    ("rather_not_answer", _full(r"i(?:'d| would) rather not answer(?: this)?")),
    ("do_not_want_answer", _full(r"i (?:do not|don't) want to answer(?: this)?")),
    ("prefer_not_say", _full(r"i prefer not to (?:say|answer)")),
    ("not_comfortable_answering", _full(r"i(?:'m| am) not comfortable answering(?: this)?")),
    ("skip_question", _full(r"skip (?:this|the) question")),
)
_DELEGATION_RULES = (
    ("you_decide", _full(r"you decide")),
    ("you_choose", _full(r"you choose")),
    ("pick_for_me", _full(r"(?:pick|choose) (?:one )?for me")),
    ("whatever_you_think", _full(r"whatever you think(?: is best)?")),
    ("you_tell_me_choose", _full(r"you tell me what i should choose")),
    ("choose_best_for_me", _full(r"choose the best one for me")),
)
_ADVICE_RULES = (
    ("which_option_best", _full(r"which option is best for me")),
    ("what_should_choose", _full(r"what should i choose")),
    ("which_recommend", _full(r"which one would you recommend")),
    ("what_recommend", _full(r"what do you recommend")),
    ("what_invest_in", _full(r"what should i invest in")),
    ("timeframe_better", _full(r"which time ?frame is better")),
    ("tell_me_pick", _full(r"tell me what i should pick")),
)
_CORRECTION = re.compile(r"\b(change my answer|correct my answer|that was wrong|i meant something else)\b", re.I)
_UNDECIDED = _full(r"(?:not sure|i (?:don't|do not) know|i (?:can't|cannot) decide)")


def _decision(
    control_type: ControlType,
    action: str,
    rule: str,
    message: str,
    *,
    flag: str,
    terminal: bool = False,
) -> ControlDecision:
    return ControlDecision(
        matched=True,
        control_type=control_type,
        action=action,
        flags=(flag,),
        matched_rule=rule,
        message=message,
        reason=message,
        should_call_model=False,
        terminal=terminal,
    )


def check_pre_model_controls(
    text: str,
    *,
    include_semantic_undecided: bool = True,
) -> ControlDecision:
    """Evaluate narrow controls in safety-first priority order."""

    normalised = text.replace("’", "'").replace("‘", "'")
    if _NEGATED_DELEGATION.match(normalised) or _NEGATED_REFUSAL.match(normalised):
        return ControlDecision(matched=False)

    if _SELF_HARM.search(normalised):
        return _decision(ControlType.SELF_HARM, "safety_stop", "explicit_self_harm", "Possible urgent self-harm language detected before model execution.", flag="possible_urgent_self_harm", terminal=True)

    for rule, pattern in _DISTRESS_RULES:
        if pattern.search(normalised):
            return _decision(ControlType.DISTRESS, "pause_for_distress", rule, "Thank you for telling me. We can pause here. Would you like to continue, pause, or stop?", flag="non_self_harm_distress")
    for rule, pattern in _STOP_RULES:
        if pattern.match(normalised):
            return _decision(ControlType.STOP, "stop", rule, "The profiling conversation has stopped.", flag="user_requested_stop", terminal=True)
    for rule, pattern in _PAUSE_RULES:
        if pattern.match(normalised):
            return _decision(ControlType.PAUSE, "pause", rule, "The profiling conversation is paused.", flag="user_requested_pause")
    for rule, pattern in _REFUSAL_RULES:
        if pattern.match(normalised):
            return _decision(ControlType.REFUSAL, "accept_refusal", rule, "Understood. I will not choose or record a D12 value from this response.", flag="refusal")
    for rule, pattern in _DELEGATION_RULES:
        if pattern.match(normalised):
            return _decision(ControlType.DELEGATION, "reoffer_options", rule, "I cannot choose for you. Please state the earliest realistic time you may need any of the money.", flag="delegation_request")
    for rule, pattern in _ADVICE_RULES:
        if pattern.match(normalised):
            return _decision(ControlType.ADVICE_REQUEST, "explain_scope_and_reoffer", rule, "I can record your timeframe but cannot recommend one. What is the earliest realistic time you may need the money?", flag="advice_request")

    if _CORRECTION.search(normalised):
        return _decision(ControlType.CORRECTION, "correction_request", "correction_request", "User requested a correction.", flag="correction")
    if include_semantic_undecided and _UNDECIDED.match(normalised):
        return _decision(ControlType.UNDECIDED, "undecided", "explicit_undecided", "User is explicitly undecided.", flag="semantic_undecided")
    return ControlDecision(matched=False)
