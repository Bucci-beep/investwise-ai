from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlResult:
    matched: bool
    action: str | None = None
    flag: str | None = None
    reason: str | None = None


_PAUSE = re.compile(
    r"^\s*(?:(?:please|can we|could we|i (?:want|need|would like) to|let's|let us)\s+)?"
    r"(?:pause|stop|quit|end this)"
    r"(?:\s+(?:please|for now|here|this|the (?:chat|conversation|questions|questionnaire)|"
    r"this (?:chat|conversation|questionnaire)|asking(?: me)?(?: these)?(?: questions)?))?"
    r"\s*[.!?]*\s*$",
    re.I,
)
_STOP = re.compile(
    r"^\s*(?:(?:please|can we|could we|i (?:want|need|would like) to|let's|let us)\s+)?"
    r"(?:stop|quit|end this)"
    r"(?:\s+(?:please|for now|here|this|the (?:chat|conversation|questions|questionnaire)|"
    r"this (?:chat|conversation|questionnaire)|asking(?: me)?(?: these)?(?: questions)?))?"
    r"\s*[.!?]*\s*$",
    re.I,
)
_STOP_WITH_CONTEXT = re.compile(
    r"\b(?:please stop (?:asking(?: me)?(?: questions)?|(?:this|the) "
    r"(?:questions|questionnaire|conversation|chat))|"
    r"i (?:want|need|would like) to stop (?:this|the) (?:conversation|chat|questionnaire)|"
    r"stop (?:this|the) (?:conversation|chat|questions|questionnaire))\b", re.I,
)
_DIRECT_REQUEST_START = r"(?:^|[.;!?]|,\s*(?:but\s+|and\s+)?|\bbut\b)\s*"
_STOP_ASKING = re.compile(
    _DIRECT_REQUEST_START + r"(?:please )?stop asking(?: me)?"
    r"(?: these| any more)? (?:profiling |financial |money )?questions\b", re.I,
)
_END_OR_CLOSE = re.compile(
    _DIRECT_REQUEST_START + r"(?:(?:please |(?:can|could) you )?(?:end|close) "
    r"(?:this|the|my) (?:questionnaire|conversation|chat|session)|"
    r"i (?:want|need|would like) (?:this|the|my) (?:conversation|chat|session|questionnaire) "
    r"(?:closed|ended)(?: now)?)\b", re.I,
)
_STOP_AND_END = re.compile(
    r"(?:^|[.;!?])\s*(?:please )?stop(?: now)? and (?:end|close) "
    r"(?:this|the|my) (?:questionnaire|conversation|chat|session)\b", re.I,
)
_END_IT_AND_NO_MORE = re.compile(
    _DIRECT_REQUEST_START + r"(?:please )?end it(?: here| now)? and "
    r"ask me no more (?:profiling |financial |money )?questions\b", re.I,
)
_WITHDRAW_PARTICIPATION = re.compile(
    _DIRECT_REQUEST_START + r"i no longer want to (?:take part in|continue with) "
    r"(?:this|the) (?:questionnaire|conversation|chat|session)\b", re.I,
)
_HOLD_QUESTIONS = re.compile(
    _DIRECT_REQUEST_START + r"(?:please |(?:can|could) we )?(?:hold|put on hold) "
    r"(?:this|the|these) (?:questions|questionnaire|conversation|chat|session)"
    r"(?: for now)?\b", re.I,
)
_PRIORITY_PACING = re.compile(
    _DIRECT_REQUEST_START + r"(?:(?:can|could) we (?:go|take this) more slowly|"
    r"please (?:go|take this) more slowly)"
    r"(?:,? with (?:only |just )?one (?:short )?question at a time)?\b", re.I,
)
_HUMAN_SUPPORT_FIRST = re.compile(
    r"\bi (?:would like|want|need) (?:a person|someone|a human|a support agent) "
    r"to (?:help|support) me\b.{0,180}\b"
    r"(?:i (?:do not|don't) want (?:any )?more (?:profiling |financial |money )?questions "
    r"until i (?:have|get|receive) (?:that |this |some )?(?:support|help)|"
    r"before (?:any )?(?:more |further )?(?:profiling |financial |money )?questions)\b", re.I,
)
_PAUSE_WITH_CONTEXT = re.compile(
    r"\b(?:please pause|can we pause|could we pause|"
    r"(?:i )?(?:want|need|would like) to pause(?: now)?|"
    r"(?:i )?need (?:you to pause|(?:this|the) (?:chat|conversation) paused|a pause)|"
    r"pause (?:please|here|for now|this(?: (?:chat|conversation|questions|questionnaire))?|the (?:chat|conversation|questions|questionnaire))|"
    r"(?:please )?stop (?:here please|for now))\b", re.I,
)
_PAUSE_CONTROL_ONLY = re.compile(
    r"^\s*(?:please |(?:can|could) (?:we|you) )?pause"
    r"(?: (?:(?:this|the) (?:chat|conversation|questions|questionnaire)|here|for now))?"
    r"(?: here)?"
    r"(?: for (?:now|a (?:short |brief )?moment))?(?:,? please)?"
    r"(?:\s*[;,.!?]\s*(?:i need (?:a |to take a )?(?:short |little |brief )?break"
    r"(?: from answering)?(?: and (?:will tell you when i(?: am|'m) ready to (?:continue|resume)|"
    r"will (?:return|come back)(?: to (?:this|the) questions)? later))?|"
    r"i need to step away(?: for a (?:short |little |brief )?while)?|"
    r"i will (?:return|come back) later|i will tell you when i(?: am|'m) ready to (?:continue|resume)))*"
    r"\s*[.!?]*\s*$",
    re.I,
)
_DIRECT_PAUSE = re.compile(
    _DIRECT_REQUEST_START + r"(?:can|could) you pause "
    r"(?:this|the) (?:chat|conversation|questions|questionnaire)\b", re.I,
)
_SAVE_PLACE_PAUSE_ONLY = re.compile(
    r"^\s*(?:please )?(?:save|keep|remember) (?:our|my|the) place and pause "
    r"(?:here|for now)(?:,? please)?[.!?;]*\s*"
    r"(?:i (?:would like|want) to (?:resume|continue)(?: answering)? later"
    r"(?: (?:this (?:morning|afternoon|evening)|today|tomorrow))?[.!?]*)?\s*$", re.I,
)
_RESUME = re.compile(
    r"^\s*(?:(?:please|can we|could we|let's|let us|i (?:want|would like) to|"
    r"i(?: am|'m) ready to)\s+)?(?:resume|continue)"
    r"(?:\s+(?:please|now|the (?:chat|conversation)|this (?:chat|conversation)))?"
    r"\s*[.!?]*\s*$",
    re.I,
)
_REFUSAL = re.compile(r"\b(prefer not to answer|rather not answer|don't want to answer|do not want to answer|prefer not to share (?:my age|this information|that information)|(?:would )?rather not say (?:what|how|when|which))\b", re.I)
_LEAVE_UNANSWERED = re.compile(
    _DIRECT_REQUEST_START + r"(?:please )?leave "
    r"(?:this|this one|that|that one|this question|the question) unanswered\b", re.I,
)
_PRIVACY_WITHHOLDING = re.compile(
    r"\b(?:keep|keeping) (?:it|this|that|(?:my|these|those) "
    r"(?:living[- ]cost )?(?:funding|income|source information|personal details|details|information)) private\b",
    re.I,
)
_UNDECIDED = re.compile(
    r"^\s*(?:(?:i(?: am|'m)?\s+)?not sure|i (?:don't|do not) know|"
    r"(?:i )?(?:can't|cannot) (?:decide|say|tell))"
    r"(?:\s+(?:when|what|which|whether|how|if|about)\b[^.!?]*)?"
    r"\s*[.!?]*\s*$|^\s*i understand but (?:can't|cannot) answer(?: yet)?\s*[.!?]*\s*$",
    re.I,
)
_CORRECTION = re.compile(r"\b(change my answer|correct my answer|that was wrong|i meant something else)\b", re.I)
_CORRECTION_WITH_ANSWER = re.compile(
    r"(?:^|[.;,])\s*(?:sorry[,;.!]?\s*)?correction\s*[:,-]\s*\S|"
    r"\bthat is not my answer now\b", re.I,
)
_HELP = re.compile(
    r"^\s*(?:help(?: me)?(?: please)?|(?:please )?help(?: me)?|i need help|"
    r"(?:can|could) you help(?: me)?|(?:please |(?:can|could) you )?explain"
    r"(?: (?:this|the))?(?: question|choices|options)(?: and (?:choices|options))?|"
    r"what does (?:this|that) mean|i (?:don't|do not) understand"
    r"(?: (?:this|the) (?:question|choices|options))?|i need different help)"
    r"\s*[.!?]*\s*$",
    re.I,
)
_HELP_CONTEXT = re.compile(
    r"^\s*(?:i need (?:a bit of |a little |some )?help(?: with (?:this|that|this one|that one|the question))?|"
    r"(?:can|could) you help(?: me)? (?:with (?:this question|the count|this one)|answer (?:this|the) (?:money |investment )?question))\s*[.!?]*\s*$|"
    r"^\s*(?:what counts as\b|are you asking\b|do you mean\b|what do you mean\b|what does\b.*\bmean\b|what (?:is|are)\b|does\b.*\bmean\b|by\b.*\bdo you mean\b|do you want what i have now\b|i (?:do not|don't) understand what\b.*\bmeans?\b|i(?: am|'m) (?:still )?not sure what\b.*\bmeans?\b)",
    re.I,
)
_HELP_ONLY_CLAUSE = (
    r"(?:i need (?:a bit of |a little |some )?help (?:understanding|with) "
    r"(?:this|the) (?:question|answer choices|choices|options)|"
    r"(?:please )?help(?: me)? with (?:this|the) question(?: please)?|"
    r"(?:please )?help me understand (?:the|these) (?:answer )?(?:choices|options)"
    r"(?: here)?(?: before i (?:give|provide) (?:an|any) answer)?(?:,? please)?|"
    r"i need an explanation of (?:this|the) (?:question|choices|options)"
    r"(?: in (?:this|the) section)?(?: before giving (?:any|an) answer)?|"
    r"(?:please |(?:can|could) you )?explain "
    r"(?:what (?:the|these) (?:answer )?(?:choices|options) mean|"
    r"what (?:this|the) question means|"
    r"(?:this|the) question(?: and (?:the )?(?:answer )?(?:choices|options))?|"
    r"(?:the|these) (?:different )?(?:answer )?(?:choices|options)))"
    r"(?: in (?:everyday language|plain language|plain words))?"
    r"(?: before i (?:try to )?(?:answer|respond)(?: (?:this|the) question)?)?"
)
_HELP_CONTROL_ONLY = re.compile(
    r"^\s*(?:before i (?:try to )?(?:answer|respond),?\s*)?" + _HELP_ONLY_CLAUSE +
    r"(?:(?:\s*[,;.!?]\s*|\s+and\s+)(?:" + _HELP_ONLY_CLAUSE +
    r"|i (?:have not|haven't) (?:given|provided|supplied) (?:an|any) answer(?: yet)?"
    r"|i(?: am|'m) asking for help,? not answering (?:it|this|the question)(?: yet)?))*"
    r"\s*[.!?]*\s*$", re.I,
)
_TALK_THROUGH_HELP_ONLY = re.compile(
    r"^\s*(?:please |(?:can|could) you )?talk me through "
    r"what (?:this|the) question is asking me to include[.!?;]*\s*"
    r"(?:i (?:want|need|would like) an explanation first[.!?]*)?\s*$", re.I,
)
_WALKTHROUGH_HELP = re.compile(
    r"^\s*(?:please |(?:can|could) you )?walk me through "
    r"(?:what counts as (?:a |an )?(?:source|answer|option) in (?:this|the) question|"
    r"(?:this|the|these) (?:question|answer choices|options))"
    r"(?: before (?:asking (?:me )?for (?:a |an )?(?:choice|answer)|i (?:answer|choose|respond)))?"
    r"[.!?;]*\s*(?:i(?: am|'m) getting (?:flustered|confused) by (?:this|the) wording[.!?]*)?\s*$", re.I,
)
_UNDECIDED_CONTEXT = re.compile(r"\b(?:can't|cannot) decide\b|\bi understand\b.*\b(?:can't|cannot) answer\b",re.I)
_HELP_WITH_ANSWER = re.compile(
    r"\b(?:please explain|(?:can|could) you explain|i need help|help me understand|"
    r"(?:please |(?:can|could) you )talk me through|"
    r"(?:please )?slow down|what does (?:this|that) mean)\b",
    re.I,
)
_ACCESSIBILITY = re.compile(
    r"\b(?:short(?:er)? sentences|one short sentence|one (?:short )?question at a time|one thing at a time|larger text|plain words|ordinary words|make the text (?:bigger|larger)|enlarge the text|"
    r"screen reader|read (?:it|this) out|explain (?:it|this) simply)\b", re.I,
)
_BREAK = re.compile(r"\bi (?:need|want) (?:a |to take a )?(?:short |little |brief )?(?:break|moment)\b",re.I)
_ACCESSIBILITY_ONLY = re.compile(
    r"^\s*(?:please\s+)?(?:use (?:larger text|plain words|short(?:er)? sentences)|"
    r"make the text (?:bigger|larger)|enlarge the text|"
    r"(?:ask|show) (?:just )?one (?:short )?question at a time)"
    r"(?:\s+and\s+(?:(?:(?:ask|show) )?(?:just )?one (?:short )?question at a time|use (?:plain words|short(?:er)? sentences|larger text)))?"
    r"(?:\s+so i can (?:follow|understand|read) (?:this|the) (?:page|question|questions)(?: more easily| better)?)?"
    r"(?:[;,.]\s*i (?:have not|haven't) answered(?: (?:this|the) (?:loss )?question)? yet)?\s*[.!?]*\s*$",re.I,
)

# Deliberately high precision. The rule layer should not try to diagnose distress.
_SELF_HARM = re.compile(
    r"\b(kill myself|end my life|suicide|hurt myself|harm myself|goodbye world)\b|"
    r"\bi(?:'?m| am)?\s+(?:going to|gonna|about to|will|plan to|planning to|intend to|want to)\s+"
    r"take\s+my(?: own)?\s+life\b(?!\s+(?:insurance|assurance)\b)|"
    r"\bi(?:'?m| am)?\s+(?:going to|gonna|about to|will|plan to|planning to|intend to|want to)\s+"
    r"jump\s+(?:off|from|out of)\s+(?:(?:a|the|this|that|my)\s+)?"
    r"(?:building|roof|rooftop|bridge|balcony|window)\b|"
    r"\bi(?: am|'m) (?:thinking (?:of|about)|considering) (?:hurting|harming|killing) myself\b",
    re.I,
)
_URGENT_OVERDOSE = re.compile(
    r"\bi(?: have|'ve) (?:just )?taken an overdose\b"
    r"(?:\s+and\s+|\s*[.;!?]\s*)(?:i\s+)?(?:need|require) "
    r"(?:emergency|urgent)(?: medical)? (?:help|assistance)(?: right)? now\b|"
    r"\bi (?:just )?took an overdose\b"
    r"(?:\s+and\s+|\s*[.;!?]\s*)(?:i\s+)?(?:need|require) "
    r"(?:emergency|urgent)(?: medical)? (?:help|assistance)(?: right)? now\b", re.I,
)
_IMMEDIATE_PHYSICAL_DANGER = re.compile(
    r"\bi(?: am|'m) in immediate danger and (?:i )?need "
    r"(?:urgent|emergency)(?: medical)? (?:help|assistance)(?: right)? now\b", re.I,
)


def _in_quotation(text: str, start: int) -> bool:
    prefix = text[:start]
    singles = re.findall(r"(?<!\w)'|'(?!\w)", prefix.replace("‘", "'").replace("’", "'"))
    return bool(len(singles) % 2 or prefix.count('"') % 2
                or prefix.count("“") > prefix.count("”"))


def _current_safety_match(pattern: re.Pattern, text: str) -> bool:
    for match in pattern.finditer(text):
        prefix = re.split(r"[.;!?]", text[:match.start()])[-1]
        negated_safety = re.search(
            r"\b(?:not|never|don't|won't)\s+"
            r"(?:(?:going|about|planning|want|intend|plan|taking|take|use|to|the|these|those|them|pills|tablets|my)\s+){0,10}$",
            prefix, re.I,
        )
        if _in_quotation(text, match.start()) or _negated_request_prefix(prefix) or negated_safety:
            continue
        if re.search(r"\b(?:if|suppose|imagine|pretend|said|says|wrote|writes|quoted|quotes)\b|"
                     r"^\s*in (?:a |the |my )?(?:fictional|fiction|novel|story|script)\b", prefix, re.I):
            continue
        return True
    return False


def _current_urgent_overdose(text: str) -> bool:
    # Completed first-person ingestion plus an explicit present emergency is a
    # safety control. Keep quoted, fictional and conditional examples out of it.
    for match in _URGENT_OVERDOSE.finditer(text):
        prefix = text[:match.start()]
        clause_prefix = re.split(r"[.;!?]", prefix)[-1]
        if _in_quotation(text, match.start()):
            continue
        if re.search(r"\b(?:if|suppose|imagine|pretend|fictional|fiction|novel|story|character|"
                     r"(?:said|says|wrote|writes|quoted|quotes))\b", clause_prefix, re.I):
            continue
        if not _negated_request_prefix(clause_prefix):
            return True
    return False


def _context_pause_action(text: str) -> str | None:
    # A negated request is local to its clause. Unrelated negations in an
    # otherwise clear answer must not suppress a later explicit pause.
    for match in sorted(
            [*_PAUSE_WITH_CONTEXT.finditer(text), *_DIRECT_PAUSE.finditer(text)],
            key=lambda item: item.start()):
        prefix = text[max(0, match.start() - 35):match.start()]
        if re.search(r"\b(?:save|keep|remember) (?:our|my|the|her|his|their) place and\s*$", prefix, re.I):
            if not re.search(_DIRECT_REQUEST_START + r"(?:please )?(?:save|keep|remember) "
                             r"(?:our|my|the) place and\s*$", text[:match.start()], re.I):
                continue
        reported_request = re.search(
            r"\b(?:(?:if i (?:asked|were to ask)|(?:tomorrow|later) i (?:may|might|will) ask)"
            r"(?: you)? to|(?:said|says|asked)\s*['\"“]"
            r"(?:(?:please|(?:can|could) (?:we|you))\s+)?)\s*$", prefix, re.I,
        )
        if not _negated_request_prefix(prefix) and not reported_request and not _in_quotation(text, match.start()):
            if not text[:match.start()].strip() and not text[match.end():].strip(" .!?"):
                return "pause"
            if not text[:match.start()].strip() and re.fullmatch(
                r"\s*(?:for now)?\s*[,;.!?]?\s*(?:please)?\s*[,;.!?]?\s*"
                r"(?:leave (?:it|this) (?:where we are|here)(?: so i can "
                r"(?:pick it up|continue|resume)(?: (?:later|tomorrow)(?: (?:morning|afternoon|evening))?)?)?)?"
                r"\s*[.!?]*\s*", text[match.end():], re.I,
            ):
                return "pause"
            return "pace_answer"
    return None


def _negated_request_prefix(prefix: str) -> bool:
    """Do not turn a locally negated conversational request into a control."""
    return bool(re.search(
        r"\b(?:don't|do not|not|never|no longer|won't|will not)"
        r"(?:\s+(?:really|actually|please|want|need|require|you|to)){0,4}\s*$", prefix, re.I,
    ))


def _unnegated_match(pattern: re.Pattern, text: str) -> bool:
    return any(not _negated_request_prefix(text[max(0, m.start() - 45):m.start()])
               and not _in_quotation(text, m.start())
               for m in pattern.finditer(text))


def _current_help_with_answer(text: str) -> bool:
    for match in _HELP_WITH_ANSWER.finditer(text):
        prefix = re.split(r"[.;!?]", text[:match.start()])[-1]
        if _negated_request_prefix(prefix) or _in_quotation(text, match.start()):
            continue
        if re.search(r"\b(?:if|suppose|imagine|pretend|fictional|fiction|novel|story|character|"
                     r"said|says|wrote|writes|quoted|quotes)\b|"
                     r"\b(?:tomorrow|later)\b.*\b(?:may|might|will|would)\b", prefix, re.I):
            continue
        return True
    return False


def pre_interpretation_pacing_requested(text: str) -> bool:
    """An explicit conversational pacing request takes priority over proposals.

    This finite request recogniser supplies no financial facts or categories.
    Generic accessibility requests accompanying an answer keep their existing
    interpretation route; asking to go more slowly is handled first.
    """
    return _unnegated_match(_PRIORITY_PACING, text.replace("’", "'"))

_DISTRESS = re.compile(
    r"\b(?:i(?: am|'m) (?:feeling )?(?:panicking|overwhelmed|distressed|grieving|upset|anxious)|i(?: am|'m) having a panic attack|"
    r"i (?:cannot|can't) cope|i feel (?:overwhelmed|distressed)|"
    r"(?:i )?feel ashamed(?: talking about it)?|(?:this|reading this) makes me anxious|"
    r"my bereavement is upsetting|"
    r"i(?: am|'m) (?:recently )?bereaved|"
    r"i(?: have|'ve) (?:recently )?(?:experienced|suffered) (?:a |my )?bereavement|"
    r"(?:after|following|since) (?:a |my )?(?:recent )?bereavement[,; ]+i\b|"
    r"my (?:husband|wife|spouse|partner|mother|father) (?:has )?(?:died|passed away)|"
    r"i (?:have )?(?:just )?lost my job|"
    r"(?:these|the|this) (?:money |financial )?questions (?:are making|make) me "
    r"(?:tearful|upset|overwhelmed|anxious))\b",
    re.I,
)


def check_pre_model_controls(text: str) -> ControlResult:
    text = text.replace("’", "'")
    if _current_safety_match(_IMMEDIATE_PHYSICAL_DANGER, text):
        return ControlResult(matched=True, action="safety_stop", flag="possible_urgent_physical_danger",
            reason="User disclosed immediate physical danger and requested urgent help before model execution.")

    if _current_safety_match(_SELF_HARM, text) or _current_urgent_overdose(text):
        return ControlResult(
            matched=True,
            action="safety_stop",
            flag="possible_urgent_self_harm",
            reason="Possible urgent self-harm language detected before model execution.",
        )

    if (_STOP.match(text) or _unnegated_match(_STOP_WITH_CONTEXT, text)
            or _unnegated_match(_STOP_ASKING, text) or _unnegated_match(_END_OR_CLOSE, text)
            or _unnegated_match(_STOP_AND_END, text) or _unnegated_match(_END_IT_AND_NO_MORE, text)
            or _unnegated_match(_WITHDRAW_PARTICIPATION, text)):
        return ControlResult(
            matched=True,
            action="stop",
            flag="pause_or_stop",
            reason="User explicitly requested to end this conversation.",
        )

    if (_PAUSE.match(text) or _PAUSE_CONTROL_ONLY.fullmatch(text)
            or _SAVE_PLACE_PAUSE_ONLY.fullmatch(text) or _unnegated_match(_HOLD_QUESTIONS, text)):
        return ControlResult(
            matched=True,
            action="pause",
            flag="pause_or_stop",
            reason="User requested pause or stop.",
        )

    context_pause = _context_pause_action(text)
    if context_pause:
        return ControlResult(
            matched=True,
            action=context_pause,
            flag="pause_or_stop",
            reason="User explicitly requested a pause; preserve supported meaning without confirmation.",
        )

    if _RESUME.match(text):
        return ControlResult(
            matched=True,
            action="resume",
            reason="User explicitly requested to resume.",
        )

    if _unnegated_match(_HUMAN_SUPPORT_FIRST, text):
        return ControlResult(matched=True, action="support_control", flag="human_support_request",
            reason="User explicitly requested human support before further profiling; no financial interpretation.")

    if _DISTRESS.search(text):
        return ControlResult(
            matched=True,
            action="care_pause",
            flag="safety_or_distress",
            reason="User disclosed distress or a significant life event; offer control over pacing.",
        )

    if pre_interpretation_pacing_requested(text):
        return ControlResult(matched=True, action="pace_answer", flag="pause_or_stop",
            reason="User explicitly requested slower pacing before further interpretation.")

    if _unnegated_match(_BREAK, text):
        return ControlResult(matched=True, action="pace_answer", flag="pause_or_stop", reason="User requested a break; preserve supported meaning but pause before confirmation.")

    # A bare leave-unanswered request is the engine's existing leave-unresolved
    # control, not refusal. Only explicit privacy/withholding adds a decline.
    if (_unnegated_match(_REFUSAL, text)
            or (_unnegated_match(_LEAVE_UNANSWERED, text) and _unnegated_match(_PRIVACY_WITHHOLDING, text))):
        return ControlResult(
            matched=True,
            action="decline",
            flag="refusal",
            reason="User declined the current question.",
        )

    if _CORRECTION.search(text):
        return ControlResult(
            matched=True,
            action="correction_request",
            flag="correction",
            reason="User requested a correction.",
        )

    if _CORRECTION_WITH_ANSWER.search(text):
        return ControlResult(
            matched=True,
            action="correction_answer",
            flag="correction",
            reason="User explicitly supplied a revised answer; preserve only supported financial meaning.",
        )

    unrelated = re.search(r"\b(?:weather|dinosaurs|restaurant|phone charger|book a flight)\b",text,re.I)
    if (_HELP.match(text) or _HELP_CONTROL_ONLY.fullmatch(text) or _WALKTHROUGH_HELP.fullmatch(text)
            or _TALK_THROUGH_HELP_ONLY.fullmatch(text) or (_HELP_CONTEXT.search(text) and not unrelated)):
        return ControlResult(
            matched=True,
            action="explain",
            flag="help_request",
            reason="User requested help without a financial answer.",
        )

    if _ACCESSIBILITY_ONLY.match(text):
        return ControlResult(matched=True, action="support_control", flag="accessibility", reason="User clearly requested an accessible presentation without a financial answer.")

    if _unnegated_match(_ACCESSIBILITY, text):
        return ControlResult(
            matched=True,
            action="support_answer",
            flag="accessibility",
            reason="User requested a simpler or more accessible presentation; no financial fact inferred.",
        )

    if _current_help_with_answer(text):
        return ControlResult(
            matched=True,
            action="support_answer",
            flag="help_request",
            reason="User requested support alongside their reply.",
        )

    if _UNDECIDED.match(text) or (_UNDECIDED_CONTEXT.search(text) and not re.search(r"\bstill prioriti[sz]e\b",text,re.I)):
        return ControlResult(
            matched=True,
            action="undecided",
            flag=None,
            reason="User is explicitly undecided.",
        )

    return ControlResult(matched=False)
