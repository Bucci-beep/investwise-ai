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
_STOP_WITH_CONTEXT = re.compile(r"\b(?:please stop (?:asking(?: me)?(?: questions)?|the questions|this conversation)|i want to stop this conversation)\b",re.I)
_PAUSE_WITH_CONTEXT = re.compile(
    r"\b(?:please pause|can we pause|could we pause|"
    r"(?:i )?(?:want|need|would like) to pause(?: now)?|"
    r"(?:i )?need (?:you to pause|(?:this|the) (?:chat|conversation) paused|a pause)|"
    r"pause (?:please|here|for now|this (?:chat|conversation))|"
    r"(?:please )?stop (?:here please|for now))\b", re.I,
)
_RESUME = re.compile(
    r"^\s*(?:(?:please|can we|could we|let's|let us|i (?:want|would like) to|"
    r"i(?: am|'m) ready to)\s+)?(?:resume|continue)"
    r"(?:\s+(?:please|now|the (?:chat|conversation)|this (?:chat|conversation)))?"
    r"\s*[.!?]*\s*$",
    re.I,
)
_REFUSAL = re.compile(r"\b(prefer not to answer|rather not answer|don't want to answer|do not want to answer|prefer not to share (?:my age|this information|that information)|(?:would )?rather not say (?:what|how|when|which))\b", re.I)
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
_UNDECIDED_CONTEXT = re.compile(r"\b(?:can't|cannot) decide\b|\bi understand\b.*\b(?:can't|cannot) answer\b",re.I)
_HELP_WITH_ANSWER = re.compile(
    r"\b(?:please explain|(?:can|could) you explain|i need help|help me understand|"
    r"(?:please )?slow down|what does (?:this|that) mean)\b",
    re.I,
)
_ACCESSIBILITY = re.compile(
    r"\b(?:short(?:er)? sentences|one short sentence|one (?:short )?question at a time|one thing at a time|larger text|plain words|ordinary words|make the text (?:bigger|larger)|enlarge the text|"
    r"screen reader|read (?:it|this) out|explain (?:it|this) simply)\b", re.I,
)
_BREAK = re.compile(r"\bi (?:need|want) (?:a |to take a )?(?:break|moment)\b",re.I)
_ACCESSIBILITY_ONLY = re.compile(
    r"^\s*(?:please\s+)?(?:use (?:larger text|plain words|short sentences)|"
    r"ask (?:just )?one (?:short )?question at a time)"
    r"(?:\s+and\s+(?:(?:ask )?(?:just )?one (?:short )?question at a time|use (?:plain words|short sentences|larger text)))?"
    r"(?:[;,.]\s*i (?:have not|haven't) answered(?: (?:this|the) (?:loss )?question)? yet)?\s*[.!?]*\s*$",re.I,
)

# Deliberately high precision. The rule layer should not try to diagnose distress.
_SELF_HARM = re.compile(
    r"\b(kill myself|end my life|suicide|hurt myself|harm myself|goodbye world)\b|"
    r"\bi(?:'?m| am)?\s+(?:going to|gonna|about to|will|plan to|planning to|intend to|want to)\s+"
    r"take\s+my(?: own)?\s+life\b(?!\s+(?:insurance|assurance)\b)|"
    r"\b(?:i(?:'?m| am)?\s+(?:going to|gonna|about to)|i (?:will|plan to))\s+"
    r"jump\s+(?:off|from|out of)\s+(?:(?:a|the|this|that|my)\s+)?"
    r"(?:building|roof|rooftop|bridge|balcony|window)\b",
    re.I,
)


def _context_pause_action(text: str) -> str | None:
    # A negated request is local to its clause. Unrelated negations in an
    # otherwise clear answer must not suppress a later explicit pause.
    for match in _PAUSE_WITH_CONTEXT.finditer(text):
        prefix = text[max(0, match.start() - 35):match.start()]
        if not re.search(r"\b(?:don't|do not|not|never|no longer|won't|will not)\s*$", prefix, re.I):
            if not text[:match.start()].strip() and not text[match.end():].strip(" .!?"):
                return "pause"
            return "pace_answer"
    return None
_DISTRESS = re.compile(
    r"\b(?:i(?: am|'m) (?:panicking|overwhelmed|distressed|grieving|upset|anxious)|i(?: am|'m) having a panic attack|"
    r"i (?:cannot|can't) cope|i feel (?:overwhelmed|distressed)|"
    r"(?:i )?feel ashamed(?: talking about it)?|(?:this|reading this) makes me anxious|"
    r"my bereavement is upsetting|"
    r"my (?:husband|wife|spouse|partner|mother|father) (?:has )?(?:died|passed away)|"
    r"i (?:have )?(?:just )?lost my job)\b",
    re.I,
)


def check_pre_model_controls(text: str) -> ControlResult:
    text = text.replace("’", "'")
    if _SELF_HARM.search(text):
        return ControlResult(
            matched=True,
            action="safety_stop",
            flag="possible_urgent_self_harm",
            reason="Possible urgent self-harm language detected before model execution.",
        )

    if _STOP.match(text) or _STOP_WITH_CONTEXT.search(text):
        return ControlResult(
            matched=True,
            action="stop",
            flag="pause_or_stop",
            reason="User explicitly requested to end this conversation.",
        )

    if _PAUSE.match(text):
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

    if _DISTRESS.search(text):
        return ControlResult(
            matched=True,
            action="care_pause",
            flag="safety_or_distress",
            reason="User disclosed distress or a significant life event; offer control over pacing.",
        )

    if _BREAK.search(text):
        return ControlResult(matched=True, action="pace_answer", flag="pause_or_stop", reason="User requested a break; preserve supported meaning but pause before confirmation.")

    if _REFUSAL.search(text):
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
    if _HELP.match(text) or (_HELP_CONTEXT.search(text) and not unrelated):
        return ControlResult(
            matched=True,
            action="explain",
            flag="help_request",
            reason="User requested help without a financial answer.",
        )

    if _ACCESSIBILITY_ONLY.match(text):
        return ControlResult(matched=True, action="support_control", flag="accessibility", reason="User clearly requested an accessible presentation without a financial answer.")

    if _ACCESSIBILITY.search(text):
        return ControlResult(
            matched=True,
            action="support_answer",
            flag="accessibility",
            reason="User requested a simpler or more accessible presentation; no financial fact inferred.",
        )

    if _HELP_WITH_ANSWER.search(text):
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
