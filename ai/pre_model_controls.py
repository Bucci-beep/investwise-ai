from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlResult:
    matched: bool
    action: str | None = None
    flag: str | None = None
    reason: str | None = None


_PAUSE = re.compile(r"\b(pause|stop|end this|quit)\b", re.I)
_REFUSAL = re.compile(r"\b(prefer not to answer|rather not answer|don't want to answer|do not want to answer)\b", re.I)
_UNDECIDED = re.compile(r"^\s*(not sure|i don't know|i do not know|can't decide|cannot decide)\s*[.!]?\s*$", re.I)
_CORRECTION = re.compile(r"\b(change my answer|correct my answer|that was wrong|i meant something else)\b", re.I)

# Deliberately high precision. The rule layer should not try to diagnose distress.
_SELF_HARM = re.compile(
    r"\b(kill myself|end my life|suicide|hurt myself|harm myself)\b",
    re.I,
)


def check_pre_model_controls(text: str) -> ControlResult:
    if _SELF_HARM.search(text):
        return ControlResult(
            matched=True,
            action="safety_stop",
            flag="possible_urgent_self_harm",
            reason="Possible urgent self-harm language detected before model execution.",
        )

    if _PAUSE.search(text):
        return ControlResult(
            matched=True,
            action="pause",
            flag="pause_or_stop",
            reason="User requested pause or stop.",
        )

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

    if _UNDECIDED.match(text):
        return ControlResult(
            matched=True,
            action="undecided",
            flag=None,
            reason="User is explicitly undecided.",
        )

    return ControlResult(matched=False)
