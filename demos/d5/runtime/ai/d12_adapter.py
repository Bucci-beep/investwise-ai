from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class D12Resolution:
    candidate_option_id: str | None = None
    ordinary_bucket: str | None = None
    model_uncertain: bool = False
    semantic_undecided: bool = False
    reason: str | None = None
    source: str = "d12_ml"


class D12Resolver(Protocol):
    def resolve(self, text: str) -> D12Resolution:
        ...


class ExistingBoundedInterpreterAdapter:
    """Wrap the existing evaluated D12 BoundedQuestionInterpreter."""

    def __init__(self, interpreter: Any):
        self.interpreter = interpreter

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        return getattr(value, "value", str(value))

    def resolve(self, text: str) -> D12Resolution:
        result = self.interpreter.interpret(text)

        if bool(getattr(result, "semantic_undecided", False)):
            return D12Resolution(
                ordinary_bucket="Undecided",
                semantic_undecided=True,
                reason="semantic_undecided",
            )

        if bool(getattr(result, "model_uncertain", False)):
            return D12Resolution(
                ordinary_bucket=None,
                model_uncertain=True,
                reason="model_uncertainty",
            )

        deterministic_mapping = getattr(result, "deterministic_mapping", None)
        deterministic_status = self._enum_value(
            getattr(deterministic_mapping, "status", None)
        )

        if deterministic_status == "ambiguous":
            return D12Resolution(
                ordinary_bucket="Confusion",
                reason="explicit_durations_cross_d12_boundaries",
            )

        proposed = getattr(result, "proposed_value", None)
        if proposed:
            return D12Resolution(
                candidate_option_id=str(proposed),
                ordinary_bucket="Clarity",
                reason="option_identified",
            )

        intent_decision = getattr(result, "intent_decision", None)
        action = self._enum_value(getattr(intent_decision, "action", None))

        if action == "restate":
            return D12Resolution(
                ordinary_bucket="Confusion",
                reason="semantic_confusion",
            )

        return D12Resolution(
            ordinary_bucket=None,
            model_uncertain=True,
            reason="unresolved_d12_interpretation",
        )
