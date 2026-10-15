"""Disabled-by-default D5 experiment; the normal application never imports it.

Only the issuer-verified contract decision can create D5 financial state. The
receipt is transient and bound to the current input, D5 record and profile
version. Reading profile evidence is deliberately read-only: a new or stale
input requires an explicit D5 answer submission, never a hidden model call.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from ai.bounded_answers import BoundedAnswer
from ai.conversation_engine import ConversationEngine
from ai.conversation_state import AnswerStatus
from ai.d5_single_interpreter_contract import D5ContractDecision
from ai.question_followups import NEUTRAL_FOLLOWUPS, question_explanation


COMPLETE_ANSWER_PROMPT = (
    "Please describe what currently pays your housing, food and bills in one "
    "complete reply. You can correct the earlier answer, leave this unresolved "
    "or pause."
)


@dataclass(frozen=True)
class _Receipt:
    decision: D5ContractDecision
    text: str
    profile_version: int
    record_sha256: str


class D5SingleInterpreterEngine(ConversationEngine):
    """Opt-in adapter retaining the existing 15-question application shell."""

    def __init__(self, business_spec, *, d5_interpreter, question_resolver, store=None):
        if question_resolver is None:
            raise ValueError("The unchanged resolver for the other 14 questions is required")
        super().__init__(business_spec, question_resolver=question_resolver, store=store)
        self.d5_interpreter = d5_interpreter
        self._d5_receipt: _Receipt | None = None
        self._d5_active_resolution = None

    def _d5_record_digest(self) -> str:
        record = self._record("D5")
        # Pure care/help flags may change without changing a proposed answer.
        # Financial details, provenance and confirmation state may not.
        value = {
            "status": record.status.value, "raw_text": record.raw_text,
            "candidate": record.candidate_option_id, "selected": record.selected_option_id,
            "label": record.selected_option_label, "source": record.source,
            "confirmed_version": record.confirmed_version,
            "context_details": record.context_details,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _bind_decision(self, decision, text):
        self._d5_receipt = _Receipt(decision, text, self.state.profile_version,
                                    self._d5_record_digest())

    def _valid_receipt(self, text, option_id=None) -> bool:
        receipt = self._d5_receipt
        return bool(
            receipt is not None and receipt.text == text
            and receipt.profile_version == self.state.profile_version
            and receipt.record_sha256 == self._d5_record_digest()
            and self.d5_interpreter.verify_decision(
                receipt.decision, text, spec=self.spec, option_id=option_id)
        )

    def _register_actual_dependency(self, text, qid):
        # The inherited method extracts a financial fact from wording before
        # interpretation. This experiment may not create such facts from a
        # rejected D5 reply. All other question routes remain unchanged.
        if qid != "D5":
            return super()._register_actual_dependency(text, qid)

    def _clear_d5_financial_state(self):
        record = self._record("D5")
        if (record.candidate_option_id is not None or record.selected_option_id is not None
                or self.state.final_accuracy_version is not None
                or self.state.save_consent_version is not None):
            self.state.bump_version()
        record.status = AnswerStatus.PENDING
        record.candidate_option_id = None
        record.selected_option_id = None
        record.selected_option_label = None
        record.confirmed_version = None
        record.context_details.clear()
        record.source = None
        self._d5_receipt = None
        self._d5_funding_followup_context = None

    def _d5_unresolved(self, *, reason, bucket=None, outcome="rejection", care=False,
                       support=False, pace=False, previous_raw=None):
        record = self._record("D5")
        record.reason = reason
        record.status = AnswerStatus.UNDECIDED if bucket == "Undecided" else AnswerStatus.CONFUSION
        # Fixed authored wording only. No rejected facts or model-authored
        # missing-information strings enter a customer-facing follow-up.
        if care:
            result = self._care_pause("D5", ordinary_bucket=bucket)
        elif pace:
            result = self._pacing_pause("D5", previous_raw=previous_raw, ordinary_bucket=bucket)
        elif support:
            result = self._support_before_confirmation("D5", ordinary_bucket=bucket)
        else:
            issued = self._issue_clarification_request("D5")
            result = {"action": "clarify" if issued else "finish_incomplete_or_pause",
                      "question_id": "D5", "reason": reason,
                      "clarification_turns": record.clarification_turns,
                      "profile_complete": False, "recorded": False}
            if issued:
                result["followup"] = (NEUTRAL_FOLLOWUPS["D5"] if bucket == "Undecided"
                                      else question_explanation(self.spec, "D5")
                                      if bucket == "Confusion" else COMPLETE_ANSWER_PROMPT)
            self.state.log("clarification_required", "D5", reason=reason,
                           factual_request_issued=issued, outcome=outcome,
                           clarification_turns=record.clarification_turns)
        result.update(bucket=bucket, model_uncertain=outcome in {"abstention", "rejection"},
                      contract_outcome=outcome, recorded=False)
        return result

    def _submit_ml_answer(self, qid, text, care, support, pace, teachback, previous_raw,
                          flags, *, explicit_selection=None):
        if qid != "D5":
            return super()._submit_ml_answer(qid, text, care, support, pace, teachback,
                                            previous_raw, flags, explicit_selection=explicit_selection)
        self._clear_d5_financial_state()
        # Exactly one interpretation for this eligible submitted answer. An
        # exception propagates: no retry, reference fallback or state commit.
        resolved = self.d5_interpreter.resolve("D5", text)
        if resolved.diagnostics.get("backend_failed"):
            raise RuntimeError("D5 interpreter stopped after transport failure")
        decision = getattr(resolved, "contract_decision", None)
        if decision is None or not self.d5_interpreter.verify_decision(decision, text, spec=self.spec):
            return self._d5_unresolved(reason="missing_or_untrusted_contract_decision",
                                       care=care, support=support, pace=pace, previous_raw=previous_raw)
        self.state.log("d5_contract_decision", "D5", outcome=decision.outcome,
                       reason=decision.reason, input_sha256=decision.input_sha256,
                       proposal_sha256=decision.proposal_sha256,
                       contract_sha256=decision.contract_sha256,
                       spec_sha256=decision.spec_sha256)
        if decision.outcome != "financial_candidate":
            bucket = (decision.interpretation.semantic_bucket.title()
                      if decision.outcome == "dialogue_only" else None)
            return self._d5_unresolved(reason=decision.reason, bucket=bucket,
                                       outcome=decision.outcome, care=care, support=support,
                                       pace=pace, previous_raw=previous_raw)
        candidate = resolved.candidate_option_id
        if candidate is None or not self.d5_interpreter.verify_decision(
                decision, text, spec=self.spec, option_id=candidate):
            return self._d5_unresolved(reason="candidate_contract_disagreement")
        if explicit_selection is not None and candidate != explicit_selection:
            return self._d5_unresolved(reason="selection_and_text_disagreement")
        self._record("D5").source = "d5_single_interpreter_v1"
        self._bind_decision(decision, text)
        self._d5_active_resolution = resolved
        try:
            result = self._set_candidate("D5", candidate, source="d5_single_interpreter_v1",
                                         reason=decision.reason, flags=flags)
        finally:
            self._d5_active_resolution = None
        if result.get("action") != "confirm":
            return result
        result = self._pace_supported_answer("D5", result, care, support, pace)
        result.update(bucket="Clarity", contract_outcome="financial_candidate")
        return result

    def _validate_d5_resolution(self, text, resolved):
        decision = getattr(resolved, "contract_decision", None)
        candidate = resolved.candidate_option_id
        if (decision is None or self._d5_receipt is None
                or decision is not self._d5_receipt.decision
                or decision.outcome != "financial_candidate"
                or candidate is None or not self._valid_receipt(text, candidate)):
            return BoundedAnswer(None, reason="stale_or_untrusted_d5_decision")
        return BoundedAnswer("Clarity", option_id=candidate, reason=decision.reason)

    def _set_candidate(self, qid, option_id, *, source, reason, flags=None):
        if qid != "D5":
            return super()._set_candidate(qid, option_id, source=source, reason=reason, flags=flags)
        if source == "explicit_user_choice":
            # Visible choice recovery remains deterministic and is not counted
            # as free-text semantics or a model-generated financial fact.
            self._d5_receipt = None
            self._record("D5").context_details.clear()
            return super()._set_candidate(qid, option_id, source=source, reason=reason, flags=flags)
        resolved = self._d5_active_resolution
        text = self._record("D5").raw_text or ""
        if (resolved is None or resolved.candidate_option_id != option_id
                or self._validate_d5_resolution(text, resolved).option_id != option_id):
            self._clear_d5_financial_state()
            return self._d5_unresolved(reason="stale_or_untrusted_d5_candidate")
        decision = self._d5_receipt.decision
        result = super()._set_candidate(qid, option_id, source=source, reason=reason, flags=flags)
        self._bind_decision(decision, text)
        return result

    def _profile_evidence(self, qid, text):
        if qid != "D5":
            return super()._profile_evidence(qid, text)
        receipt = self._d5_receipt
        if (self.state.paused or self.state.stopped_for_safety
                or "user_stop" in self.state.active_holds
                or receipt is None or receipt.decision.outcome != "financial_candidate"
                or not self._valid_receipt(text)):
            return BoundedAnswer(None, reason="d5_complete_resubmission_required",
                                 followup=COMPLETE_ANSWER_PROMPT)
        return BoundedAnswer("Clarity", option_id=receipt.decision.interpretation.option_id,
                             reason=receipt.decision.reason)

    def confirm_current(self, confirmed):
        qid = self.state.current_question_id
        receipt = self._d5_receipt
        if (qid == "D5" and not self.state.paused and not self.state.stopped_for_safety
                and self._record("D5").status == AnswerStatus.CANDIDATE
                and self._record("D5").source != "explicit_user_choice"
                and not self._valid_receipt(self._record("D5").raw_text or "",
                                            self._record("D5").candidate_option_id)):
            self._clear_d5_financial_state()
            return self._d5_unresolved(reason="stale_or_untrusted_d5_confirmation")
        result = super().confirm_current(confirmed)
        if qid == "D5" and result.get("action") in {"continue", "final_review"} and receipt is not None:
            self._bind_decision(receipt.decision, receipt.text)
        elif qid == "D5" and result.get("action") == "reask":
            self._d5_receipt = None
        return result

    def correct_answer(self, question_id):
        result = super().correct_answer(question_id)
        if "D5" in result["invalidated_questions"]:
            self._d5_receipt = None
        return result
