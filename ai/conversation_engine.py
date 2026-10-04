from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ai.conversation_state import AnswerRecord, AnswerStatus, ConversationState
from ai.d12_adapter import D12Resolver
from ai.deterministic_conversation import detect_d12_independent_flags, match_fixed_choice
from ai.pre_model_controls import check_pre_model_controls
from ai.question_spec import BusinessSpec


OPTIONAL_CONTEXT = {"D2", "D3", "D8", "D9", "D10"}
MAX_CLARIFICATION_TURNS = 3


class ConversationEngine:
    """
    Complete D1-D15 shell.

    Scope rules implemented here:
      * safety/control checks execute before D12 ML
      * all answers remain candidates until explicit per-answer confirmation
      * no middle default for undecided replies
      * D12 uses deterministic explicit boundaries, then the existing evaluated ML path
      * D13-D15 remain deterministic/fixed-choice for the MVP
      * corrections create a new profile version and invalidate final accuracy/save consent
      * final accuracy confirmation and save consent are separate
      * audit events are transient and not included in the saved profile
    """

    def __init__(
        self,
        business_spec: BusinessSpec,
        d12_resolver: D12Resolver | None = None,
        store: Any | None = None,
    ):
        self.spec = business_spec
        self.d12_resolver = d12_resolver
        self.store = store or InMemoryProfileStore()

        self.state = ConversationState(
            question_order=list(self.spec.question_order),
            answers={
                qid: AnswerRecord(question_id=qid)
                for qid in self.spec.question_order
            },
        )

    @property
    def current_question(self):
        qid = self.state.current_question_id
        return None if qid is None else self.spec.question(qid)

    def _record(self, qid: str) -> AnswerRecord:
        return self.state.answers[qid]

    def _advance(self) -> None:
        while self.state.current_index < len(self.state.question_order):
            qid = self.state.question_order[self.state.current_index]
            status = self._record(qid).status
            if status in {AnswerStatus.CONFIRMED, AnswerStatus.DECLINED}:
                self.state.current_index += 1
                continue
            break

    def submit_answer(self, text: str) -> dict[str, Any]:
        if self.state.stopped_for_safety:
            return {"action": "safety_stop", "reason": "conversation_already_stopped"}

        if self.state.paused:
            return {"action": "paused", "reason": "conversation_is_paused"}

        qid = self.state.current_question_id
        if qid is None:
            return {"action": "final_review", "reason": "all_questions_handled"}

        record = self._record(qid)
        record.raw_text = text

        # 1. Safety and controls always run before D12 ML.
        control = check_pre_model_controls(text)
        if control.matched:
            if control.flag:
                record.independent_flags.add(control.flag)

            self.state.log(
                "pre_model_control",
                qid,
                action=control.action,
                flag=control.flag,
                reason=control.reason,
            )

            if control.action == "safety_stop":
                self.state.stopped_for_safety = True
                self.state.active_holds.add("safety_or_distress")
                return {
                    "action": "safety_stop",
                    "flag": control.flag,
                    "reason": control.reason,
                    "simulated_handoff": True,
                }

            if control.action == "pause":
                self.state.paused = True
                record.status = AnswerStatus.PAUSED
                return {"action": "pause", "reason": control.reason}

            if control.action == "stop":
                self.state.stopped_for_safety = True
                return {"action": "stop", "reason": control.reason}

            if control.action == "pause_for_distress":
                self.state.paused = True
                record.status = AnswerStatus.PAUSED
                return {"action": "pause_for_distress", "reason": control.reason}

            if control.action in {"reoffer_options", "explain_scope_and_reoffer"}:
                return {
                    "action": control.action,
                    "reason": control.reason,
                    "control_type": control.control_type.value,
                }

            if control.action in {"decline", "accept_refusal"}:
                record.status = AnswerStatus.DECLINED
                record.reason = control.reason
                self.state.bump_version()
                self.state.log("answer_declined", qid)
                self._advance()
                return {
                    "action": "continue",
                    "handled_as": "declined",
                    "next_question_id": self.state.current_question_id,
                }

            if control.action == "correction_request":
                return {
                    "action": "request_correction_target",
                    "reason": control.reason,
                }

            if control.action == "undecided":
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.UNDECIDED,
                    "explicit_user_undecided",
                )

        # 2. Exact fixed choices are deterministic for all questions.
        exact = match_fixed_choice(self.spec, qid, text)
        if exact is not None:
            return self._set_candidate(
                qid,
                exact.candidate_option_id,
                source=exact.source,
                reason=exact.reason,
                flags=set(exact.independent_flags),
            )

        # 3. D12 independent dependency flags may be detected before ML,
        # but they never assign a financial category.
        if qid == "D12":
            d12_flags = detect_d12_independent_flags(text)

            # 4. Only D12 gets the evaluated free-text ML path in this MVP.
            # The existing BoundedQuestionInterpreter owns the sequence:
            # intent classifier -> conformal clarity gate -> deterministic
            # explicit-horizon mapper -> option classifier when needed.
            if self.d12_resolver is None:
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.CONFUSION,
                    "d12_free_text_ml_not_configured",
                )

            resolved = self.d12_resolver.resolve(text)

            if resolved.semantic_undecided or resolved.ordinary_bucket == "Undecided":
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.UNDECIDED,
                    resolved.reason or "semantic_undecided",
                )

            if resolved.model_uncertain:
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.CONFUSION,
                    resolved.reason or "model_uncertainty",
                    model_uncertain=True,
                )

            if resolved.ordinary_bucket == "Confusion":
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.CONFUSION,
                    resolved.reason or "semantic_confusion",
                )

            if resolved.candidate_option_id:
                return self._set_candidate(
                    qid,
                    resolved.candidate_option_id,
                    source=resolved.source,
                    reason=resolved.reason,
                    flags=d12_flags,
                )

            return self._mark_unresolved(
                qid,
                AnswerStatus.CONFUSION,
                "d12_unresolved",
            )

        # D1-D11 and D13-D15 remain fixed-choice/deterministic until evaluated ML exists.
        return self._mark_unresolved(
            qid,
            AnswerStatus.CONFUSION,
            "free_text_not_enabled_for_this_question_in_mvp",
        )

    def _set_candidate(
        self,
        qid: str,
        option_id: str | None,
        *,
        source: str | None,
        reason: str | None,
        flags: set[str] | None = None,
    ) -> dict[str, Any]:
        if option_id is None:
            raise ValueError("candidate option cannot be None")

        option = self.spec.option(qid, option_id)
        record = self._record(qid)
        record.status = AnswerStatus.CANDIDATE
        record.candidate_option_id = option_id
        record.selected_option_id = None
        record.selected_option_label = None
        record.source = source
        record.reason = reason
        record.independent_flags = set(flags or set())

        if "same_money_dependency" in record.independent_flags:
            self.state.active_holds.add("same_money_dependency")

        self.state.log(
            "candidate_created",
            qid,
            candidate_option_id=option_id,
            source=source,
            reason=reason,
            flags=sorted(record.independent_flags),
        )

        return {
            "action": "confirm",
            "question_id": qid,
            "candidate_option_id": option_id,
            "candidate_option_label": option.label,
            "confirmation_text": self._confirmation_text(qid, option.label),
            "recorded": False,
        }

    def _confirmation_text(self, qid: str, label: str) -> str:
        if qid == "D12":
            return f"You said the earliest realistic need is {label}. Is that right?"
        return f"You selected {label}. Is that right?"

    def confirm_current(self, confirmed: bool) -> dict[str, Any]:
        qid = self.state.current_question_id
        if qid is None:
            raise RuntimeError("No current question to confirm")

        record = self._record(qid)
        if record.status != AnswerStatus.CANDIDATE:
            raise RuntimeError(f"{qid} has no candidate awaiting confirmation")

        if not confirmed:
            record.status = AnswerStatus.PENDING
            record.candidate_option_id = None
            record.reason = "candidate_rejected_by_user"
            record.source = None
            self.state.log("candidate_rejected", qid)
            return {"action": "reask", "question_id": qid}

        option = self.spec.option(qid, record.candidate_option_id)
        record.status = AnswerStatus.CONFIRMED
        record.selected_option_id = option.id
        record.selected_option_label = option.label
        record.confirmed_version = self.state.profile_version
        record.candidate_option_id = None

        self.state.bump_version()
        record.confirmed_version = self.state.profile_version

        self.state.log(
            "answer_confirmed",
            qid,
            selected_option_id=record.selected_option_id,
            source=record.source,
        )

        self._refresh_cross_question_holds()
        self._advance()

        return {
            "action": "continue" if self.state.current_question_id else "final_review",
            "confirmed_question_id": qid,
            "next_question_id": self.state.current_question_id,
            "profile_version": self.state.profile_version,
        }

    def _mark_unresolved(
        self,
        qid: str,
        status: AnswerStatus,
        reason: str,
        *,
        model_uncertain: bool = False,
    ) -> dict[str, Any]:
        record = self._record(qid)
        record.status = status
        record.reason = reason
        record.clarification_turns += 1
        record.candidate_option_id = None
        record.selected_option_id = None
        record.selected_option_label = None

        self.state.log(
            "clarification_required",
            qid,
            status=status.value,
            reason=reason,
            clarification_turns=record.clarification_turns,
            model_uncertain=model_uncertain,
        )

        if record.clarification_turns >= MAX_CLARIFICATION_TURNS:
            return {
                "action": "finish_incomplete_or_pause",
                "question_id": qid,
                "reason": reason,
                "clarification_turns": record.clarification_turns,
                "profile_complete": False,
            }

        return {
            "action": "clarify",
            "question_id": qid,
            "bucket": "Undecided" if status == AnswerStatus.UNDECIDED else "Confusion",
            "reason": reason,
            "clarification_turns": record.clarification_turns,
            "model_uncertain": model_uncertain,
        }

    def correct_answer(self, question_id: str) -> dict[str, Any]:
        if question_id not in self.state.answers:
            raise KeyError(question_id)

        self.state.bump_version()

        targets = {question_id}

        # Scope/dependency corrections can alter D12 and D13 interpretation.
        if question_id in {"D1", "D3", "D4", "D5", "D6", "D7"}:
            targets.update({"D12", "D13"})

        for qid in targets:
            record = self._record(qid)
            record.status = AnswerStatus.PENDING
            record.raw_text = None
            record.candidate_option_id = None
            record.selected_option_id = None
            record.selected_option_label = None
            record.source = None
            record.reason = "invalidated_by_correction"
            record.independent_flags.clear()
            record.confirmed_version = None

        earliest = min(self.state.question_order.index(qid) for qid in targets)
        self.state.current_index = earliest
        self.state.active_holds.clear()
        self._refresh_cross_question_holds()

        self.state.log(
            "correction_started",
            question_id,
            invalidated_questions=sorted(targets),
        )

        return {
            "action": "reask",
            "question_id": self.state.current_question_id,
            "invalidated_questions": sorted(targets),
            "profile_version": self.state.profile_version,
        }

    def acknowledge_preference_comfort_tension(self) -> None:
        self.state.active_holds.discard("d14_d15_tension_unacknowledged")
        self.state.log("tension_acknowledged", "D14")

    def clear_dependency_hold(self) -> None:
        self.state.active_holds.discard("same_money_dependency")
        self.state.active_holds.discard("emergency_long_horizon_dependency")
        self.state.log("dependency_hold_cleared")

    def _refresh_cross_question_holds(self) -> None:
        # Recompute only the specific consistency controls in the agreed MVP scope.
        self.state.active_holds.discard("emergency_long_horizon_dependency")
        self.state.active_holds.discard("d14_d15_tension_unacknowledged")

        d1 = self._record("D1").selected_option_id
        d12 = self._record("D12").selected_option_id
        d14 = self._record("D14").selected_option_id
        d15 = self._record("D15").selected_option_id

        if d1 == "D1_EMERGENCIES" and d12 == "D12_LONG":
            self.state.active_holds.add("emergency_long_horizon_dependency")

        if d14 == "D14_GREATER" and d15 == "D15_UNACCEPTABLE":
            self.state.active_holds.add("d14_d15_tension_unacknowledged")

    def completion_report(self) -> dict[str, Any]:
        missing_required: list[str] = []
        handled: list[str] = []

        for qid in self.state.question_order:
            record = self._record(qid)
            if record.status in {AnswerStatus.CONFIRMED, AnswerStatus.DECLINED}:
                handled.append(qid)

            if qid in OPTIONAL_CONTEXT:
                continue

            if record.status != AnswerStatus.CONFIRMED:
                missing_required.append(qid)

        return {
            "all_15_handled": len(handled) == 15,
            "handled_questions": handled,
            "missing_required": missing_required,
            "active_holds": sorted(self.state.active_holds),
            "eligible_for_final_accuracy_confirmation": (
                len(handled) == 15
                and not missing_required
                and not self.state.active_holds
                and not self.state.stopped_for_safety
            ),
        }

    def final_playback(self) -> dict[str, Any]:
        answers = []
        for qid in self.state.question_order:
            record = self._record(qid)
            answers.append(
                {
                    "question_id": qid,
                    "status": record.status.value,
                    "selected_option_id": record.selected_option_id,
                    "selected_option_label": record.selected_option_label,
                }
            )

        return {
            "profile_version": self.state.profile_version,
            "answers": answers,
            "completion": self.completion_report(),
        }

    def confirm_final_accuracy(self, confirmed: bool) -> dict[str, Any]:
        report = self.completion_report()
        if not report["eligible_for_final_accuracy_confirmation"]:
            return {
                "accepted": False,
                "reason": "profile_not_eligible_for_final_accuracy_confirmation",
                "completion": report,
            }

        if not confirmed:
            self.state.final_accuracy_version = None
            return {"accepted": False, "reason": "accuracy_not_confirmed"}

        self.state.final_accuracy_version = self.state.profile_version
        self.state.save_consent_version = None
        self.state.log("final_accuracy_confirmed")

        return {
            "accepted": True,
            "profile_version": self.state.profile_version,
            "next_action": "request_separate_save_consent",
        }

    def give_save_consent(self, consent: bool) -> dict[str, Any]:
        if self.state.final_accuracy_version != self.state.profile_version:
            return {
                "accepted": False,
                "reason": "current_profile_accuracy_not_confirmed",
            }

        if not consent:
            self.state.save_consent_version = None
            return {"accepted": False, "reason": "save_consent_not_given"}

        self.state.save_consent_version = self.state.profile_version
        self.state.log("save_consent_given")

        return {
            "accepted": True,
            "profile_version": self.state.profile_version,
            "next_action": "save",
        }

    def save(self) -> dict[str, Any]:
        report = self.completion_report()

        # Deterministic transaction gate immediately before write.
        if not report["eligible_for_final_accuracy_confirmation"]:
            return {"saved": False, "reason": "completion_gate_failed", "completion": report}

        if self.state.final_accuracy_version != self.state.profile_version:
            return {"saved": False, "reason": "accuracy_confirmation_stale_or_missing"}

        if self.state.save_consent_version != self.state.profile_version:
            return {"saved": False, "reason": "save_consent_stale_or_missing"}

        profile = self._persistent_profile_payload()
        receipt = self.store.save(profile)

        self.state.saved_version = self.state.profile_version
        self.state.log("profile_saved", receipt=receipt)

        return {"saved": True, "receipt": receipt, "profile": profile}

    def _persistent_profile_payload(self) -> dict[str, Any]:
        # Audit is deliberately excluded. Only confirmed business inputs plus consent metadata persist.
        confirmed_answers = {
            qid: {
                "option_id": record.selected_option_id,
                "option_label": record.selected_option_label,
            }
            for qid, record in self.state.answers.items()
            if record.status == AnswerStatus.CONFIRMED
        }

        declined = [
            qid
            for qid, record in self.state.answers.items()
            if record.status == AnswerStatus.DECLINED
        ]

        return {
            "business_spec_version": self.spec.version,
            "profile_version": self.state.profile_version,
            "confirmed_answers": confirmed_answers,
            "declined_questions": declined,
            "final_accuracy_confirmed": True,
            "explicit_save_consent": True,
        }


class InMemoryProfileStore:
    def __init__(self):
        self.saved_profiles: list[dict[str, Any]] = []

    def save(self, profile: dict[str, Any]) -> str:
        self.saved_profiles.append(profile)
        return f"in-memory-{len(self.saved_profiles)}"
