from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import Any

from ai.conversation_state import AnswerRecord, AnswerStatus, ConversationState
from ai.d12_adapter import D12Resolver
from ai.deterministic_conversation import detect_d12_independent_flags, match_fixed_choice
from ai.pre_model_controls import check_pre_model_controls
from ai.question_spec import BusinessSpec
from ai.bounded_answers import BoundedAnswer, interpret_bounded_answer
from ai.horizon_mapper import HorizonMappingStatus, extract_supported_need_clause, map_explicit_horizon

_ENGINE_SOURCE_STAMP = Path(__file__).stat().st_mtime_ns


OPTIONAL_CONTEXT = {"D2", "D3", "D8", "D9", "D10"}
CONDITIONAL_CONTEXT = {"D4", "D5", "D6", "D7"}
MAX_CLARIFICATION_TURNS = 3
UNDERSTANDING_EXPLANATION = "An investment can fall in value. Recovery is not guaranteed, and some or all of the original money may be permanently lost."
UNDERSTANDING_FOLLOWUP = "In your own words, what could happen to the original money, and is recovery guaranteed?"
CARE_PACING = "Thank you for telling me. We can take this at your pace. Would you like to continue, pause or stop?"
MATERIAL_HOLDS = {"same_pot_backup_conflict", "usable_backup_check", "need_before_access", "essential_funding_timing"}
DEPENDENCY_HOLDS = {"same_money_dependency", "emergency_long_horizon_dependency", "living_costs_horizon_dependency"}


class ConversationEngine:
    """
    Complete D1-D15 shell.

    Scope rules implemented here:
      * safety/control checks execute before every question's ML
      * all answers remain candidates until explicit per-answer confirmation
      * no middle default for undecided replies
      * the app uses all-question ML with intent and fixed-option conformal gates
      * business constraints may veto a proposal but never supply an ML fallback
      * archived hybrid mode is retained only for comparison/policy regressions
      * corrections create a new profile version and invalidate final accuracy/save consent
      * final accuracy confirmation and save consent are separate
      * audit events are transient and not included in the saved profile
    """

    def __init__(
        self,
        business_spec: BusinessSpec,
        d12_resolver: D12Resolver | None = None,
        store: Any | None = None,
        question_resolver: Any | None = None,
    ):
        self.spec = business_spec
        self.d12_resolver = d12_resolver
        self.question_resolver = question_resolver
        self.store = store or InMemoryProfileStore()
        self._pending_material_check: dict[str, Any] | None = None

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

    @property
    def pending_material_check(self) -> dict[str, Any] | None:
        """Only a versioned, unconfirmed proposal; never part of saved answers."""
        return dict(self._pending_material_check) if self._pending_material_check else None

    def _advance(self) -> None:
        while self.state.current_index < len(self.state.question_order):
            qid = self.state.question_order[self.state.current_index]
            if qid == "D11" and "understanding_hold" in self.state.active_holds:
                break
            status = self._record(qid).status
            if status in {AnswerStatus.CONFIRMED, AnswerStatus.DECLINED}:
                self.state.current_index += 1
                continue
            break

    @staticmethod
    def _actual_same_pot_dependency(text: str) -> bool:
        """Flag an actual necessary-cost assertion; never infer a category."""
        raw = text.replace("’", "'")
        scope = r"(?:this (?:same |exact )?(?:money|pot)|these funds|money we are discussing)"
        costs = r"(?:rent|essential(?:s| costs)?|necessary (?:living )?costs|living costs|bills|repayments|food)"
        actual_need = re.search(
            r"\bi\s+(?:(?:actually|currently|really|now)\s+)?(?:need|use|rely on|depend on)\s+" + scope
            + r"\b[^.!?]{0,100}\b" + costs + r"\b|\b(?:my|our)\s+" + costs
            + r"\b[^.!?]{0,60}\b(?:is|are) (?:currently )?paid (?:from|using|with)\s+" + scope + r"\b",
            raw, re.I,
        )
        if actual_need is None:
            return False
        explicitly_actual = bool(re.search(r"\b(?:in reality|in real life|actually|currently)\b", actual_need.group(0), re.I)) or bool(re.search(r"\b(?:in reality|in real life)\s*$", raw[:actual_need.start()], re.I))
        if re.search(r"\b(?:if|hypothetical|suppos\w*|assuming|assume|imagin\w*|would|covered[- ]cost example|this (?:covered[- ]cost )?example)\b", raw, re.I) and not explicitly_actual:
            return False
        return True

    def _register_actual_dependency(self, text: str, qid: str | None) -> None:
        if not self._actual_same_pot_dependency(text):
            return
        if qid == "D12":
            explicit_need = map_explicit_horizon(extract_supported_need_clause(text) or text)
            if explicit_need.status == HorizonMappingStatus.MAPPED:
                # This question already owns that supplied timing, followed
                # by the required practical-impact question. Avoid a duplicate
                # check; earlier unresolved dependence flags remain untouched.
                return
        record = self._record(qid or "D12")
        evidence = " ".join(text.split())
        is_new = self._details(record).get("material.actual_dependency_evidence") != evidence
        record.independent_flags.add("same_money_dependency")
        self._details(record)["material.actual_dependency_evidence"] = evidence
        self._details(record).pop("dependency.checked", None)
        self._details(self._record("D12")).pop("material.same_money_dependency.resolved", None)
        self.state.active_holds.add("same_money_dependency")
        self._pending_material_check = None
        if is_new:
            # A new material fact at review also invalidates an old playback
            # or save consent, while the financial conclusions stay intact.
            self.state.bump_version()
            self.state.log("actual_same_money_dependency", qid, text=text)

    def submit_answer(self, text: str, *, explicit_option_id: str | None = None) -> dict[str, Any]:
        if self.state.stopped_for_safety:
            return {"action": "safety_stop", "reason": "conversation_already_stopped"}

        control = check_pre_model_controls(text)
        if "user_stop" in self.state.active_holds and control.action != "safety_stop":
            return {"action": "stop", "reason": "conversation_already_stopped_by_user"}
        if control.action == "resume":
            return self.resume()

        # An explicit safety request still takes priority during a pause.
        if self.state.paused and control.action not in {"safety_stop", "stop", "pause"}:
            return {"action": "paused", "reason": "conversation_is_paused"}

        qid = self.state.current_question_id
        self._register_actual_dependency(text, qid)
        if qid is None:
            # Review and consent are still part of the conversation: controls
            # must run here even though there is no current financial question.
            if control.matched:
                self.state.log(
                    "pre_model_control",
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
                return {"action": "pause", "question_id": None, "reason": control.reason}
            if control.action == "stop":
                self.state.paused = True
                self.state.active_holds.add("user_stop")
                return {"action": "stop", "question_id": None, "reason": control.reason}
            if control.action == "care_pause":
                return self._care_pause(None)
            if control.action == "pace_answer":
                return self._pacing_pause(None)
            if control.action == "support_control":
                return self._support_before_confirmation(None, control_only=True)
            if control.action == "explain":
                return {
                    "action": "explain",
                    "question_id": None,
                    "bucket": "Confusion",
                    "flag": "help_request",
                    "reason": control.reason,
                    "help_prompt": "Would you like help reviewing your answers or changing one of them?",
                    "recorded": False,
                }

            return {"action": "final_review", "reason": "all_questions_handled"}

        record = self._record(qid)

        # Safety and explicit controls run before every question's ML models.
        if control.matched:
            if control.flag:
                record.independent_flags.add(control.flag)
                if control.flag == "accessibility":
                    record.independent_flags.add("accessibility_request")
            if control.action == "care_pause" and re.search(r"\bi (?:need|want) (?:a |to take a )?(?:break|moment)\b", text, re.I):
                record.independent_flags.add("pause_or_stop")

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
                return {"action": "pause", "question_id": qid, "reason": control.reason}
            if control.action == "stop":
                self.state.paused = True
                self.state.active_holds.add("user_stop")
                return {"action": "stop", "question_id": qid, "reason": control.reason}

            if control.action == "explain":
                if qid == "D11":
                    record.independent_flags.add("understanding_hold")
                    self.state.active_holds.add("understanding_hold")
                    self._details(record)["understanding.teachback_required"] = "true"
                return {
                    "action": "explain",
                    "question_id": qid,
                    "bucket": "Confusion",
                    "flag": "help_request",
                    "reason": control.reason,
                    "help_prompt": "Would you like me to explain the question and choices, or do you understand them but cannot answer yet?",
                    "clarification_turns": record.clarification_turns,
                    "recorded": False,
                }

            if control.action == "support_control":
                return self._support_before_confirmation(qid, control_only=True)

            if control.action == "pace_answer" and re.fullmatch(r"\s*i (?:need|want) (?:a |to take a )?(?:break|moment)(?: to think| please)?\s*[.!?]*\s*", text, re.I):
                return self._pacing_pause(qid, previous_raw=record.raw_text)

            if control.action == "decline":
                record.status = AnswerStatus.DECLINED
                record.reason = control.reason
                record.raw_text = text
                record.candidate_option_id = None
                record.selected_option_id = None
                record.selected_option_label = None
                record.confirmed_version = None
                record.source = None
                # Declining supplies a clear control choice, not a financial
                # value. Preserve material/safety controls, clear pending facts.
                retained_controls = {
                    key: value for key, value in self._details(record).items()
                    if key in {"understanding.teachback_required", "understanding.own_word_check_requested", "understanding.previous_belief"}
                }
                self._details(record).clear()
                self._details(record).update(retained_controls)
                self._pending_material_check = None
                self.state.bump_version()
                self.state.log("answer_declined", qid)
                self._advance()
                return {
                    "action": "continue",
                    "handled_as": "declined",
                    "control_meaning": "declined",
                    "bucket": "Clarity",
                    "candidate_option_id": None,
                    "candidate_option_label": None,
                    "next_question_id": self.state.current_question_id,
                    "recorded": False,
                }

            if control.action == "correction_request":
                return {
                    "action": "request_correction_target",
                    "reason": control.reason,
                }

            if control.action == "undecided" and self.question_resolver is None:
                record.raw_text = text
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.UNDECIDED,
                    "explicit_user_undecided",
                )

        previous_raw_text = record.raw_text
        record.raw_text = text
        ordinary_care = control.action == "care_pause"
        support_requested = control.action == "support_answer"
        pace_requested = control.action == "pace_answer"
        global_flags = detect_d12_independent_flags(text)
        if global_flags:
            record.independent_flags.update(global_flags)
            self.state.active_holds.update(global_flags)

        if record.clarification_turns >= MAX_CLARIFICATION_TURNS:
            if ordinary_care:
                return self._care_pause(qid)
            if pace_requested:
                return self._pacing_pause(qid, previous_raw=previous_raw_text)
            if support_requested:
                return {"action": "explain", "question_id": qid, "help_prompt": "We can explain this question or pause. Its financial meaning remains unresolved.", "recorded": False, "clarification_turns": record.clarification_turns}
            return {"action": "finish_incomplete_or_pause", "question_id": qid, "reason": "clarification_limit_reached", "clarification_turns": record.clarification_turns, "profile_complete": False}

        # After teaching a misconception, an option ID or a yes-only reply
        # cannot substitute for the requested own-word understanding check.
        teachback_required = (
            qid == "D11"
            and self._details(record).get("understanding.teachback_required") == "true"
        )

        # A visible shortcut is an explicit user selection. Typed labels and
        # IDs remain free text and use ML when the all-question resolver is on.
        if explicit_option_id is not None:
            option = self.spec.option(qid, explicit_option_id)
            if teachback_required:
                return self._understanding_recheck(qid)
            if qid in {"D3", "D10"} and option.id not in {"D3_NO_EXISTING_POT", "D10_ZERO"}:
                self._details(record)["pending_option_id"] = option.id
                result = self._mark_unresolved(qid, AnswerStatus.CONFUSION, "currency_and_range_required")
                result["followup"] = f"You selected '{option.label}'. Which currency is that range in? Please include the range and named currency together in your own words; an approximate range is enough. You may also skip this optional question."
                return result
            result = self._set_candidate(qid, option.id, source="explicit_user_choice", reason="visible_shortcut_selected", flags=global_flags)
            return self._pace_supported_answer(qid, result, ordinary_care, support_requested, pace_requested)

        # A currency-only reply can complete the immediately preceding amount
        # follow-up. Combine it with the retained band before invoking ML; the
        # option model is not expected to infer an amount band from "GBP".
        if qid in {"D3", "D10"} and self._details(record).get("pending_option_id"):
            completion = interpret_bounded_answer(self.spec, qid, text, self._details(record))
            if completion.option_id is not None:
                self._details(record).update(completion.context_details)
                result = self._set_candidate(
                    qid,
                    completion.option_id,
                    source="bounded_material_completion",
                    reason="currency_followup_completed",
                    flags=set(completion.flags) | global_flags,
                )
                result["bucket"] = "Clarity"
                return self._pace_supported_answer(qid, result, ordinary_care, support_requested, pace_requested)

        if self.question_resolver is not None:
            return self._submit_ml_answer(qid, text, ordinary_care, support_requested, pace_requested, teachback_required, previous_raw_text, global_flags)

        # 2. Exact fixed choices are deterministic for all questions.
        exact = match_fixed_choice(self.spec, qid, text)
        currency_question = qid in {"D3", "D10"} and any(
            o.id in {"D3_NO_EXISTING_POT", "D10_ZERO"}
            for o in self.spec.question(qid).options
        )
        if exact is not None and not teachback_required and not currency_question:
            result = self._set_candidate(
                qid,
                exact.candidate_option_id,
                source=exact.source,
                reason=exact.reason,
                flags=set(exact.independent_flags) | global_flags,
            )
            return self._pace_supported_answer(qid, result, ordinary_care, support_requested, pace_requested)

        if qid != "D12":
            resolution = interpret_bounded_answer(self.spec, qid, text, self._details(record))
            record.independent_flags.update(resolution.flags)
            self.state.active_holds.update(set(resolution.flags) & MATERIAL_HOLDS)
            self._details(record).update(resolution.context_details)
            if resolution.option_id is not None:
                if teachback_required and resolution.context_details.get("understanding.own_words_supported") != "true":
                    if ordinary_care:
                        return self._care_pause(qid)
                    if pace_requested:
                        return self._pacing_pause(qid, previous_raw=previous_raw_text)
                    if support_requested:
                        return self._support_before_confirmation(qid)
                    return self._understanding_recheck(qid)
                control_details = {
                    key: value for key, value in self._details(record).items()
                    if key in {"understanding.teachback_required", "understanding.previous_belief"}
                }
                record.context_details = dict(resolution.context_details)
                record.context_details.update(control_details)
                record.context_details.pop("pending_option_id", None)
                result = self._set_candidate(
                    qid, resolution.option_id,
                    source="bounded_business_rule",
                    reason=resolution.reason,
                    flags=set(resolution.flags) | global_flags,
                )
                return self._pace_supported_answer(qid, result, ordinary_care, support_requested, pace_requested)
            if ordinary_care:
                return self._care_pause(qid)
            if pace_requested:
                return self._pacing_pause(qid, previous_raw=previous_raw_text)
            if support_requested:
                return self._support_before_confirmation(qid)
            if teachback_required:
                return self._understanding_recheck(qid)
            result = self._mark_unresolved(
                qid,
                AnswerStatus.UNDECIDED if resolution.bucket == "Undecided" else AnswerStatus.CONFUSION,
                resolution.reason,
            )
            if resolution.followup:
                result["followup"] = resolution.followup
            return result

        # 3. D12 independent dependency flags may be detected before ML,
        # but they never assign a financial category.
        if qid == "D12":
            d12_flags = detect_d12_independent_flags(text)

            # Product availability, a maturity date or an ideal goal date
            # does not establish the earliest realistic need for this money.
            describes_access_or_goal = re.search(
                r"\b(?:matur\w*|fixed[- ]?(?:deposit|term)|locked|unlocked|access|available|"
                r"want to invest|plan to invest|investment goal|target date)\b", text, re.I,
            )
            describes_need = re.search(r"\b(?:need|required|necessary|living costs|bills|untouched)\b", text, re.I)
            if describes_access_or_goal and not describes_need:
                if ordinary_care:
                    return self._care_pause(qid)
                if pace_requested:
                    return self._pacing_pause(qid, previous_raw=previous_raw_text)
                if support_requested:
                    return self._support_before_confirmation(qid)
                result = self._mark_unresolved(qid, AnswerStatus.CONFUSION, "availability_or_goal_is_not_earliest_need")
                result["followup"] = "That tells me about access or your intended goal. What is the earliest point when you may realistically need any of this same money?"
                return result

            # 4. Only D12 gets the evaluated free-text ML path in this MVP.
            # The existing BoundedQuestionInterpreter owns the sequence:
            # intent classifier -> conformal clarity gate -> deterministic
            # explicit-horizon mapper -> option classifier when needed.
            if self.d12_resolver is None:
                if ordinary_care:
                    return self._care_pause(qid)
                if pace_requested:
                    return self._pacing_pause(qid, previous_raw=previous_raw_text)
                if support_requested:
                    return self._support_before_confirmation(qid)
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.CONFUSION,
                    "d12_free_text_ml_not_configured",
                )

            supported_need_clause = extract_supported_need_clause(text)
            resolved = self.d12_resolver.resolve(supported_need_clause or text)

            if resolved.semantic_undecided or resolved.ordinary_bucket == "Undecided":
                if ordinary_care:
                    return self._care_pause(qid)
                if pace_requested:
                    return self._pacing_pause(qid, previous_raw=previous_raw_text)
                if support_requested:
                    return self._support_before_confirmation(qid)
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.UNDECIDED,
                    resolved.reason or "semantic_undecided",
                )

            if resolved.model_uncertain:
                if ordinary_care:
                    return self._care_pause(qid)
                if pace_requested:
                    return self._pacing_pause(qid, previous_raw=previous_raw_text)
                if support_requested:
                    return self._support_before_confirmation(qid)
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.CONFUSION,
                    resolved.reason or "model_uncertainty",
                    model_uncertain=True,
                )

            if resolved.ordinary_bucket == "Confusion":
                if ordinary_care:
                    return self._care_pause(qid)
                if pace_requested:
                    return self._pacing_pause(qid, previous_raw=previous_raw_text)
                if support_requested:
                    return self._support_before_confirmation(qid)
                return self._mark_unresolved(
                    qid,
                    AnswerStatus.UNDECIDED if resolved.reason == "explicit_durations_cross_d12_boundaries" else AnswerStatus.CONFUSION,
                    resolved.reason or "semantic_confusion",
                )

            if resolved.candidate_option_id:
                result = self._set_candidate(
                    qid,
                    resolved.candidate_option_id,
                    source=resolved.source,
                    reason=resolved.reason,
                    flags=d12_flags | global_flags,
                )
                return self._pace_supported_answer(qid, result, ordinary_care, support_requested, pace_requested)

            if ordinary_care:
                return self._care_pause(qid)
            if pace_requested:
                return self._pacing_pause(qid, previous_raw=previous_raw_text)
            if support_requested:
                return self._support_before_confirmation(qid)
            return self._mark_unresolved(
                qid,
                AnswerStatus.CONFUSION,
                "d12_unresolved",
            )

        if ordinary_care:
            return self._care_pause(qid)
        if pace_requested:
            return self._pacing_pause(qid, previous_raw=previous_raw_text)

        if teachback_required:
            return self._understanding_recheck(qid)

        # Other bounded free-text interpretation is wired separately below.
        return self._mark_unresolved(
            qid,
            AnswerStatus.CONFUSION,
            "free_text_not_enabled_for_this_question_in_mvp",
        )

    def _submit_ml_answer(self, qid: str, text: str, care: bool, support: bool, pace: bool, teachback: bool, previous_raw: str | None, flags: set[str]) -> dict[str, Any]:
        """ML alone proposes financial options; business validation can veto."""
        record = self._record(qid)
        resolved = self.question_resolver.resolve(qid, text)
        trace = {
            "source": resolved.source,
            "semantic_bucket": resolved.ordinary_bucket,
            "intent_set": sorted(resolved.intent_set),
            "option_set": sorted(resolved.option_set),
            "intent_probabilities": resolved.intent_probabilities,
            "option_probabilities": resolved.option_probabilities,
            "model_uncertain": resolved.model_uncertain,
            "reason": resolved.reason,
        }
        self.state.log("ml_interpretation", qid, **trace)
        record.source = resolved.source

        def held_or_unresolved(reason: str, *, undecided: bool = False, uncertain: bool = False, followup: str | None = None):
            if care:
                return self._care_pause(qid)
            if pace:
                return self._pacing_pause(qid, previous_raw=previous_raw)
            if support:
                return self._support_before_confirmation(qid)
            if teachback:
                result = self._understanding_recheck(qid)
            else:
                result = self._mark_unresolved(qid, AnswerStatus.UNDECIDED if undecided else AnswerStatus.CONFUSION, reason, model_uncertain=uncertain)
            if followup:
                result["followup"] = followup
            if uncertain:
                # An ambiguous conformal set does not establish Confusion or
                # Undecided. State status describes dialogue handling only.
                result["bucket"] = None
                result["handling"] = "clarification"
            result["ml_evidence"] = trace
            return result

        if resolved.candidate_option_id is None:
            return held_or_unresolved(resolved.reason or "ml_unresolved", undecided=resolved.semantic_undecided or resolved.ordinary_bucket == "Undecided", uncertain=resolved.model_uncertain)

        # Legacy interpretation is a validation oracle here, never a fallback
        # proposal: disagreement or explicit uncertainty blocks ML acceptance.
        validation = interpret_bounded_answer(self.spec, qid, text, self._details(record))
        unknown_wording = {"missing_material_detail", "no_supported_answer", "unsupported_meaning"}
        if validation.option_id is None and validation.reason not in unknown_wording:
            return held_or_unresolved("business_validation_" + validation.reason, uncertain=True, followup=validation.followup)
        if validation.option_id is not None and validation.option_id != resolved.candidate_option_id:
            return held_or_unresolved("ml_business_boundary_disagreement", uncertain=True, followup="I cannot confirm one meaning from that answer. Could you clarify the relevant fact in your own words?")
        if qid == "D12":
            mapped = map_explicit_horizon(extract_supported_need_clause(text) or text)
            if mapped.status == HorizonMappingStatus.AMBIGUOUS or (mapped.status == HorizonMappingStatus.MAPPED and mapped.category != resolved.candidate_option_id):
                return held_or_unresolved("ml_horizon_boundary_disagreement", uncertain=True, followup="What is the earliest point when you may realistically need any of this same money?")
            goal_or_access = re.search(r"\b(?:matur\w*|fixed[- ]?(?:deposit|term)|locked|unlocked|access|available|want to invest|plan to invest|investment goal|target date)\b", text, re.I)
            if goal_or_access and not re.search(r"\b(?:need|required|necessary|living costs|bills|untouched)\b", text, re.I):
                return held_or_unresolved("goal_or_access_is_not_earliest_need", uncertain=True, followup="That describes a goal or access date. What is the earliest point when you may realistically need this money?")
        if qid in {"D3", "D10"} and resolved.candidate_option_id not in {"D3_NO_EXISTING_POT", "D10_ZERO"} and not validation.context_details.get("currency"):
            self._details(record).update(validation.context_details)
            return held_or_unresolved("currency_and_range_required", followup="Please include the currency and the approximate range for this same money. You can also leave this optional context unanswered.")
        controls = {key: value for key, value in self._details(record).items() if key in {"understanding.teachback_required", "understanding.previous_belief"} or key.startswith("material.")}
        record.context_details = dict(validation.context_details) if validation.option_id == resolved.candidate_option_id else {}
        record.context_details.update(controls)
        record.independent_flags.update(validation.flags)
        self.state.active_holds.update(set(validation.flags) & MATERIAL_HOLDS)
        if teachback and record.context_details.get("understanding.own_words_supported") != "true":
            return held_or_unresolved("loss_understanding_requires_own_words")
        result = self._set_candidate(qid, resolved.candidate_option_id, source=resolved.source, reason=resolved.reason, flags=set(validation.flags) | flags)
        result["ml_evidence"] = trace
        return self._pace_supported_answer(qid, result, care, support, pace)

    def _profile_evidence(self, qid: str, text: str) -> BoundedAnswer:
        """Use the same ML option gates for financial follow-up evidence."""
        validation = interpret_bounded_answer(self.spec, qid, text)
        if self.question_resolver is None:
            return validation
        resolved = self.question_resolver.resolve(qid, text)
        self.state.log("ml_material_interpretation", qid, source=resolved.source,
                       intent_set=sorted(resolved.intent_set), option_set=sorted(resolved.option_set),
                       model_uncertain=resolved.model_uncertain, reason=resolved.reason)
        candidate = resolved.candidate_option_id
        if validation.option_id is None and validation.reason not in {"missing_material_detail", "no_supported_answer", "unsupported_meaning"}:
            candidate = None
        if validation.option_id is not None and validation.option_id != candidate:
            candidate = None
        if qid == "D12":
            mapped = map_explicit_horizon(extract_supported_need_clause(text) or text)
            if mapped.status == HorizonMappingStatus.AMBIGUOUS or (mapped.status == HorizonMappingStatus.MAPPED and mapped.category != candidate):
                candidate = None
        return BoundedAnswer(resolved.ordinary_bucket or "Confusion", option_id=candidate,
                             reason=resolved.reason or "ml_material_unresolved")

    def resume(self) -> dict[str, Any]:
        """Resume an ordinary user pause without resetting answer progress."""
        if self.state.stopped_for_safety:
            return {"action": "safety_stop", "reason": "conversation_already_stopped"}
        if "user_stop" in self.state.active_holds:
            return {"action": "stop", "reason": "conversation_already_stopped_by_user"}

        self.state.paused = False
        self.state.active_holds.discard("care_pacing")
        self.state.active_holds.discard("support_request")
        qid = self.state.current_question_id
        self.state.log("conversation_resumed", qid)
        result: dict[str, Any] = {
            "action": "resume" if qid is not None else "final_review",
            "question_id": qid,
        }
        if qid is not None:
            record = self._record(qid)
            result["clarification_turns"] = record.clarification_turns
            if record.status == AnswerStatus.CANDIDATE:
                option = self.spec.option(qid, record.candidate_option_id)
                result.update(
                    candidate_option_id=option.id,
                    candidate_option_label=option.label,
                    confirmation_text=self._confirmation_text(qid, option.label),
                    recorded=False,
                    context_details=self._public_details(record),
                )
        return result

    @staticmethod
    def _details(record: AnswerRecord) -> dict[str, str]:
        if not hasattr(record, "context_details"):
            record.context_details = {}
        return record.context_details

    def _public_details(self, record: AnswerRecord) -> dict[str, str]:
        internal = {"pending_option_id", "understanding.teachback_required", "understanding.own_word_check_requested", "understanding.previous_belief", "dependency.checked", "risk.tension_acknowledged"}
        return {key: value for key, value in self._details(record).items() if key not in internal and not key.startswith(("material.", "access.", "understanding."))}

    def _care_pause(self, qid: str | None, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
        self.state.paused = True
        self.state.active_holds.add("care_pacing")
        self.state.log("care_pacing_offered", qid)
        result = dict(candidate or {})
        result.pop("confirmation_text", None)
        result.update(action="care_pause", question_id=qid, pacing_text=CARE_PACING, recorded=False, bucket="Clarity" if candidate and candidate.get("candidate_option_id") else "Confusion")
        if qid is not None:
            self._record(qid).independent_flags.add("safety_or_distress")
            result["clarification_turns"] = self._record(qid).clarification_turns
        return result

    def _pacing_pause(self, qid: str | None, candidate: dict[str, Any] | None = None, *, previous_raw: str | None = None) -> dict[str, Any]:
        """A requested break holds confirmation without diagnosing distress."""
        self.state.paused = True
        self.state.active_holds.add("care_pacing")
        self.state.log("requested_break", qid)
        result = dict(candidate or {})
        if candidate is None and qid is not None and self._record(qid).status == AnswerStatus.CANDIDATE:
            record = self._record(qid)
            option = self.spec.option(qid, record.candidate_option_id)
            record.raw_text = previous_raw if previous_raw is not None else record.raw_text
            result.update(candidate_option_id=option.id, candidate_option_label=option.label, confirmation_text=self._confirmation_text(qid, option.label), context_details=self._public_details(record))
        result.pop("confirmation_text", None)
        result.update(action="pace_pause", question_id=qid, pacing_text="Of course. We can take a break. Choose continue when you are ready, or pause or stop here.", flag="pause_or_stop", recorded=False, bucket="Clarity" if result.get("candidate_option_id") else "Confusion")
        if qid is not None:
            self._record(qid).independent_flags.add("pause_or_stop")
            result["clarification_turns"] = self._record(qid).clarification_turns
        return result

    def _support_before_confirmation(self, qid: str | None, candidate: dict[str, Any] | None = None, *, control_only: bool = False) -> dict[str, Any]:
        self.state.paused = True
        self.state.active_holds.add("support_request")
        accessibility = control_only or (qid is not None and bool({"accessibility", "accessibility_request"} & self._record(qid).independent_flags))
        if not accessibility and qid is not None:
            self._record(qid).independent_flags.add("help_request")
        result = dict(candidate or {})
        result.pop("confirmation_text", None)
        support_text = ("We can use short sentences and one question at a time. When you are ready, we can check my proposed understanding together." if accessibility else "I can explain the question and choices first. When you are ready, we can check my proposed understanding together.")
        result.update(action="support_before_confirmation", question_id=qid, help_prompt=support_text, recorded=False, bucket="Clarity" if control_only or (candidate and candidate.get("candidate_option_id")) else "Confusion")
        if control_only:
            result.update(control_meaning="accessibility_request", candidate_option_id=None, candidate_option_label=None)
        if qid is not None:
            result["clarification_turns"] = self._record(qid).clarification_turns
        return result

    def _pace_supported_answer(self, qid: str, result: dict[str, Any], care: bool, support: bool, pace: bool = False) -> dict[str, Any]:
        if care:
            return self._care_pause(qid, result)
        if pace:
            return self._pacing_pause(qid, result)
        if support:
            return self._support_before_confirmation(qid, result)
        return result

    def _understanding_recheck(self, qid: str, *, count_turn: bool = True) -> dict[str, Any]:
        record = self._record(qid)
        already_requested = self._details(record).get("understanding.own_word_check_requested") == "true"
        self.state.active_holds.add("understanding_hold")
        self._details(record)["understanding.teachback_required"] = "true"
        self._details(record)["understanding.own_word_check_requested"] = "true"
        if count_turn:
            result = self._mark_unresolved(qid, AnswerStatus.CONFUSION, "loss_understanding_requires_own_words")
        else:
            result = {"action": "understanding_hold", "question_id": qid, "clarification_turns": record.clarification_turns, "recorded": False}
        if result["action"] != "finish_incomplete_or_pause":
            result["action"] = "understanding_hold"
        result.update(
            explanation=UNDERSTANDING_EXPLANATION,
            followup=("Your understanding remains unresolved. We can leave this part incomplete, explain it differently, or pause." if already_requested else UNDERSTANDING_FOLLOWUP),
        )
        return result

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
        if (
            qid == "D11" and option_id == "D11_LOSS_POSSIBLE"
            and self._details(record).get("understanding.teachback_required") == "true"
            and self._details(record).get("understanding.own_words_supported") != "true"
        ):
            return self._understanding_recheck(qid)
        record.status = AnswerStatus.CANDIDATE
        record.candidate_option_id = option_id
        record.selected_option_id = None
        record.selected_option_label = None
        record.source = source
        record.reason = reason
        previous_dependency = record.independent_flags & ({"same_money_dependency", "accessibility", "accessibility_request", "help_request", "pause_or_stop", "correction"} | MATERIAL_HOLDS)
        record.independent_flags = set(flags or set()) | previous_dependency

        if "same_money_dependency" in record.independent_flags:
            self._details(record).pop("dependency.checked", None)
            self._details(self._record("D12")).pop("material.same_money_dependency.resolved", None)
            self.state.active_holds.add("same_money_dependency")
        self.state.active_holds.update(record.independent_flags & MATERIAL_HOLDS)

        self.state.log(
            "candidate_created",
            qid,
            candidate_option_id=option_id,
            source=source,
            reason=reason,
            flags=sorted(record.independent_flags),
        )

        if qid == "D11":
            if option_id in {"D11_RECOVERY_ALWAYS", "D11_CAPITAL_PROTECTED"}:
                record.independent_flags.add("understanding_hold")
                self.state.active_holds.add("understanding_hold")
            elif self._details(record).get("understanding.teachback_required") != "true":
                self.state.active_holds.discard("understanding_hold")

        return {
            "action": "confirm",
            "question_id": qid,
            "candidate_option_id": option_id,
            "candidate_option_label": option.label,
            "confirmation_text": self._confirmation_text(qid, option.label),
            "context_details": self._public_details(record),
            "recorded": False,
        }

    def _confirmation_text(self, qid: str, label: str) -> str:
        if qid == "D12":
            return f"You said the earliest realistic need is {label}. Is that right?"
        return f"You selected {label}. Is that right?"

    def confirm_current(self, confirmed: bool) -> dict[str, Any]:
        if self.state.stopped_for_safety:
            return {"action": "safety_stop", "reason": "conversation_already_stopped"}
        if self.state.paused:
            return {"action": "paused", "reason": "conversation_is_paused"}

        qid = self.state.current_question_id
        if qid is None:
            raise RuntimeError("No current question to confirm")

        record = self._record(qid)
        if record.status != AnswerStatus.CANDIDATE:
            raise RuntimeError(f"{qid} has no candidate awaiting confirmation")

        if not confirmed:
            if qid == "D12" and "same_money_dependency" in record.independent_flags:
                self.state.active_holds.discard("same_money_dependency")
            record.status = AnswerStatus.PENDING
            record.candidate_option_id = None
            record.independent_flags.clear()
            record.reason = "candidate_rejected_by_user"
            record.source = None
            understanding_controls = {
                key: value for key, value in self._details(record).items()
                if key in {"understanding.teachback_required", "understanding.own_word_check_requested", "understanding.previous_belief"}
            }
            self._details(record).clear()
            self._details(record).update(understanding_controls)
            if qid == "D11" and understanding_controls.get("understanding.teachback_required") != "true":
                self.state.active_holds.discard("understanding_hold")
            self.state.log("candidate_rejected", qid)
            return {"action": "reask", "question_id": qid}

        if (
            qid == "D11" and record.candidate_option_id == "D11_LOSS_POSSIBLE"
            and self._details(record).get("understanding.teachback_required") == "true"
            and self._details(record).get("understanding.own_words_supported") != "true"
        ):
            return self._understanding_recheck(qid)

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
        if qid == "D11" and option.id in {"D11_RECOVERY_ALWAYS", "D11_CAPITAL_PROTECTED"}:
            self._details(record)["understanding.previous_belief"] = option.id
            self.state.current_index = self.state.question_order.index("D11")
            return self._understanding_recheck(qid)

        if qid == "D11" and option.id == "D11_LOSS_POSSIBLE":
            self.state.active_holds.discard("understanding_hold")
            self._details(record).pop("understanding.teachback_required", None)
            self._details(record).pop("understanding.own_word_check_requested", None)
        self._advance()

        return {
            "action": "continue" if self.state.current_question_id else "final_review",
            "confirmed_question_id": qid,
            "next_question_id": self.state.current_question_id,
            "profile_version": self.state.profile_version,
            "context_details": self._public_details(record),
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
        self._pending_material_check = None

        targets = {question_id}

        # D1 may change the named money, not just its purpose. Without an
        # explicitly reviewed same-scope assertion, old pot-specific context
        # must not silently be reused for the new money.
        if question_id == "D1":
            targets.update({"D3", "D4", "D5", "D6", "D7"})

        # Scope/dependency corrections can alter D12 and D13 interpretation.
        if question_id in {"D1", "D3", "D4", "D5", "D6", "D7", "D11"}:
            targets.update({"D12", "D13", "D14", "D15"})

        for qid in targets:
            record = self._record(qid)
            keep_teachback = qid == "D11" and self._details(record).get("understanding.teachback_required") == "true"
            record.status = AnswerStatus.PENDING
            record.raw_text = None
            record.candidate_option_id = None
            record.selected_option_id = None
            record.selected_option_label = None
            record.source = None
            record.reason = "invalidated_by_correction"
            record.independent_flags.clear()
            record.confirmed_version = None
            self._details(record).clear()
            if keep_teachback:
                self._details(record)["understanding.teachback_required"] = "true"
                self._details(record)["understanding.own_word_check_requested"] = "true"

        earliest = min(self.state.question_order.index(qid) for qid in targets)
        self.state.current_index = earliest
        if question_id in {"D1", "D3", "D4", "D5", "D6", "D7", "D11", "D12", "D13", "D14", "D15"}:
            for record in self.state.answers.values():
                for key in list(self._details(record)):
                    if key.startswith("material.") or key in {"funding_when_need_precedes_access", "usable_other_backup", "dependency_review", "essential_funding_review"}:
                        self._details(record).pop(key, None)
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
        for qid in {"D14", "D15"}:
            self._details(self._record(qid))["risk.tension_acknowledged"] = "true"
        self.state.active_holds.discard("d14_d15_tension_unacknowledged")
        self.state.bump_version()
        self._pending_material_check = None
        self.state.log("tension_acknowledged", "D14")

    def clear_dependency_hold(self) -> None:
        for record in self.state.answers.values():
            if "same_money_dependency" in record.independent_flags or record.question_id == "D12":
                self._details(record)["dependency.checked"] = "true"
        self.state.active_holds.discard("same_money_dependency")
        self.state.active_holds.discard("emergency_long_horizon_dependency")
        self.state.active_holds.discard("living_costs_horizon_dependency")
        self.state.bump_version()
        self._pending_material_check = None
        self.state.log("dependency_hold_cleared")

    def _current_material_check(self) -> tuple[str, str] | None:
        for hold in ("same_pot_backup_conflict", "usable_backup_check", "need_before_access", "essential_funding_timing", "same_money_dependency", "emergency_long_horizon_dependency", "living_costs_horizon_dependency"):
            if hold in self.state.active_holds:
                return hold, "D12" if hold in {"need_before_access", "essential_funding_timing"} or hold in DEPENDENCY_HOLDS else "D6"
        return None

    def material_check_prompt(self) -> str | None:
        check = self._current_material_check()
        if check is None:
            return None
        if check[0] == "need_before_access":
            return "Your earliest-need answer may fall before this money is accessible. If that payment is due while this money is unavailable, how would it be met? Please name any other resources and whether they are usable when the payment is due."
        if check[0] == "essential_funding_timing":
            return "How would your essential costs be paid during the current year? Please name the money or income actually usable for them, and say whether any of this same pot may be needed earlier than your confirmed timeframe."
        if check[0] in DEPENDENCY_HOLDS:
            return "For this same money, what is the earliest point you may realistically need any of it for emergencies, living costs or repayments? In the 20% loss example, would essentials still be paid, and would your budget or other plans change?"
        return "Excluding this same pot and money unavailable during the year, what other resources could cover necessary living costs for the full 12 months, and are they usable during that period?"

    def _stage_material_candidate(self, check_id: str, owner: str, details: dict[str, str], fact: str) -> dict[str, Any]:
        self._pending_material_check = {"check_id": check_id, "counter_owner": owner, "profile_version": self.state.profile_version, "context_details": details, "confirmation_text": f"You said {fact}. Is that exact understanding right?"}
        self.state.log("material_check_candidate", owner, check_id=check_id, context_details=details)
        return {"action": "confirm_check", **self._pending_material_check, "recorded": False}

    def _resolve_dependency_check(self, raw: str, check_id: str, owner: str) -> dict[str, Any]:
        horizon_record, impact_record = self._record("D12"), self._record("D13")
        if horizon_record.status != AnswerStatus.CONFIRMED or impact_record.status != AnswerStatus.CONFIRMED:
            missing = "D12" if horizon_record.status != AnswerStatus.CONFIRMED else "D13"
            self._pending_material_check = None
            return {"action": "request_correction_target", "question_id": missing, "check_id": check_id, "reason": "dependency_requires_supported_need_and_practical_effect", "recorded": False}
        describes_need = bool(re.search(r"\b(?:earliest|realistic|need|required|living costs|repayments|emergencies)\b", raw, re.I))
        mapped = map_explicit_horizon(extract_supported_need_clause(raw) or raw) if describes_need else None
        no_earlier = bool(re.search(r"\b(?:no earlier (?:realistic )?need|(?:will not|won't|would not|wouldn't) need (?:any of )?(?:this|the same) (?:pot|money) (?:any )?earlier|no need .*before (?:the )?(?:confirmed|stated) (?:date|timeframe|horizon))\b", raw, re.I))
        need_option = mapped.category if mapped and mapped.status == HorizonMappingStatus.MAPPED else horizon_record.selected_option_id if no_earlier else None
        if self.question_resolver is not None and not no_earlier:
            need_option = self._profile_evidence("D12", raw).option_id
        if need_option is not None and need_option != horizon_record.selected_option_id:
            self._pending_material_check = None
            return {"action": "request_correction_target", "question_id": "D12", "check_id": check_id, "reason": "dependency_changes_earliest_need", "followup": "The realistic need you described changes the confirmed timeframe. Please update that answer; I have kept the existing need and have not moved it to a maturity date.", "recorded": False}
        # Check the scenario stated by the user, rather than treating backup,
        # willingness or a reassurance alone as the practical 20% loss effect.
        impact_reply = re.search(r"\b(?:20\s*%|twenty (?:per ?cent|percent))[^.!?]*(?:[.!?][^.!?]*)?", raw, re.I)
        impact = self._profile_evidence("D13", impact_reply.group(0)) if impact_reply else None
        if impact is not None and impact.option_id is not None and impact.option_id != impact_record.selected_option_id:
            self._pending_material_check = None
            return {"action": "request_correction_target", "question_id": "D13", "check_id": check_id, "reason": "dependency_changes_practical_loss_effect", "followup": "That changes the practical effect of the 20% loss example. Please update that answer and confirm the new meaning.", "recorded": False}
        if need_option is None or impact is None or impact.option_id is None:
            return self._material_unresolved(check_id, owner, "earliest_need_or_practical_effect_missing")
        need_fact = f"the earliest realistic need remains {horizon_record.selected_option_label}"
        if mapped and mapped.status == HorizonMappingStatus.MAPPED:
            timing = ", ".join(f"{duration:g}" for duration in mapped.durations_years)
            need_fact += f" (stated timing in years: {timing})"
        fact = f"for this same money, {need_fact}; in the 20% loss example, {impact_record.selected_option_label}"
        return self._stage_material_candidate(check_id, owner, {"dependency_review": fact}, fact)

    def _resolve_essential_funding(self, raw: str, check_id: str, owner: str) -> dict[str, Any]:
        record = self._record("D12")
        # Keep need evidence separate from the timing of another resource.
        needs = [sentence for sentence in re.split(r"[.!?]", raw) if re.search(r"\b(?:earliest (?:realistic )?need|need (?:any of )?(?:this (?:same )?money|this (?:same )?pot))\b", sentence, re.I)]
        mapped = map_explicit_horizon(extract_supported_need_clause(". ".join(needs)) or ". ".join(needs)) if needs else None
        if mapped and mapped.status == HorizonMappingStatus.MAPPED and mapped.category != record.selected_option_id:
            self._pending_material_check = None
            return {"action": "request_correction_target", "question_id": "D12", "check_id": check_id, "reason": "essential_funding_changes_earliest_need", "followup": "You described a realistic earlier need for this same money. Please update the timeframe; I have not changed it automatically or assumed replacement funds.", "recorded": False}

        # All-year other funding may change the total-unavailability answer.
        # It must be reviewed there, rather than accepted as a hidden override.
        period = bool(re.search(r"\b(?:full|entire|whole|all)\s+(?:12[- ]?months?|year)|\b(?:for|throughout|during)\s+(?:the )?(?:current|next)?\s*(?:12[- ]?months?|year)\b", raw, re.I))
        resources = re.findall(r"\b(?:salary|wages|pension payments|pension income|benefits|rental income|separate (?:(?:easy|instant)[- ]access )?(?:savings|bank account|account)|other (?:(?:easy|instant)[- ]access )?(?:savings|bank account))\b", raw, re.I)
        available = bool(re.search(r"\b(?:available|accessible|usable|can (?:access|use)|paid|receive|receiving)\b", raw, re.I))
        blocked = bool(re.search(r"\b(?:locked|inaccessible|unavailable|cannot access|can't access|not (?:available|accessible|usable)|cannot (?:pay|cover)|can't (?:pay|cover)|not enough)\b", raw, re.I))
        independent = bool(re.search(r"\b(?:without (?:using|touching)|excluding|not using)\s+(?:this (?:same )?(?:pot|money)|the pot)\b|\bseparate (?:savings|bank account|account)\b", raw, re.I)) and not bool(re.search(r"\b(?:not without|only (?:by )?(?:using|withdrawing|drawing))\b", raw, re.I))
        coverage = self._profile_evidence("D6", raw)
        if resources and available and independent and period and not blocked and coverage.option_id is not None:
            if coverage.option_id != self._record("D6").selected_option_id:
                self._pending_material_check = None
                return {"action": "request_correction_target", "question_id": "D6", "check_id": check_id, "reason": "essential_funding_changes_usable_backup", "followup": "Those other usable resources change the coverage previously described for the full year without this pot. Please update and confirm the backup answer.", "recorded": False}
            source = self._profile_evidence("D5", raw)
            if source.option_id is not None and source.option_id != self._record("D5").selected_option_id:
                self._pending_material_check = None
                return {"action": "request_correction_target", "question_id": "D5", "check_id": check_id, "reason": "essential_funding_changes_current_source", "followup": "That identifies a different current source for living costs. Please review and confirm that source before completing the funding check.", "recorded": False}
            # Partial backup leaves the genuinely missing funding for the
            # remaining essentials unresolved. Do not invent it or count the pot.
            return self._material_unresolved(check_id, owner, "remaining_essential_funding_or_timing_missing")
        return self._material_unresolved(check_id, owner, "actual_usable_essential_funding_or_timing_missing")

    def _material_unresolved(self, check_id: str, owner: str, reason: str, *, bucket: str = "Undecided") -> dict[str, Any]:
        # A check shares its financial item's counter. It does not change an
        # already confirmed horizon or silently overwrite the backup answer.
        record = self._record(owner)
        record.clarification_turns += 1
        self._pending_material_check = None
        self.state.log("material_check_clarification", owner, check_id=check_id, reason=reason, clarification_turns=record.clarification_turns)
        return {
            "action": "finish_incomplete_or_pause" if record.clarification_turns >= MAX_CLARIFICATION_TURNS else "clarify",
            "question_id": owner,
            "counter_owner": owner,
            "check_id": check_id,
            "bucket": bucket,
            "reason": reason,
            "clarification_turns": record.clarification_turns,
            "followup": self.material_check_prompt(),
            "profile_complete": False,
            "recorded": False,
        }

    def resolve_material_check(self, text: str) -> dict[str, Any]:
        """Propose only the missing supported funding fact, then confirm it.

        This deliberately narrow rule path abstains on unknown resource names,
        timing or coverage. It is separate from the three financial conclusions.
        """
        control = check_pre_model_controls(text)
        if self.state.stopped_for_safety or "user_stop" in self.state.active_holds:
            return self.submit_answer(text)
        if control.action in {"safety_stop", "stop", "pause", "resume"}:
            return self.submit_answer(text)
        if self.state.paused:
            return {"action": "paused", "reason": "conversation_is_paused"}
        if control.action == "care_pause":
            return self._care_pause(None)
        if control.action == "pace_answer":
            return self._pacing_pause(None)
        check = self._current_material_check()
        if check is None:
            return {"action": "final_review", "reason": "no_material_check_pending"}
        check_id, owner = check
        record = self._record(owner)
        if control.action in {"explain", "support_answer", "support_control"}:
            if control.flag:
                record.independent_flags.add(control.flag)
                if control.flag == "accessibility":
                    record.independent_flags.add("accessibility_request")
            if control.action in {"support_answer", "support_control"}:
                return self._support_before_confirmation(owner, control_only=control.action == "support_control")
            return {"action": "explain", "question_id": owner, "check_id": check_id, "bucket": "Confusion", "followup": self.material_check_prompt(), "help_prompt": "We need only the missing funding fact; your confirmed financial answer stays unchanged. You can pause or leave this incomplete.", "clarification_turns": record.clarification_turns, "recorded": False}
        if record.clarification_turns >= MAX_CLARIFICATION_TURNS:
            return {"action": "finish_incomplete_or_pause", "question_id": owner, "counter_owner": owner, "check_id": check_id, "reason": "clarification_limit_reached", "clarification_turns": record.clarification_turns, "profile_complete": False}

        raw = text.replace("’", "'")
        self.state.log("material_check_reply", owner, check_id=check_id, text=raw)
        if control.action == "decline":
            self._pending_material_check = None
            return {"action": "finish_incomplete_or_pause", "question_id": owner, "check_id": check_id, "reason": "material_fact_declined", "profile_complete": False, "recorded": False}
        if control.action == "correction_request":
            self._pending_material_check = None
            return {"action": "request_correction_target", "question_id": owner, "check_id": check_id, "reason": "material_fact_requires_review", "recorded": False}
        uncertain = control.action == "undecided" or re.search(r"\b(?:not sure|don't know|do not know|maybe|might|perhaps|hopefully|probably|if .*enough)\b", raw, re.I)
        if uncertain:
            return self._material_unresolved(check_id, owner, "funding_or_usability_unresolved")
        if check_id in DEPENDENCY_HOLDS:
            return self._resolve_dependency_check(raw, check_id, owner)
        if check_id == "essential_funding_timing":
            return self._resolve_essential_funding(raw, check_id, owner)

        same_pot = re.search(r"\b(?:this (?:same |exact )?(?:pot|money)|same money|money we are discussing)\b", raw, re.I)
        excludes_pot = re.search(r"\b(?:without (?:using|touching|drawing from)|excluding|not (?:using|counting))\s+(?:this (?:same |exact )?(?:pot|money)|the (?:same )?pot)\b", raw, re.I)
        depends_on_pot = bool(same_pot and not excludes_pot and re.search(r"\b(?:need|use|using|depend|withdraw\w*|draw\w*|fund|count\w*)\b", raw, re.I))
        depends_on_pot = depends_on_pot or bool(re.search(r"\b(?:cannot|can't|couldn't|could not)\s+(?:pay|cover|meet|manage)\b.*\bwithout\b.*\b(?:this|same) (?:pot|money)\b", raw, re.I))
        depends_on_pot = depends_on_pot or bool(same_pot and re.search(r"\b(?:not without|only (?:by )?(?:using|withdrawing|drawing))\b", raw, re.I))
        if depends_on_pot:
            self._pending_material_check = None
            return {"action": "request_correction_target", "question_id": "D6" if owner == "D6" else "D12", "check_id": check_id, "reason": "same_money_dependency_remains", "correction_question_ids": ["D6"] if owner == "D6" else ["D4", "D12"], "followup": "This same money cannot provide separate backup while it is unavailable. Please review the affected answer; I have kept the realistic need date and have not assumed replacement funds.", "profile_complete": False, "recorded": False}

        resources = re.findall(r"\b(?:salary|wages|pension payments|pension income|benefits|rental income|separate (?:(?:easy|instant)[- ]access )?(?:savings|bank account|account)|other (?:(?:easy|instant)[- ]access )?(?:savings|bank account))\b", raw, re.I)
        # A currently identified separate cash account is payment funding,
        # not an automatic revision to the full-year D6 backup answer.
        identified_cash = bool(re.search(r"\bi (?:have (?:(?:now|already) )?(?:identified|opened|found|located)|already have|have)\s+(?:a |the )?separate cash account\b", raw, re.I))
        cash_payment_covered = bool(re.search(r"\b(?:it|(?:the|that|this) (?:cash )?account)\s+(?:already )?(?:covers|pays|meets|funds)\s+(?:that|this|the) payment in full\b", raw, re.I))
        cash_future_promise = bool(re.search(r"\b(?:hope|plan|intend|expect|promis\w*|waiting)\b|\b(?:will|going to)\s+(?:open|identify|find|create|fund|set up)\b", raw, re.I))
        if check_id == "need_before_access" and identified_cash and cash_payment_covered and not cash_future_promise:
            resources.append("separate cash account")
        available = bool(re.search(r"\b(?:available|accessible|usable|can (?:access|use)|paid|receive|receiving)\b", raw, re.I))
        blocked = bool(re.search(r"\b(?:locked|inaccessible|unavailable|cannot (?:access|use)|can't (?:access|use)|not (?:available|accessible|usable))\b", raw, re.I))
        blocked = blocked or bool(re.search(r"\b(?:cannot|can't|couldn't|could not|won't|will not|doesn't|does not|don't|do not)\s+(?:(?:fully|completely)\s+)?(?:pay|cover|meet|fund)\b|\bnot enough\b", raw, re.I))
        blocked = blocked or bool(re.search(r"\b(?:payment|bill|rent)\s+(?:(?:is|are|will|would|can)\s+)?(?:not|never)\s+(?:be\s+)?(?:(?:fully|completely)\s+)?(?:covered|paid|funded)\b", raw, re.I))
        due_timing = bool(re.search(r"\b(?:when (?:the )?(?:payment|bill|rent) is due|when (?:it|they) (?:is|are) (?:needed|due)|in time for (?:the )?(?:payment|bill|rent)|by (?:the )?due date)\b", raw, re.I))
        due_timing = due_timing or bool(identified_cash and re.search(r"\bi can (?:use|access)(?: it)? on (?:the )?payment date\b", raw, re.I))
        period = bool(re.search(r"\b(?:full|entire|whole|all)\s+(?:12[- ]?months?|year)|\b(?:for|throughout|during)\s+(?:the )?(?:next )?(?:12[- ]?months?|year)\b", raw, re.I))
        full_coverage = bool(re.search(r"\b(?:cover|pay|meet)\w*\s+(?:all|every|the full)\b.*\b(?:costs|bills|essentials|rent)|\b(?:all|every|full)\b.*\b(?:costs|bills|essentials)\b.*\b(?:covered|paid)\b", raw, re.I))
        some_coverage = bool(re.search(r"\b(?:cover|pay|meet)\w*\s+(?:only )?(?:some|part)\b.*\b(?:costs|bills|essentials)|\b(?:some|part)\b.*\b(?:costs|bills|essentials)\b.*\b(?:covered|paid)\b", raw, re.I))
        payment_funded = bool(re.search(r"\b(?:cover|pay|meet|fund)\w*\s+(?:that|this|the|my)\s+(?:(?:full|entire|whole)\s+)?(?:tuition\s+)?(?:payment|bill|rent)|\b(?:payment|bill|rent)\b.*\b(?:covered|paid|funded)\b", raw, re.I))
        if check_id == "need_before_access":
            partial_payment = bool(re.search(r"\b(?:only some|only part|partly|half|not fully|in part|not in full)\b", raw, re.I))
            supported = resources and available and due_timing and payment_funded and not blocked and not partial_payment
        else:
            supported = resources and available and period and (full_coverage != some_coverage) and not blocked
            expected = "D6_ALL" if full_coverage else "D6_SOME"
            if supported and self.question_resolver is not None:
                supported = self._profile_evidence("D6", raw).option_id == expected
            if supported and (record.status != AnswerStatus.CONFIRMED or record.selected_option_id != expected):
                self._pending_material_check = None
                return {"action": "request_correction_target", "question_id": "D6", "check_id": check_id, "reason": "backup_coverage_requires_fresh_answer_confirmation", "followup": "Please update the backup question with only the other usable resources you identified, then confirm its coverage for the full year.", "recorded": False}
        if not supported:
            return self._material_unresolved(check_id, owner, "named_resources_usability_or_coverage_missing", bucket="Confusion" if not resources else "Undecided")

        named = ", ".join(dict.fromkeys(resource.lower() for resource in resources))
        fact = (f"{named} can fund that payment and is usable when it is due, while this pot is unavailable" if check_id == "need_before_access" else f"{named} is usable during the full 12 months and covers {'all' if full_coverage else 'some'} necessary costs without this pot")
        detail_key = "funding_when_need_precedes_access" if check_id == "need_before_access" else "usable_other_backup"
        return self._stage_material_candidate(check_id, owner, {detail_key: fact}, fact)

    def confirm_material_check(self, confirmed: bool) -> dict[str, Any]:
        if self.state.stopped_for_safety:
            return {"action": "safety_stop", "reason": "conversation_already_stopped"}
        if self.state.paused or "user_stop" in self.state.active_holds:
            return {"action": "paused", "reason": "conversation_is_paused"}
        pending = self._pending_material_check
        if pending is None:
            return {"action": "clarify", "reason": "no_material_candidate_to_confirm", "followup": self.material_check_prompt(), "recorded": False}
        if pending["profile_version"] != self.state.profile_version or pending["check_id"] not in self.state.active_holds:
            self._pending_material_check = None
            return {"action": "clarify", "reason": "material_candidate_stale", "followup": self.material_check_prompt(), "recorded": False}
        self._pending_material_check = None
        if not confirmed:
            return {"action": "clarify", "check_id": pending["check_id"], "reason": "material_candidate_rejected", "followup": self.material_check_prompt(), "recorded": False}
        record = self._record(pending["counter_owner"])
        self._details(record).update(pending["context_details"])
        self._details(record)[f"material.{pending['check_id']}.resolved"] = "true"
        self.state.bump_version()
        self._refresh_cross_question_holds()
        self.state.log("material_check_confirmed", record.question_id, check_id=pending["check_id"], context_details=pending["context_details"])
        return {"action": "material_check_resolved", "check_id": pending["check_id"], "profile_version": self.state.profile_version, "active_holds": sorted(self.state.active_holds), "context_details": pending["context_details"], "recorded": True}

    def _refresh_cross_question_holds(self) -> None:
        # Recompute only the specific consistency controls in the agreed MVP scope.
        self.state.active_holds.discard("emergency_long_horizon_dependency")
        self.state.active_holds.discard("living_costs_horizon_dependency")
        self.state.active_holds.discard("d14_d15_tension_unacknowledged")
        self.state.active_holds.discard("same_money_dependency")
        self.state.active_holds.discard("understanding_hold")
        self.state.active_holds.difference_update(MATERIAL_HOLDS)

        for record in self.state.answers.values():
            if (
                "same_money_dependency" in record.independent_flags
                and self._details(record).get("dependency.checked") != "true"
                and self._details(self._record("D12")).get("material.same_money_dependency.resolved") != "true"
            ):
                self.state.active_holds.add("same_money_dependency")
            for flag in record.independent_flags & MATERIAL_HOLDS:
                if self._details(self._record("D6")).get(f"material.{flag}.resolved") != "true":
                    self.state.active_holds.add(flag)
        d11 = self._record("D11")
        if (
            d11.selected_option_id in {"D11_RECOVERY_ALWAYS", "D11_CAPITAL_PROTECTED"}
            or self._details(d11).get("understanding.teachback_required") == "true"
        ):
            self.state.active_holds.add("understanding_hold")

        d1 = self._record("D1").selected_option_id
        d12 = self._record("D12").selected_option_id
        d6 = self._record("D6").selected_option_id
        d13 = self._record("D13").selected_option_id
        d14 = self._record("D14").selected_option_id
        d15 = self._record("D15").selected_option_id
        available = self._details(self._record("D4")).get("access.available_after_years")
        d12_record = self._record("D12")
        if available is not None and d12_record.status == AnswerStatus.CONFIRMED:
            try:
                available_years = float(available)
            except ValueError:
                available_years = None
            text = d12_record.raw_text or ""
            mapped_need = map_explicit_horizon(extract_supported_need_clause(text) or text)
            need_years = min(mapped_need.durations_years) if mapped_need.status == HorizonMappingStatus.MAPPED else None
            # A selected range is not its upper endpoint. If an allowed need
            # within that band could precede access, timing/funding is missing.
            band_lower = {"D12_SHORT": 0.0, "D12_MEDIUM": 3.0, "D12_LONG": 10.0}.get(d12)
            need_may_precede_access = (
                available_years is not None
                and ((need_years is not None and need_years < available_years)
                     or (need_years is None and band_lower is not None and band_lower < available_years))
            )
            if (
                need_may_precede_access
                and self._details(d12_record).get("material.need_before_access.resolved") != "true"
            ):
                self.state.active_holds.add("need_before_access")

        if (
            d1 == "D1_EMERGENCIES" and d12 == "D12_LONG"
            and self._details(self._record("D12")).get("dependency.checked") != "true"
            and self._details(self._record("D12")).get("material.emergency_long_horizon_dependency.resolved") != "true"
        ):
            self.state.active_holds.add("emergency_long_horizon_dependency")

        if (
            d1 == "D1_LIVING_COSTS" and d12 in {"D12_MEDIUM", "D12_LONG"}
            and self._details(self._record("D12")).get("dependency.checked") != "true"
            and self._details(self._record("D12")).get("material.living_costs_horizon_dependency.resolved") != "true"
        ):
            self.state.active_holds.add("living_costs_horizon_dependency")

        if (
            d6 in {"D6_NONE", "D6_SOME"} and d12 in {"D12_MEDIUM", "D12_LONG"}
            and d13 == "D13_ESSENTIALS_DIFFICULT"
            and self._details(d12_record).get("material.essential_funding_timing.resolved") != "true"
        ):
            self.state.active_holds.add("essential_funding_timing")

        if (
            d14 == "D14_GREATER" and d15 == "D15_UNACCEPTABLE"
            and not all(self._details(self._record(qid)).get("risk.tension_acknowledged") == "true" for qid in {"D14", "D15"})
        ):
            self.state.active_holds.add("d14_d15_tension_unacknowledged")

    def completion_report(self) -> dict[str, Any]:
        missing_required: list[str] = []
        handled: list[str] = []

        for qid in self.state.question_order:
            record = self._record(qid)
            if record.status in {AnswerStatus.CONFIRMED, AnswerStatus.DECLINED}:
                handled.append(qid)

            if qid in OPTIONAL_CONTEXT or qid in CONDITIONAL_CONTEXT:
                continue

            if record.status != AnswerStatus.CONFIRMED:
                missing_required.append(qid)
            elif qid == "D11" and record.selected_option_id in {"D11_RECOVERY_ALWAYS", "D11_CAPITAL_PROTECTED"}:
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
                and not self.state.paused
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
                    "context_details": self._public_details(record),
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
        if not self.completion_report()["eligible_for_final_accuracy_confirmation"]:
            return {"accepted": False, "reason": "completion_gate_failed"}
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
                "context_details": self._public_details(record),
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
