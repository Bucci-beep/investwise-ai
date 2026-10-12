"""Opt-in D5 wording from verified v2 decisions; normal factory is unchanged.

No source-word matching, extra interpreter, free-form model text rendering or
changes to control/state transitions. Follow-up topics come from the same call.
"""
from ai.d5_single_interpreter_engine import D5SingleInterpreterEngine
from ai.d5_single_interpreter_contract_v2 import SOURCE, SOURCE_USE_TOPICS


D5_EXPLANATION = (
    "This asks what pays your housing, food and bills now. Pension payments "
    "count when you receive and use them for these costs; simply holding money "
    "inside a pension fund does not count. Receiving money does not by itself "
    "mean you spend it on living costs. You can name one source or several. "
    "Benefits and regular support share one answer category."
)


class D5BusinessStandardEngine(D5SingleInterpreterEngine):
    def _set_candidate(self, qid, option_id, *, source, reason, flags=None):
        # The inherited guard verifies the old receipt first, then binds a new
        # receipt after setting the candidate with this route's provenance.
        if qid == "D5" and source == "d5_single_interpreter_v1":
            source = SOURCE
        return super()._set_candidate(qid, option_id, source=source,
                                     reason=reason, flags=flags)

    def _d5_unresolved(self, **kwargs):
        result = super()._d5_unresolved(**kwargs)
        # Keep care, accessibility, support, pause and exhausted-budget routes.
        if result.get("action") != "clarify":
            return result
        decision = self.d5_interpreter.last_decision
        text = self._record("D5").raw_text or ""
        if (decision is None or decision.outcome not in {"dialogue_only", "abstention"}
                or not self.d5_interpreter.verify_decision(decision, text, spec=self.spec)
                or decision.interpretation is None):
            return result
        meaning = decision.interpretation
        topic = meaning.missing_information[0]
        if topic in SOURCE_USE_TOPICS:
            category = topic.split(":", 1)[1]
            labels = {option.id: option.label.lower() for option in self.spec.question("D5").options}
            question = (
                f"Does money from {labels[category]} currently pay any of your living "
                "costs, or is that still unclear? Please include any other sources "
                "paying these costs so your reply gives the complete current answer."
            )
        else:
            question = {
                "current_funding_unknown": (
                    "Which sources, if any, actually pay your housing, food and "
                    "bills now? They may differ from money you receive or hold. "
                    "You can name another source or leave this unresolved."
                ),
                "complete_answer_needed": (
                    "Please give your complete current funding answer, including any "
                    "correction, so I do not have to guess which sources still apply."
                ),
                "question_meaning": D5_EXPLANATION + " What currently pays these costs for you?",
                "reply_unclear": (
                    "I haven't understood your wording. Could you describe what pays "
                    "your living costs in another way? You can leave this unresolved."
                ),
            }.get(topic)
        if question is not None:
            if meaning.semantic_bucket == "undecided":
                question = "It is fine to be unsure. " + question
            result["followup"] = question
            result["clarification_topic"] = topic
        return result

    def submit_answer(self, text, *, explicit_option_id=None):
        result = super().submit_answer(text, explicit_option_id=explicit_option_id)
        if (result.get("question_id") == "D5" and result.get("action") == "explain"
                and result.get("definition_only") is True
                and result.get("help_explanation_provided") is True):
            # A single authored D5 definition: no phrase detection or model call.
            result["help_prompt"] = D5_EXPLANATION
        if result.get("question_id") != "D5" or result.get("action") != "confirm":
            return result
        receipt = self._d5_receipt
        if receipt is None or not self._valid_receipt(text, result.get("candidate_option_id")):
            return result
        meaning = receipt.decision.interpretation
        labels = {option.id: option.label.lower() for option in self.spec.question("D5").options}
        active = {fact.source_category for fact in meaning.facts if fact.currently_funds_customer}
        if meaning.option_id == "D5_NO_REGULAR":
            playback = "You have no regular source paying your living costs at the moment."
        else:
            names = " and ".join(labels[category] for category in sorted(active))
            playback = f"My understanding is that your living costs are currently paid by {names}."
        excluded = {fact.source_category for fact in meaning.facts
                    if fact.source_category not in active | {"D5_NO_REGULAR"}
                    and (fact.owner == "other" or fact.time in {"past", "future"}
                         or fact.pays_living_costs == "no")}
        if excluded:
            names = " and ".join(labels[category] for category in sorted(excluded))
            playback += f" Other sources mentioned but not counted as current funding: {names}."
        result["confirmation_text"] = playback + " Have I understood correctly?"
        return result
