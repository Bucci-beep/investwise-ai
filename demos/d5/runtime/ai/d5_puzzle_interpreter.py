"""Independent, opt-in D5 ML-first cascade with an injected LLM backend.

The baseline owns source interpretation until it actually abstains. The LLM
then proposes a semantic bucket, an allowed option and explicit source facts.
The checks here only veto that proposal: they never infer a replacement source,
repair a bucket, confirm an answer, or retain a mutable conversation/store.

Exact-substring checks establish provenance of quoted evidence, not the truth
of its category, ownership, time or use. Those relations remain LLM judgments.
The ordinary engine still owns its business checks, clarification limit,
confirmation, correction and recording. LLM results have no CP guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from ai.all_question_ml import QuestionMLResolution
from ai.llm_shadow import (BUCKET_NAMES, ShadowContractError, canonical_json,
                           sha256_bytes, strict_json)
from ai.pre_model_controls import check_pre_model_controls
from ai.question_spec import BusinessSpec


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config/d5-puzzle-v1.json"
SOURCE = "d5_puzzle_llm"
D5_OPTION_IDS = frozenset({
    "D5_SALARY", "D5_SELF_EMPLOYMENT", "D5_PENSION_PAYMENTS",
    "D5_BENEFITS_SUPPORT", "D5_WITHDRAWALS", "D5_OTHER_REGULAR",
    "D5_NO_REGULAR", "D5_MULTIPLE",
})
SCHEMA_KEYS = frozenset({"semantic_bucket", "option_id", "facts", "evidence",
                         "missing_information", "model_abstain"})
FACT_KEYS = frozenset({"source_category", "owner", "time", "pays_living_costs", "evidence"})
SEMANTIC_LABELS = frozenset({"clarity", "undecided", "confusion"})
OWNERS = frozenset({"self", "other", "unknown"})
TIMES = frozenset({"current", "future", "past", "unknown"})
USES = frozenset({"yes", "no", "unknown"})


class D5PuzzleContractError(ShadowContractError):
    """A generated proposal violates the D5 contract; no repair is attempted."""


def _json(payload: str):
    try:
        return strict_json(payload)
    except ShadowContractError as exc:
        raise D5PuzzleContractError(str(exc)) from exc


@dataclass(frozen=True)
class PuzzleFact:
    source_category: str
    owner: str
    time: str
    pays_living_costs: str
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {"source_category": self.source_category, "owner": self.owner,
                "time": self.time, "pays_living_costs": self.pays_living_costs,
                "evidence": self.evidence}

    @property
    def currently_funds_customer(self) -> bool:
        return (self.source_category != "D5_NO_REGULAR" and self.owner == "self"
                and self.time == "current" and self.pays_living_costs == "yes")


@dataclass(frozen=True)
class D5PuzzleInterpretation:
    semantic_bucket: str | None
    option_id: str | None
    facts: tuple[PuzzleFact, ...]
    evidence: tuple[str, ...]
    missing_information: tuple[str, ...]
    model_abstain: bool

    def as_dict(self) -> dict[str, object]:
        return {"semantic_bucket": self.semantic_bucket, "option_id": self.option_id,
                "facts": [fact.as_dict() for fact in self.facts],
                "evidence": list(self.evidence),
                "missing_information": list(self.missing_information),
                "model_abstain": self.model_abstain}


def output_schema(allowed_option_ids: frozenset[str]) -> dict[str, object]:
    """JSON Schema passed separately to a structured-output-capable backend."""
    allowed = sorted(allowed_option_ids)
    sources = sorted(allowed_option_ids - {"D5_MULTIPLE"})
    strings = {"type": "array", "items": {"type": "string", "minLength": 1},
               "uniqueItems": True}
    return {"type": "object", "additionalProperties": False,
            "required": sorted(SCHEMA_KEYS), "properties": {
                "semantic_bucket": {"type": ["string", "null"],
                                    "enum": ["clarity", "undecided", "confusion", None]},
                "option_id": {"type": ["string", "null"], "enum": allowed + [None]},
                "facts": {"type": "array", "uniqueItems": True, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": sorted(FACT_KEYS), "properties": {
                        "source_category": {"type": "string", "enum": sources},
                        "owner": {"type": "string", "enum": sorted(OWNERS)},
                        "time": {"type": "string", "enum": sorted(TIMES)},
                        "pays_living_costs": {"type": "string", "enum": sorted(USES)},
                        "evidence": {"type": "string", "minLength": 1}}}},
                "evidence": dict(strings), "missing_information": dict(strings),
                "model_abstain": {"type": "boolean"}}}


def parse_puzzle_interpretation(raw_output: str, original_reply: str,
                                allowed_option_ids: frozenset[str], *,
                                require_diagnostic_evidence: bool = False) -> D5PuzzleInterpretation:
    """Reject unsupported structure while keeping semantic U/C distinct from abstention.

    This validates the LLM's *asserted* relations. It cannot certify that an
    exact span semantically entails them; no regex source mapper is used here.
    """
    payload = _json(raw_output)
    if not isinstance(payload, dict) or set(payload) != SCHEMA_KEYS:
        raise D5PuzzleContractError("exact_schema_keys_required")
    bucket, option = payload["semantic_bucket"], payload["option_id"]
    if bucket is not None and (not isinstance(bucket, str) or bucket not in SEMANTIC_LABELS):
        raise D5PuzzleContractError("invalid_semantic_bucket")
    if option is not None and (not isinstance(option, str) or option not in allowed_option_ids
                               or option not in D5_OPTION_IDS):
        raise D5PuzzleContractError("unapproved_option_id")
    if type(payload["model_abstain"]) is not bool:
        raise D5PuzzleContractError("model_abstain_must_be_boolean")
    for key in ("evidence", "missing_information"):
        values = payload[key]
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise D5PuzzleContractError(key + "_must_be_nonempty_strings")
        if len(values) != len(set(values)):
            raise D5PuzzleContractError(key + "_contains_duplicates")
    if not isinstance(original_reply, str):
        raise D5PuzzleContractError("original_reply_must_be_text")
    if any(span not in original_reply for span in payload["evidence"]):
        raise D5PuzzleContractError("evidence_is_not_exact_reply_span")
    if not isinstance(payload["facts"], list):
        raise D5PuzzleContractError("facts_must_be_array")
    facts = []
    sources = (allowed_option_ids & D5_OPTION_IDS) - {"D5_MULTIPLE"}
    for item in payload["facts"]:
        if not isinstance(item, dict) or set(item) != FACT_KEYS:
            raise D5PuzzleContractError("exact_fact_keys_required")
        if any(not isinstance(item[key], str) for key in FACT_KEYS):
            raise D5PuzzleContractError("fact_values_must_be_strings")
        if item["source_category"] not in sources:
            raise D5PuzzleContractError("unapproved_source_category")
        if item["owner"] not in OWNERS or item["time"] not in TIMES or item["pays_living_costs"] not in USES:
            raise D5PuzzleContractError("invalid_fact_relation")
        if not item["evidence"].strip() or item["evidence"] not in original_reply:
            raise D5PuzzleContractError("fact_evidence_is_not_exact_reply_span")
        if item["evidence"] not in payload["evidence"]:
            raise D5PuzzleContractError("fact_evidence_missing_from_evidence_list")
        if item["source_category"] == "D5_NO_REGULAR" and item["pays_living_costs"] == "yes":
            raise D5PuzzleContractError("no_regular_source_cannot_be_positive_funding")
        fact = PuzzleFact(**item)
        if fact in facts:
            raise D5PuzzleContractError("duplicate_fact")
        if any(f.source_category == fact.source_category and f.evidence == fact.evidence
               and f != fact for f in facts):
            raise D5PuzzleContractError("contradictory_fact_relations")
        facts.append(fact)

    abstain = payload["model_abstain"]
    if abstain:
        if bucket is not None or option is not None:
            raise D5PuzzleContractError("abstention_cannot_invent_semantic_answer")
        if require_diagnostic_evidence and not payload["missing_information"]:
            raise D5PuzzleContractError("abstention_requires_missing_information")
    elif bucket is None:
        raise D5PuzzleContractError("null_bucket_requires_independent_abstention")
    if require_diagnostic_evidence and bucket in {"undecided", "confusion"}:
        if not payload["evidence"] or not payload["missing_information"]:
            raise D5PuzzleContractError("nonclarity_requires_evidence_and_missing_information")
    if bucket == "clarity":
        if option is None or not facts or not payload["evidence"] or payload["missing_information"]:
            raise D5PuzzleContractError("clarity_requires_option_facts_evidence_and_no_missing_information")
        active = {fact.source_category for fact in facts if fact.currently_funds_customer}
        inactive_current = {f.source_category for f in facts if f.owner == "self"
                            and f.time == "current" and f.pays_living_costs == "no"
                            and f.source_category != "D5_NO_REGULAR"}
        if active & inactive_current:
            # The schema contains categories, not an identity for each payer.
            # Opposing present-use relations in one category cannot certify a
            # clear proposal here. A semantic U/C output may retain a conflict.
            raise D5PuzzleContractError("contradictory_current_source_use")
        no_regular = any(f.source_category == "D5_NO_REGULAR" and f.owner == "self"
                         and f.time == "current" and f.pays_living_costs == "no" for f in facts)
        if active and no_regular:
            raise D5PuzzleContractError("contradictory_current_no_regular_and_funding")
        possible_additional = {f.source_category for f in facts
            if f.source_category != "D5_NO_REGULAR" and f.source_category not in active
            and f.owner in {"self", "unknown"} and f.time in {"current", "unknown"}
            and f.pays_living_costs in {"yes", "unknown"}
            and "unknown" in {f.owner, f.time, f.pays_living_costs}}
        if possible_additional and not (option == "D5_MULTIPLE" and len(active) >= 2):
            # Unknown facts are not positive sources. An additional category
            # that could currently fund this customer can still change single
            # source to MULTIPLE, or invalidate NO_REGULAR. With two established
            # categories, another possible category does not change MULTIPLE.
            raise D5PuzzleContractError("additional_source_relation_materially_unknown")
        if option == "D5_MULTIPLE":
            if len(active) < 2:
                raise D5PuzzleContractError("multiple_requires_two_distinct_current_actual_self_sources")
        elif option == "D5_NO_REGULAR":
            if not no_regular or active:
                raise D5PuzzleContractError("no_regular_requires_explicit_current_self_negative_fact")
        elif active != {option}:
            raise D5PuzzleContractError("candidate_requires_exact_current_actual_self_source_support")
    elif option is not None:
        raise D5PuzzleContractError("only_clarity_may_propose_option")
    return D5PuzzleInterpretation(bucket, option, tuple(facts), tuple(payload["evidence"]),
                                  tuple(payload["missing_information"]), abstain)


def load_puzzle_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    payload = _json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise D5PuzzleContractError("config_must_be_object")
    _validate_config(payload)
    return payload


def _validate_config(config: Mapping[str, object]) -> None:
    if config.get("question_id") != "D5" or config.get("semantic_labels") != ["clarity", "undecided", "confusion"]:
        raise D5PuzzleContractError("invalid_config_question_or_labels")
    if set(config.get("allowed_option_ids", [])) != D5_OPTION_IDS:
        raise D5PuzzleContractError("invalid_config_allowed_options")
    if set(config.get("source_categories", [])) != D5_OPTION_IDS - {"D5_MULTIPLE"}:
        raise D5PuzzleContractError("invalid_config_source_categories")
    if config.get("max_calls_per_reply") != 1 or config.get("retry") is not False:
        raise D5PuzzleContractError("one_call_without_retry_required")
    if config.get("conformal_coverage_guarantee") is not False:
        raise D5PuzzleContractError("llm_cannot_claim_cp_guarantee")
    for key in ("version", "prompt_version", "schema_version", "system_prompt"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise D5PuzzleContractError(key + "_required")
    if "require_diagnostic_evidence" in config and type(config["require_diagnostic_evidence"]) is not bool:
        raise D5PuzzleContractError("diagnostic_evidence_policy_must_be_boolean")
    if config.get("grounding_policy") not in {None, "d5-llm-grounding-v3.1"}:
        raise D5PuzzleContractError("unknown_grounding_policy")
    if config.get("require_diagnostic_evidence"):
        if config.get("generation") != {"temperature": 0, "max_tokens": 768, "timeout_seconds": 30}:
            raise D5PuzzleContractError("bounded_pilot_generation_limits_required")


def build_puzzle_messages(spec: BusinessSpec, original_reply: str,
                          config: Mapping[str, object]) -> list[dict[str, str]]:
    """Full original text and current definitions only; no case/answer-key input."""
    if not isinstance(original_reply, str) or not original_reply.strip():
        raise D5PuzzleContractError("empty_reply")
    question = spec.question("D5")
    data = {"current_question": {"id": question.id, "text": question.question},
            "allowed_options": [{"id": o.id, "label": o.label, "definition": o.meaning}
                                for o in question.options], "original_reply": original_reply}
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": canonical_json(data)}]


def _pre_control(question_id: str, text: str) -> QuestionMLResolution | None:
    """Match the engine's pure controls; preserve answer-bearing input for ML.

    Care, pacing and support accompany a financial answer in the engine. Their
    flags do not supply a financial meaning, so they must not swallow baseline
    interpretation here. Explicit undecided replies also reach the ML engine.
    """
    control = check_pre_model_controls(text)
    if not control.matched:
        return None
    pure = control.action in {"safety_stop", "stop", "pause", "resume", "support_control",
                              "explain", "decline", "correction_request"}
    if control.action == "care_pause":
        pure = bool(re.fullmatch(
            r"\s*(?:i(?: am|'m) (?:panicking|overwhelmed|distressed|grieving|upset|anxious)|"
            r"i feel (?:overwhelmed|distressed))\s*[.!?]*\s*", text, re.I))
    elif control.action == "pace_answer":
        pure = bool(re.fullmatch(
            r"\s*i (?:need|want) (?:a |to take a )?(?:break|moment)(?: to think| please)?"
            r"\s*[.!?]*\s*", text, re.I))
    if not pure:
        return None
    return QuestionMLResolution(source="d5_puzzle_pre_model_control",
        reason="pre_model_control_" + (control.action or "matched"), diagnostics={
            "question_id": question_id, "route": "pre_model_control", "llm_called": False,
            "no_cp": True, "conformal_coverage_guarantee": False,
            "pre_model_control": {"action": control.action, "flag": control.flag, "reason": control.reason}})


class D5PuzzleInterpreter:
    """One backend call produces a proposal; no retry, fallback, advice or state writes.

    ``generate(messages, schema)`` is supplied by the caller. This module does
    not import a remote SDK, download a model, or choose local/hosted execution.
    """

    def __init__(self, spec: BusinessSpec,
                 generate: Callable[[list[dict[str, str]], dict[str, object]], str], *,
                 config: Mapping[str, object] | None = None,
                 model_metadata: Mapping[str, object] | None = None):
        self.spec = BusinessSpec(_json(canonical_json(spec.payload)))
        self.config = _json(canonical_json(load_puzzle_config() if config is None else config))
        _validate_config(self.config)
        self.allowed_option_ids = frozenset(o.id for o in self.spec.question("D5").options)
        if self.allowed_option_ids != D5_OPTION_IDS:
            raise D5PuzzleContractError("current_d5_definitions_do_not_match_pilot_contract")
        self.generate = generate
        self.model_metadata = _json(canonical_json(dict(model_metadata or {})))
        self.schema = output_schema(self.allowed_option_ids)
        definitions = build_puzzle_messages(self.spec, "source-hash-placeholder", self.config)[1]
        self.source_hash = sha256_bytes(canonical_json({
            "config": self.config, "spec_version": self.spec.version,
            "definitions": _json(definitions["content"])["allowed_options"],
            "question": self.spec.question("D5").question,
            "schema": self.schema, "model": self.model_metadata,
            "interpreter_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "grounding_source_sha256": sha256_bytes((ROOT / "ai/d5_llm_grounding.py").read_bytes())
                if self.config.get("grounding_policy") else None,
        }).encode("utf-8"))
        self.calls = 0

    def resolve(self, question_id: str, text: str) -> QuestionMLResolution:
        diagnostics: dict[str, object] = {
            "question_id": question_id, "source_hash": self.source_hash,
            "route": "d5_puzzle_llm", "no_cp": True,
            "conformal_coverage_guarantee": False, "llm_called": False,
            "version": self.config["version"], "prompt_version": self.config["prompt_version"],
            "schema_version": self.config["schema_version"], "model": dict(self.model_metadata),
            "evidence_span_verification_is_semantic_guarantee": False}
        if question_id != "D5":
            return QuestionMLResolution(model_uncertain=True, source=SOURCE,
                                        reason="d5_puzzle_question_not_supported", diagnostics=diagnostics)
        if isinstance(text, str):
            control = _pre_control(question_id, text)
            if control is not None:
                return control
        try:
            messages = build_puzzle_messages(self.spec, text, self.config)
            diagnostics["messages_sha256"] = sha256_bytes(canonical_json(messages).encode("utf-8"))
            self.calls += 1
            diagnostics["llm_called"] = True
            # Detached value objects prevent backend mutation of the contract.
            raw = self.generate(_json(canonical_json(messages)), _json(canonical_json(self.schema)))
            diagnostics["raw_output"] = raw if isinstance(raw, str) else None
            call = getattr(self.generate, "last_call", None)
            if isinstance(call, Mapping):
                diagnostics["backend_call"] = _json(canonical_json(dict(call)))
                if self.config.get("require_diagnostic_evidence") and call.get("token_limit_reached"):
                    raise D5PuzzleContractError("generation_token_limit_reached")
            parsed = parse_puzzle_interpretation(raw, text, self.allowed_option_ids,
                require_diagnostic_evidence=self.config.get("require_diagnostic_evidence", False))
            diagnostics["interpretation"] = parsed.as_dict()
            diagnostics["facts"] = [f.as_dict() for f in parsed.facts]
            if self.config.get("grounding_policy") and parsed.semantic_bucket == "clarity":
                from ai.d5_llm_grounding import grounding_violations
                violations = grounding_violations(text, diagnostics["facts"], parsed.option_id)
                diagnostics["grounding_policy"] = self.config["grounding_policy"]
                diagnostics["grounding_violations"] = list(violations)
                if violations:
                    raise D5PuzzleContractError("grounding:" + ",".join(violations))
            return QuestionMLResolution(candidate_option_id=parsed.option_id,
                ordinary_bucket=BUCKET_NAMES.get(parsed.semantic_bucket),
                model_uncertain=parsed.model_abstain,
                semantic_undecided=parsed.semantic_bucket == "undecided", source=SOURCE,
                reason="d5_puzzle_independent_abstention" if parsed.model_abstain else
                       "d5_puzzle_candidate_requires_confirmation" if parsed.semantic_bucket == "clarity" else
                       "d5_puzzle_semantic_" + parsed.semantic_bucket, diagnostics=diagnostics)
        except (D5PuzzleContractError, KeyError, TypeError) as exc:
            diagnostics["contract_error"] = str(exc)
            reason = "d5_puzzle_contract_rejected"
        except Exception as exc:
            diagnostics["backend_error_type"] = type(exc).__name__
            call = getattr(self.generate, "last_call", None)
            if isinstance(call, Mapping):
                diagnostics["backend_call"] = _json(canonical_json(dict(call)))
            reason = "d5_puzzle_backend_timeout" if isinstance(exc, TimeoutError) else "d5_puzzle_backend_failed"
        return QuestionMLResolution(model_uncertain=True, source=SOURCE,
                                    reason=reason, diagnostics=diagnostics)


class D5PuzzleCascade:
    """Preserve baseline outcomes; route only actual D5 abstention to the LLM."""

    def __init__(self, baseline, interpreter: D5PuzzleInterpreter):
        self.baseline = baseline
        self.interpreter = interpreter
        self.source_hash = sha256_bytes(canonical_json({
            "baseline_source_hash": getattr(baseline, "source_hash", ""),
            "interpreter_source_hash": interpreter.source_hash,
            "routing": "d5_actual_model_uncertain_without_valid_candidate_v1",
        }).encode("utf-8"))

    def resolve(self, question_id: str, text: str) -> QuestionMLResolution:
        if question_id == "D5" and isinstance(text, str):
            control = _pre_control(question_id, text)
            if control is not None:
                return control
        baseline = self.baseline.resolve(question_id, text)
        if (question_id != "D5" or not baseline.model_uncertain
                or baseline.candidate_option_id in self.interpreter.allowed_option_ids):
            return baseline
        interpreted = self.interpreter.resolve(question_id, text)
        diagnostics = dict(interpreted.diagnostics)
        diagnostics["baseline"] = {
            "source": baseline.source, "source_hash": baseline.diagnostics.get("source_hash"),
            "candidate_option_id": baseline.candidate_option_id,
            "semantic_bucket": baseline.ordinary_bucket, "model_uncertain": baseline.model_uncertain,
            "reason": baseline.reason, "intent_set": sorted(baseline.intent_set),
            "option_set": sorted(baseline.option_set),
            "intent_probabilities": dict(baseline.intent_probabilities),
            "option_probabilities": dict(baseline.option_probabilities),
            "diagnostics": dict(baseline.diagnostics)}
        diagnostics["cascade_source_hash"] = self.source_hash
        diagnostics["route"] = "baseline_abstention_to_d5_puzzle_llm"
        return replace(interpreted, diagnostics=diagnostics)
