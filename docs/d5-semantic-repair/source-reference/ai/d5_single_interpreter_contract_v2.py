"""Isolated D5 v2: category/relation groups and authored clarification topics.

Nothing here examines funding words to derive a second semantic answer. Exact
quotation and internally consistent facts cannot prove semantic truth. The
frozen development and fresh evaluations must measure that separately.

This file is not wired into the normal demo. The injected backend owns the
existing complete-response, tool-free, bounded, no-retry transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from ai.all_question_ml import QuestionMLResolution
from ai.d5_puzzle_interpreter import D5_OPTION_IDS, _pre_control
from ai.llm_shadow import BUCKET_NAMES, ShadowContractError, canonical_json, sha256_bytes, strict_json
from ai.question_spec import BusinessSpec


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "config/d5-single-interpreter-v2-prompt.txt"
CONTRACT_VERSION = "d5-single-interpreter-v2"
SOURCE = "d5_single_interpreter_v2"
MAX_INPUT_CHARS = 32768
MAX_OUTPUT_BYTES = 65536
MAX_FACTS = 24
MAX_EVIDENCE = 32
MAX_MISSING = 8
SCHEMA_KEYS = frozenset({"semantic_bucket", "option_id", "facts", "evidence",
    "missing_information", "model_abstain", "answer_scope"})
FACT_KEYS = frozenset({"source_instance_id", "source_category", "owner", "time",
    "pays_living_costs", "evidence"})
BUCKETS = frozenset({"clarity", "undecided", "confusion"})
OWNERS = frozenset({"self", "other", "unknown"})
TIMES = frozenset({"current", "future", "past", "unknown"})
USES = frozenset({"yes", "no", "unknown"})
SCOPES = frozenset({"complete", "partial", "unknown"})
# These are question topics, not asserted financial facts or model-authored text.
SOURCE_USE_TOPICS = frozenset(
    "source_use_unknown:" + option for option in D5_OPTION_IDS
    if option not in {"D5_MULTIPLE", "D5_NO_REGULAR"})
MISSING_TOPICS = SOURCE_USE_TOPICS | frozenset({
    "current_funding_unknown", "complete_answer_needed", "question_meaning", "reply_unclear"})


class D5SingleContractError(ShadowContractError):
    """A fixed structural rejection; output is never repaired."""


def _hash(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def output_schema(allowed_option_ids: frozenset[str] = D5_OPTION_IDS) -> dict:
    if frozenset(allowed_option_ids) != D5_OPTION_IDS:
        raise D5SingleContractError("unchanged_d5_option_ids_required")
    text_array = {"type": "array", "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": MAX_INPUT_CHARS}}
    return {"type": "object", "additionalProperties": False,
        "required": sorted(SCHEMA_KEYS), "properties": {
            "semantic_bucket": {"type": ["string", "null"], "enum": sorted(BUCKETS) + [None]},
            "option_id": {"type": ["string", "null"], "enum": sorted(D5_OPTION_IDS) + [None]},
            "answer_scope": {"type": "string", "enum": sorted(SCOPES)},
            "model_abstain": {"type": "boolean"},
            "facts": {"type": "array", "maxItems": MAX_FACTS, "uniqueItems": True,
                "items": {"type": "object", "additionalProperties": False,
                    "required": sorted(FACT_KEYS), "properties": {
                        "source_instance_id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "source_category": {"type": "string", "enum": sorted(D5_OPTION_IDS - {"D5_MULTIPLE"})},
                        "owner": {"type": "string", "enum": sorted(OWNERS)},
                        "time": {"type": "string", "enum": sorted(TIMES)},
                        "pays_living_costs": {"type": "string", "enum": sorted(USES)},
                        "evidence": {"type": "string", "minLength": 1, "maxLength": MAX_INPUT_CHARS}}}},
            "evidence": dict(text_array, maxItems=MAX_EVIDENCE),
            "missing_information": dict(text_array, maxItems=MAX_MISSING)}}


@dataclass(frozen=True)
class D5SourceFact:
    source_instance_id: str
    source_category: str
    owner: str
    time: str
    pays_living_costs: str
    evidence: str

    @property
    def currently_funds_customer(self) -> bool:
        return (self.source_category != "D5_NO_REGULAR" and self.owner == "self"
            and self.time == "current" and self.pays_living_costs == "yes")

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in sorted(FACT_KEYS)}


@dataclass(frozen=True)
class D5Interpretation:
    semantic_bucket: str | None
    option_id: str | None
    facts: tuple[D5SourceFact, ...]
    evidence: tuple[str, ...]
    missing_information: tuple[str, ...]
    model_abstain: bool
    answer_scope: str

    def as_dict(self) -> dict:
        return {"semantic_bucket": self.semantic_bucket, "option_id": self.option_id,
            "facts": [fact.as_dict() for fact in self.facts], "evidence": list(self.evidence),
            "missing_information": list(self.missing_information),
            "model_abstain": self.model_abstain, "answer_scope": self.answer_scope}


def _input(text: str) -> None:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_INPUT_CHARS:
        raise D5SingleContractError("invalid_or_oversized_input")


def _strings(value: object, name: str, limit: int) -> tuple[str, ...]:
    if (not isinstance(value, list) or len(value) > limit
            or any(not isinstance(item, str) or not item.strip()
                   or len(item) > MAX_INPUT_CHARS for item in value)):
        raise D5SingleContractError(name + "_invalid_or_oversized")
    if len(value) != len(set(value)):
        raise D5SingleContractError(name + "_duplicates")
    return tuple(value)


def parse_interpretation(raw: str, text: str,
                         allowed_option_ids: frozenset[str] = D5_OPTION_IDS) -> D5Interpretation:
    """Reject structural, provenance and internal-contract errors, never relabel.

    Source IDs identify evidenced category/owner/time/use groups. A group may
    contain several payers with the same relations. Categories determine the
    answer; no source count or semantic entailment is inferred from wording.
    """
    _input(text)
    if frozenset(allowed_option_ids) != D5_OPTION_IDS:
        raise D5SingleContractError("unchanged_d5_option_ids_required")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise D5SingleContractError("invalid_or_oversized_output")
    try:
        data = strict_json(raw)
    except (ShadowContractError, RecursionError) as exc:
        reason = str(exc) if isinstance(exc, ShadowContractError) else "json_nesting_limit"
        raise D5SingleContractError(reason) from exc
    if not isinstance(data, dict) or set(data) != SCHEMA_KEYS:
        raise D5SingleContractError("exact_schema_keys_required")
    bucket, option, scope = data["semantic_bucket"], data["option_id"], data["answer_scope"]
    if bucket is not None and (not isinstance(bucket, str) or bucket not in BUCKETS):
        raise D5SingleContractError("invalid_semantic_bucket")
    if option is not None and (not isinstance(option, str) or option not in D5_OPTION_IDS):
        raise D5SingleContractError("unapproved_option_id")
    if not isinstance(scope, str) or scope not in SCOPES:
        raise D5SingleContractError("invalid_answer_scope")
    if type(data["model_abstain"]) is not bool:
        raise D5SingleContractError("model_abstain_must_be_boolean")
    evidence = _strings(data["evidence"], "evidence", MAX_EVIDENCE)
    missing = _strings(data["missing_information"], "missing_information", MAX_MISSING)
    if any(quote not in text for quote in evidence):
        raise D5SingleContractError("evidence_is_not_exact_reply_span")
    if not isinstance(data["facts"], list) or len(data["facts"]) > MAX_FACTS:
        raise D5SingleContractError("facts_invalid_or_oversized")
    facts: list[D5SourceFact] = []
    ids: set[str] = set()
    signatures: set[tuple] = set()
    for item in data["facts"]:
        if not isinstance(item, dict) or set(item) != FACT_KEYS:
            raise D5SingleContractError("exact_fact_keys_required")
        if any(not isinstance(item[key], str) for key in FACT_KEYS):
            raise D5SingleContractError("fact_values_must_be_strings")
        source_id = item["source_instance_id"]
        if (not source_id or len(source_id) > 64 or not source_id.isascii()
                or any(not (char.isalnum() or char in "_-") for char in source_id)):
            raise D5SingleContractError("invalid_source_instance_id")
        if source_id in ids:
            raise D5SingleContractError("duplicate_source_instance_id")
        ids.add(source_id)
        if item["source_category"] not in D5_OPTION_IDS - {"D5_MULTIPLE"}:
            raise D5SingleContractError("unapproved_source_category")
        if (item["owner"] not in OWNERS or item["time"] not in TIMES
                or item["pays_living_costs"] not in USES):
            raise D5SingleContractError("invalid_fact_relation")
        quote = item["evidence"]
        if not quote.strip() or quote not in text or quote not in evidence:
            raise D5SingleContractError("fact_evidence_not_exact_listed_span")
        if item["source_category"] == "D5_NO_REGULAR" and item["pays_living_costs"] != "no":
            raise D5SingleContractError("no_regular_requires_negative_cost_use")
        # Identical groups remain duplicates even if quoted using different spans.
        signature = tuple(item[key] for key in sorted(
            FACT_KEYS - {"source_instance_id", "evidence"}))
        if signature in signatures:
            raise D5SingleContractError("duplicate_fact_under_different_id")
        signatures.add(signature)
        fact = D5SourceFact(**item)
        if any(previous.source_category == fact.source_category
               and previous.evidence == fact.evidence
               and (previous.owner, previous.time, previous.pays_living_costs)
               != (fact.owner, fact.time, fact.pays_living_costs) for previous in facts):
            raise D5SingleContractError("contradictory_fact_relations")
        facts.append(fact)
    abstain = data["model_abstain"]
    if abstain:
        if bucket is not None or option is not None or facts:
            raise D5SingleContractError("abstention_cannot_contain_semantic_answer_or_facts")
        if not missing or scope == "complete":
            raise D5SingleContractError("abstention_requires_missing_information_and_incomplete_scope")
    elif bucket is None:
        raise D5SingleContractError("null_bucket_requires_independent_abstention")
    elif bucket in {"undecided", "confusion"}:
        if option is not None or facts:
            raise D5SingleContractError("dialogue_only_cannot_contain_financial_option_or_facts")
        if not evidence or not missing:
            raise D5SingleContractError("dialogue_only_requires_evidence_and_missing_information")
    else:
        if not option or not facts or not evidence:
            raise D5SingleContractError("clarity_requires_option_facts_and_evidence")
        if scope != "complete":
            raise D5SingleContractError("complete_funding_answer_required")
        material_unknown = any(
            fact.owner in {"self", "unknown"} and fact.time in {"current", "unknown"}
            and fact.pays_living_costs in {"yes", "unknown"}
            and "unknown" in (fact.owner, fact.time, fact.pays_living_costs)
            for fact in facts)
        if missing or material_unknown:
            raise D5SingleContractError("clarity_contains_unresolved_information")
        active = {fact.source_category for fact in facts if fact.currently_funds_customer}
        no_regular = any(fact.source_category == "D5_NO_REGULAR" and fact.owner == "self"
            and fact.time == "current" and fact.pays_living_costs == "no" for fact in facts)
        if no_regular and active:
            raise D5SingleContractError("contradictory_no_regular_and_active_funding")
        if option == "D5_MULTIPLE":
            if len(active) < 2:
                raise D5SingleContractError("multiple_requires_distinct_active_categories")
        elif option == "D5_NO_REGULAR":
            if not no_regular or active:
                raise D5SingleContractError("no_regular_requires_current_self_negative_fact")
        elif active != {option}:
            raise D5SingleContractError("option_disagrees_with_active_categories")
    if bucket != "clarity":
        if len(missing) != 1 or missing[0] not in MISSING_TOPICS:
            raise D5SingleContractError("one_authored_clarification_topic_required")
        if bucket == "confusion" and missing[0] != "question_meaning":
            raise D5SingleContractError("confusion_requires_question_meaning_topic")
        if bucket == "undecided" and missing[0] not in (
                SOURCE_USE_TOPICS | {"current_funding_unknown"}):
            raise D5SingleContractError("undecided_requires_funding_uncertainty_topic")
        if abstain and missing[0] == "question_meaning":
            raise D5SingleContractError("meaning_question_is_not_model_abstention")
    return D5Interpretation(bucket, option, tuple(facts), evidence, missing, abstain, scope)


def _spec_hash(spec: BusinessSpec) -> str:
    if frozenset(option.id for option in spec.question("D5").options) != D5_OPTION_IDS:
        raise D5SingleContractError("unchanged_d5_option_ids_required")
    return _hash(spec.payload)


def _contract_hash(prompt: str) -> str:
    return _hash({"version": CONTRACT_VERSION, "prompt": prompt, "schema": output_schema(),
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes())})


def build_messages(spec: BusinessSpec, text: str, prompt: str | None = None) -> list[dict[str, str]]:
    _input(text)
    _spec_hash(spec)
    question = spec.question("D5")
    return [{"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8") if prompt is None else prompt},
        {"role": "user", "content": canonical_json({
            "current_question": {"id": question.id, "text": question.question},
            "allowed_options": [{"id": option.id, "label": option.label, "definition": option.meaning}
                for option in question.options], "original_reply": text})}]


@dataclass(frozen=True)
class D5ContractDecision:
    outcome: str
    reason: str
    input_sha256: str
    proposal_sha256: str
    spec_sha256: str
    contract_sha256: str
    raw_output: str
    interpretation: D5Interpretation | None

    def as_dict(self) -> dict:
        """Bounded metadata only; rejected financial material is never exposed."""
        parsed = self.interpretation
        return {"outcome": self.outcome, "reason": self.reason,
            "semantic_bucket": parsed.semantic_bucket if parsed else None,
            "option_id": parsed.option_id if parsed else None,
            "answer_scope": parsed.answer_scope if parsed else None,
            "input_sha256": self.input_sha256, "proposal_sha256": self.proposal_sha256,
            "spec_sha256": self.spec_sha256, "contract_sha256": self.contract_sha256,
            "contract_version": CONTRACT_VERSION}


def decide(raw: str, text: str, spec: BusinessSpec, *, prompt: str | None = None) -> D5ContractDecision:
    """Standalone deterministic evaluation; not an issued application decision."""
    _input(text)
    if not isinstance(raw, str):
        raise D5SingleContractError("output_must_be_text")
    frozen_prompt = PROMPT_PATH.read_text(encoding="utf-8") if prompt is None else prompt
    try:
        parsed = parse_interpretation(raw, text)
        outcome = ("abstention" if parsed.model_abstain else
            "financial_candidate" if parsed.semantic_bucket == "clarity" else "dialogue_only")
        reason = {"abstention": "independent_model_abstention", "financial_candidate": "candidate_requires_confirmation",
            "dialogue_only": "authored_dialogue_only"}[outcome]
    except D5SingleContractError as exc:
        parsed, outcome, reason = None, "rejection", str(exc)
    return D5ContractDecision(outcome, reason, _text_hash(text), _text_hash(raw),
        _spec_hash(spec), _contract_hash(frozen_prompt), raw, parsed)


@dataclass(frozen=True)
class D5SingleResolution(QuestionMLResolution):
    contract_decision: D5ContractDecision | None = None


class D5SingleInterpreter:
    """Exactly one backend call per eligible answer, without retry or fallback.

    The engine must bind the issued immutable decision to its own submission
    and state. Later validation reuses this object and makes no inference call.
    A backend failure permanently halts this interpreter instance.
    """

    def __init__(self, spec: BusinessSpec,
                 generate: Callable[[list[dict[str, str]], dict], str], *,
                 prompt: str | None = None, model_metadata: Mapping | None = None):
        self.spec = BusinessSpec(strict_json(canonical_json(spec.payload)))
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8") if prompt is None else prompt
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise D5SingleContractError("nonempty_prompt_required")
        self.spec_sha256 = _spec_hash(self.spec)
        self.contract_sha256 = _contract_hash(self.prompt)
        self.source_hash = _hash({"contract": self.contract_sha256, "spec": self.spec_sha256,
            "model": dict(model_metadata or {})})
        self.generate = generate
        self.calls = 0
        self.halted = False
        self.last_decision: D5ContractDecision | None = None
        self._issued: dict[int, D5ContractDecision] = {}

    def verify_decision(self, decision: D5ContractDecision, text: str, *,
                        spec: BusinessSpec | None = None, option_id: str | None = None) -> bool:
        """Verify issued identity and recompute; mutable diagnostics confer no trust."""
        if not isinstance(decision, D5ContractDecision) or self._issued.get(id(decision)) is not decision:
            return False
        try:
            current_spec = self.spec if spec is None else spec
            if (_spec_hash(current_spec) != self.spec_sha256
                    or _spec_hash(self.spec) != self.spec_sha256
                    or _contract_hash(self.prompt) != self.contract_sha256):
                return False
            if option_id is not None and (decision.outcome != "financial_candidate"
                    or decision.interpretation is None or decision.interpretation.option_id != option_id):
                return False
            return decision == decide(decision.raw_output, text, current_spec, prompt=self.prompt)
        except (D5SingleContractError, KeyError, TypeError, ValueError):
            return False

    def resolve(self, question_id: str, text: str) -> QuestionMLResolution:
        self.last_decision = None
        diagnostic: dict = {"question_id": question_id, "source_hash": self.source_hash,
            "route": SOURCE, "contract_version": CONTRACT_VERSION, "llm_called": False,
            "no_cp": True, "conformal_coverage_guarantee": False,
            "evidence_span_verification_is_semantic_guarantee": False}
        if question_id != "D5":
            return D5SingleResolution(model_uncertain=True, source=SOURCE,
                reason="d5_question_not_supported", diagnostics=diagnostic)
        if isinstance(text, str):
            control = _pre_control(question_id, text)
            if control is not None:
                return control
        if self.halted:
            diagnostic["backend_failed"] = True
            return D5SingleResolution(model_uncertain=True, source=SOURCE,
                reason="d5_transport_halted", diagnostics=diagnostic)
        try:
            messages = build_messages(self.spec, text, self.prompt)
            if _spec_hash(self.spec) != self.spec_sha256 or _contract_hash(self.prompt) != self.contract_sha256:
                raise D5SingleContractError("contract_or_spec_changed")
        except D5SingleContractError as exc:
            diagnostic["contract_error"] = str(exc)
            return D5SingleResolution(model_uncertain=True, source=SOURCE,
                reason="d5_contract_rejected", diagnostics=diagnostic)
        self.calls += 1
        diagnostic["llm_called"] = True
        try:
            raw = self.generate(strict_json(canonical_json(messages)), output_schema())
        except Exception as exc:
            self.halted = True
            diagnostic.update(backend_failed=True, backend_error_type=type(exc).__name__)
            return D5SingleResolution(model_uncertain=True, source=SOURCE,
                reason="d5_backend_failed", diagnostics=diagnostic)
        if not isinstance(raw, str):
            diagnostic["contract_error"] = "output_must_be_text"
            return D5SingleResolution(model_uncertain=True, source=SOURCE,
                reason="d5_contract_rejected", diagnostics=diagnostic)
        # Evaluation can inspect the raw model answer separately from acceptance.
        # Neither this diagnostic dictionary nor its contents authorise a fact.
        if len(raw.encode("utf-8")) <= MAX_OUTPUT_BYTES:
            diagnostic["raw_output"] = raw
        decision = decide(raw, text, self.spec, prompt=self.prompt)
        self._issued[id(decision)] = decision
        self.last_decision = decision
        diagnostic["contract_decision"] = decision.as_dict()
        parsed = decision.interpretation
        if parsed is None:
            diagnostic["contract_error"] = decision.reason
        else:
            diagnostic["interpretation"] = parsed.as_dict()
        return D5SingleResolution(
            candidate_option_id=parsed.option_id if parsed else None,
            ordinary_bucket=BUCKET_NAMES.get(parsed.semantic_bucket) if parsed else None,
            model_uncertain=decision.outcome in {"abstention", "rejection"},
            semantic_undecided=bool(parsed and parsed.semantic_bucket == "undecided"),
            reason=decision.reason, source=SOURCE, diagnostics=diagnostic,
            contract_decision=decision)
