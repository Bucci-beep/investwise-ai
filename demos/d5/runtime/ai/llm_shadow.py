"""Opt-in, local-only meaning interpretation; never confirms or saves answers.

The JSON contract is deliberately separate from conformal prediction. Empty
probability maps and prediction sets mean no CP evidence exists; they are not
fabricated singleton sets. The ordinary ConversationEngine remains the owner
of controls, business vetoes, confirmation and recording.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ai.all_question_ml import QuestionMLResolution
from ai.question_spec import BusinessSpec


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config/llm-shadow-v1.json"
SCHEMA_KEYS = frozenset({"bucket", "option_id", "evidence", "missing_information", "model_abstain"})
SEMANTIC_LABELS = frozenset({"clarity", "undecided", "confusion"})
BUCKET_NAMES = {"clarity": "Clarity", "undecided": "Undecided", "confusion": "Confusion"}


class ShadowContractError(ValueError):
    """Generated output or input violates the frozen interpretation contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ShadowContractError("duplicate_json_key")
        result[key] = value
    return result


def strict_json(payload: str):
    if not isinstance(payload, str):
        raise ShadowContractError("output_must_be_text")
    try:
        return json.loads(payload, object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ShadowContractError("nonfinite_json_number")))
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ShadowContractError):
            raise
        raise ShadowContractError("invalid_json") from exc


@dataclass(frozen=True)
class ShadowInterpretation:
    bucket: str | None
    option_id: str | None
    evidence: tuple[str, ...]
    missing_information: tuple[str, ...]
    model_abstain: bool

    def as_dict(self) -> dict[str, object]:
        return {"bucket": self.bucket, "option_id": self.option_id,
                "evidence": list(self.evidence), "missing_information": list(self.missing_information),
                "model_abstain": self.model_abstain}


def parse_interpretation(raw_output: str, original_reply: str,
                         allowed_option_ids: frozenset[str]) -> ShadowInterpretation:
    """Reject malformed/unsupported output without a financial fallback."""
    payload = strict_json(raw_output)
    if not isinstance(payload, dict) or set(payload) != SCHEMA_KEYS:
        raise ShadowContractError("exact_schema_keys_required")
    bucket, option = payload["bucket"], payload["option_id"]
    if bucket is not None and (not isinstance(bucket, str) or bucket not in SEMANTIC_LABELS):
        raise ShadowContractError("invalid_semantic_bucket")
    if option is not None and (not isinstance(option, str) or option not in allowed_option_ids):
        raise ShadowContractError("unapproved_option_id")
    if type(payload["model_abstain"]) is not bool:
        raise ShadowContractError("model_abstain_must_be_boolean")
    for key in ("evidence", "missing_information"):
        values = payload[key]
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ShadowContractError(key + "_must_be_nonempty_strings")
        if len(values) != len(set(values)):
            raise ShadowContractError(key + "_contains_duplicates")
    if any(span not in original_reply for span in payload["evidence"]):
        raise ShadowContractError("evidence_is_not_exact_reply_span")
    abstain = payload["model_abstain"]
    if abstain:
        if bucket is not None or option is not None:
            raise ShadowContractError("abstention_cannot_invent_semantic_answer")
    elif bucket is None:
        raise ShadowContractError("null_bucket_requires_independent_abstention")
    if bucket == "clarity":
        if option is None or not payload["evidence"] or payload["missing_information"]:
            raise ShadowContractError("clarity_requires_option_evidence_and_no_missing_fact")
    elif option is not None:
        raise ShadowContractError("only_clarity_may_propose_option")
    return ShadowInterpretation(bucket, option, tuple(payload["evidence"]),
                                tuple(payload["missing_information"]), abstain)


@dataclass(frozen=True)
class ConfirmedContext:
    question_id: str
    option_id: str


def load_shadow_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    payload = strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("semantic_labels") != ["clarity", "undecided", "confusion"]:
        raise ShadowContractError("invalid_config_labels")
    if payload.get("backend") != "mlx_lm_local_only" or payload.get("generation") != {"temperature": 0, "max_tokens": 256}:
        raise ShadowContractError("frozen_local_greedy_generation_required")
    if not isinstance(payload.get("system_prompt"), str) or not payload["system_prompt"].strip():
        raise ShadowContractError("system_prompt_required")
    allowlist = payload.get("confirmed_context_allowlist")
    if not isinstance(allowlist, dict) or any(not isinstance(k, str) or not isinstance(v, list)
                                            or any(not isinstance(qid, str) for qid in v)
                                            for k, v in allowlist.items()):
        raise ShadowContractError("invalid_context_allowlist")
    return payload


def build_messages(spec: BusinessSpec, question_id: str, original_reply: str,
                   config: Mapping[str, object], confirmed_context: Sequence[ConfirmedContext] = ()) -> list[dict[str, str]]:
    """Accept only frozen question/option definitions and approved confirmed IDs.

    Context is a value object, never an engine, store or conversation reference.
    No label, answer key, confidence or case metadata is accepted by this API.
    """
    if not isinstance(original_reply, str) or not original_reply.strip():
        raise ShadowContractError("empty_reply")
    question = spec.question(question_id)
    allowed_context = set(config["confirmed_context_allowlist"].get(question_id, []))
    seen = set()
    contexts = []
    for context in confirmed_context:
        if not isinstance(context, ConfirmedContext) or context.question_id not in allowed_context or context.question_id in seen:
            raise ShadowContractError("unapproved_or_duplicate_confirmed_context")
        option = spec.option(context.question_id, context.option_id)
        prior = spec.question(context.question_id)
        contexts.append({"question_id": prior.id, "question": prior.question,
                         "option_id": option.id, "definition": option.meaning})
        seen.add(context.question_id)
    data = {"current_question": {"id": question.id, "text": question.question},
            "allowed_options": [{"id": option.id, "label": option.label, "definition": option.meaning}
                                for option in question.options],
            "original_reply": original_reply, "approved_confirmed_context": contexts}
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": canonical_json(data)}]


class LLMShadowResolver:
    """Generate a proposal with the existing resolution attributes, no recording.

    A backend receives message data only. No mutable engine or store reference
    is retained. Backend errors and invalid JSON become independent abstention.
    """
    def __init__(self, spec: BusinessSpec, generate: Callable[[list[dict[str, str]]], str], *,
                 config: Mapping[str, object] | None = None,
                 confirmed_context: Mapping[str, Sequence[ConfirmedContext]] | None = None,
                 model_metadata: Mapping[str, str] | None = None):
        self.spec = spec
        self.generate = generate
        # Copy external mappings through JSON so later mutation cannot tune a run.
        self.config = strict_json(canonical_json(config or load_shadow_config()))
        self.confirmed_context = {qid: tuple(values) for qid, values in (confirmed_context or {}).items()}
        self.model_metadata = dict(model_metadata or {})
        self.source_hash = sha256_bytes(canonical_json({"config": self.config,
                         "spec": self.spec.payload, "model": self.model_metadata}).encode("utf-8"))
        self.calls = 0

    def resolve(self, question_id: str, text: str) -> QuestionMLResolution:
        diagnostics: dict[str, object] = {"source_hash": self.source_hash, "question_id": question_id,
            "llm_shadow": True, "no_cp": True, "conformal_coverage_guarantee": False,
            "prompt_version": self.config["prompt_version"], "model": dict(self.model_metadata)}
        try:
            messages = build_messages(self.spec, question_id, text, self.config,
                                      self.confirmed_context.get(question_id, ()))
            diagnostics["messages_sha256"] = sha256_bytes(canonical_json(messages).encode("utf-8"))
            self.calls += 1
            raw = self.generate(messages)
            diagnostics["raw_output"] = raw if isinstance(raw, str) else None
            parsed = parse_interpretation(raw, text, frozenset(o.id for o in self.spec.question(question_id).options))
            diagnostics["interpretation"] = parsed.as_dict()
            return QuestionMLResolution(candidate_option_id=parsed.option_id,
                ordinary_bucket=BUCKET_NAMES.get(parsed.bucket), model_uncertain=parsed.model_abstain,
                semantic_undecided=parsed.bucket == "undecided", source="llm_shadow",
                reason="llm_shadow_independent_abstention" if parsed.model_abstain else
                       "llm_shadow_candidate_requires_confirmation" if parsed.bucket == "clarity" else
                       "llm_shadow_semantic_" + parsed.bucket,
                diagnostics=diagnostics)
        except (ShadowContractError, KeyError, TypeError) as exc:
            diagnostics["contract_error"] = str(exc)
            reason = "llm_shadow_contract_rejected"
        except Exception as exc:
            # Failures remain in the immutable runner trace; no second generation
            # or non-LLM interpretation is attempted for this reply.
            diagnostics["backend_error_type"] = type(exc).__name__
            diagnostics["backend_error"] = str(exc)
            reason = "llm_shadow_backend_failed"
        return QuestionMLResolution(model_uncertain=True, source="llm_shadow", reason=reason,
                                    diagnostics=diagnostics)


class MLXLocalGenerator:
    """Load an already present snapshot lazily; never calls a hosted API."""
    def __init__(self, model_path: str | Path, *, max_tokens: int = 256):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir() or not (self.model_path / "config.json").is_file():
            raise ValueError("A local MLX model snapshot with config.json is required")
        if max_tokens != 256:
            raise ValueError("The frozen generation limit is 256")
        self.max_tokens = max_tokens
        self._loaded = None

    def __call__(self, messages: list[dict[str, str]]) -> str:
        if self._loaded is None:
            from mlx_lm import load
            self._loaded = load(str(self.model_path))
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = self._loaded
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(model, tokenizer, prompt=prompt, max_tokens=self.max_tokens,
                        sampler=make_sampler(temp=0), verbose=False)


def local_snapshot(model_id: str, revision: str, model_path: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A pinned 40-character lowercase model commit is required")
    if model_path is not None:
        return Path(model_path).expanduser().resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id, revision=revision, local_files_only=True)).resolve()
