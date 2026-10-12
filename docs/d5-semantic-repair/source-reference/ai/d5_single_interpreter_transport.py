"""Versioned D5 schema over the existing bounded, tool-free OAuth transport.

Only the passive schema/prompt change. HTTP compatibility, stream assembly,
complete response, EOF, metadata, credential and local limits are reused.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import threading
import time

from ai import d5_chatgpt_plan_transport as frozen
from ai.d5_chatgpt_missing_mime import MissingMimeChatGPTPlanGenerator
from ai.d5_chatgpt_stream_assembly import normalised_chunks
from ai.d5_chatgpt_plan_diagnostic import failure_details
from ai.d5_puzzle_interpreter import D5_OPTION_IDS, output_schema as old_schema
from ai.d5_single_interpreter_contract import output_schema
from ai.d5_tool_free_gpt_experiment import generation_schema
from ai.llm_shadow import canonical_json


def request_payload(messages, schema):
    if canonical_json(schema) != canonical_json(output_schema()):
        raise ValueError("Only the frozen single-interpreter schema is permitted")
    # Reuse the exact input/roles/D5/options boundary, then explicitly version
    # the newly authorised passive schema. No tools or context are added.
    payload = frozen.request_payload(messages, old_schema(D5_OPTION_IDS))
    payload["text"]["format"] = {
        "type": "json_schema", "name": "d5_single_interpreter_v1", "strict": True,
        "schema": generation_schema(deepcopy(schema)),
    }
    return payload


class SingleInterpreterGenerator(MissingMimeChatGPTPlanGenerator):
    """One POST per case; successful passive JSON only, never output repair."""

    def __call__(self, messages, schema):
        payload = request_payload(messages, schema)
        if not self.case_id or not re.fullmatch(r"[A-Za-z0-9_-]+", self.case_id):
            raise ValueError("A unique safe case ID is required")
        target = self.trace_root / self.case_id
        target.mkdir(parents=True, exist_ok=False)
        http = {"lock": threading.Lock(), "sealed": False, "http": {}}
        assembly = {"lock": threading.Lock(), "sealed": False,
                    "detail": {"version": 1, "attempted": False, "applied": False}}
        self._observation = http
        self.last_call = {
            "backend": "single_interpreter_existing_chatgpt_plan_stream",
            "endpoint": frozen.ENDPOINT, "model_requested": frozen.MODEL,
            "reasoning_effort_requested": frozen.EFFORT, "tools": [],
            "tool_choice": "none", "store": False, "stream": True,
            "timeout_seconds": frozen.TIMEOUT_SECONDS,
            "max_response_bytes": frozen.MAX_RESPONSE_BYTES,
            "provider_output_token_cap": None, "actual_calls": 0,
            "retry": False, "output_repair": False, "fresh_context": True,
            "tool_dispatcher_present": False, "blocked": False,
        }
        started = time.monotonic()
        deadline = started + frozen.TIMEOUT_SECONDS
        phase, secrets = "credentials", ()
        try:
            token, secrets = frozen._credentials(self.credential_provider, deadline)
            if (not isinstance(token, str) or not token
                    or any(not isinstance(s, str) for s in secrets)):
                raise frozen.RouteBlocked("Invalid credential provider result")
            phase = "request"
            encoded = canonical_json(payload)
            if any(secret and secret in encoded for secret in secrets):
                raise frozen.RouteBlocked("Credential reflection in request")
            frozen._remaining(deadline)
            (target / "request.json").write_text(encoded, encoding="utf-8")
            self.calls += 1
            self.last_call["actual_calls"] = 1
            send = self._injected_transport or self._http_sse
            chunks = frozen._bounded_chunks(send, payload, token, deadline, secrets)
            phase = "stream"
            try:
                response = frozen.completed_response(
                    normalised_chunks(chunks, deadline, assembly), deadline, secrets)
            finally:
                chunks.close()
            phase = "passive_output"
            raw = frozen.passive_text(response)
            frozen._remaining(deadline)
            serialised = canonical_json(response)
            if any(secret and secret in serialised for secret in secrets):
                raise frozen.RouteBlocked("Credential reflection in provider output")
            (target / "response.json").write_text(serialised, encoding="utf-8")
            usage = response.get("usage")
            self.last_call.update(
                terminal_completed=True, status="completed", provider_model=response["model"],
                provider_reasoning={"effort": frozen.EFFORT},
                usage={name: usage[name] for name in ("input_tokens", "output_tokens", "total_tokens")
                       if isinstance(usage, dict) and type(usage.get(name)) is int and usage[name] >= 0})
            return raw
        except Exception as exc:
            code, stage = failure_details(exc, phase)
            self.last_call.update(blocked=True, timed_out=isinstance(exc, TimeoutError),
                                  failure_code=code, stage=stage)
            if isinstance(exc, frozen.ProviderBlocked):
                err = exc.provider_error
                self.last_call["provider_error"] = frozen._safe_error(
                    err, secrets, status=err.get("http_status"))
            # No arbitrary exception/provider text, tokens or rejected output.
            raise frozen.RouteBlocked("D5 transport stopped; no retry") from None
        finally:
            with http["lock"]:
                http["sealed"] = True
                self.last_call["http"] = dict(http["http"])
            with assembly["lock"]:
                assembly["sealed"] = True
                self.last_call["stream_assembly"] = dict(assembly["detail"])
            self.last_call["elapsed_seconds"] = time.monotonic() - started
            (target / "receipt.json").write_text(canonical_json(self.last_call), encoding="utf-8")
