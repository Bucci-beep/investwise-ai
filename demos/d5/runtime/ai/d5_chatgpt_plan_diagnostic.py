"""Observational diagnostics for the unchanged, frozen ChatGPT-plan transport.

Request construction, credentials, SSE validation and passive-output validation
are delegated to the existing implementation. Failed responses persist only
safe summaries, never raw text, reasoning or arbitrary exception messages.
Validated success responses retain the original experiment artifact unchanged.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from ai import d5_chatgpt_plan_transport as frozen
from ai.llm_shadow import canonical_json

MAX_EVENT_TYPES = 16
MAX_SUMMARY_EVENTS = 512
MAX_TOOL_TYPES = 8
MAX_SUMMARY_BYTES = 8192

# Only exact, fixed messages from local guard code may name a failure.
GUARDS = {
    "Expected an SSE response": ("unexpected_content_type", "http"),
    "Whole-request deadline exceeded": ("deadline_exceeded", "deadline"),
    "Invalid stream byte chunk": ("invalid_stream_chunk", "sse"),
    "Response exceeded local byte limit": ("response_byte_limit", "sse"),
    "Malformed SSE field": ("malformed_sse_field", "sse"),
    "Duplicate SSE event field": ("duplicate_sse_event_field", "sse"),
    "Malformed or conflicting SSE event": ("conflicting_sse_event", "sse"),
    "Stream ended in an unfinished SSE frame": ("unfinished_sse_frame", "sse"),
    "Tool or unsupported output item forbidden": ("unsupported_output_item", "output"),
    "Invalid assistant output item": ("invalid_assistant_item", "output"),
    "Refusal or non-text content forbidden": ("refusal_or_nontext", "output"),
    "Unexpected or repeated stream terminator": ("unexpected_stream_terminator", "completion"),
    "Event after terminal response": ("event_after_terminal", "completion"),
    "Provider rejected ChatGPT-plan inference": ("provider_rejected", "provider"),
    "Credential reflection in provider output": ("credential_reflection", "output"),
    "Invalid response metadata": ("invalid_response_metadata", "provider_boundary"),
    "Conflicting provider boundary metadata": ("provider_boundary_mismatch", "provider_boundary"),
    "Conflicting provider reasoning effort": ("reasoning_effort_mismatch", "provider_boundary"),
    "Conflicting response identity": ("response_identity_mismatch", "provider_boundary"),
    "Stream text disagrees with completed response": ("stream_text_mismatch", "completion"),
    "Finished text disagrees with completed response": ("finished_text_mismatch", "completion"),
    "Invalid streamed output text": ("invalid_stream_text", "output"),
    "Repeated or annotated streamed text": ("repeated_or_annotated_text", "output"),
    "Conflicting text completion": ("text_completion_mismatch", "completion"),
    "Unexpected streaming output type": ("unexpected_event_type", "sse"),
    "Stream ended without completed inference": ("missing_completed_response", "completion"),
    "Invalid credential provider result": ("invalid_credential_result", "credentials"),
    "Credential reflection in request": ("request_credential_reflection", "request"),
    "Provider did not report the exact requested model": ("model_mismatch", "provider_boundary"),
    "Response did not complete successfully": ("unsuccessful_response", "completion"),
    "Provider did not echo the tool-free request boundary": ("tools_boundary_mismatch", "provider_boundary"),
    "Provider did not echo low reasoning effort": ("reasoning_effort_mismatch", "provider_boundary"),
    "Provider did not echo store=false": ("store_mismatch", "provider_boundary"),
    "Missing response output": ("missing_response_output", "output"),
    "Invalid output item": ("invalid_output_item", "output"),
    "Non-message output is forbidden": ("non_message_output", "output"),
    "Invalid message content": ("invalid_message_content", "output"),
    "Refusal, annotation or non-text output is not an interpretation": ("nonpassive_content", "output"),
    "Exactly one passive JSON text is required": ("passive_text_count", "output"),
    "invalid_json": ("invalid_json", "parsing"),
    "duplicate_json_key": ("duplicate_json_key", "parsing"),
    "nonfinite_json_number": ("nonfinite_json_number", "parsing"),
    "output_must_be_text": ("output_must_be_text", "parsing"),
}


def _value(value, secrets):
    if value is None or type(value) is bool:
        return value
    if type(value) is int and abs(value) <= 1_000_000:
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,100}", value):
        if not any(secret and secret in value for secret in secrets):
            return value
    return "redacted_or_invalid"


def _field(data, name, secrets):
    return {"present": name in data, "actual": _value(data.get(name), secrets)}


def _metadata(response, secrets):
    if not isinstance(response, dict):
        return {"invalid_response_object": True}
    result = {name: _field(response, name, secrets) for name in ("model", "tool_choice", "store")}
    reasoning = response.get("reasoning")
    result["reasoning.effort"] = _field(reasoning if isinstance(reasoning, dict) else {}, "effort", secrets)
    tools = response.get("tools")
    result["tools"] = {"present": "tools" in response, "is_list": isinstance(tools, list)}
    if isinstance(tools, list):
        result["tools"].update(count=len(tools), types=[
            _value(item.get("type") if isinstance(item, dict) else None, secrets) for item in tools[:MAX_TOOL_TYPES]])
    return result


def _mismatches(response, code):
    """Describe the same first metadata guard; never relax its decision."""
    if not isinstance(response, dict):
        return []
    boundary = (("model", frozen.MODEL), ("tools", []), ("tool_choice", "none"), ("store", False))
    if code == "provider_boundary_mismatch":
        return [name for name, expected in boundary if name in response and response[name] != expected]
    if code == "reasoning_effort_mismatch":
        reasoning = response.get("reasoning")
        return ["reasoning.effort"] if not isinstance(reasoning, dict) or reasoning.get("effort") != frozen.EFFORT else []
    if code == "model_mismatch":
        return ["model"] if response.get("model") != frozen.MODEL else []
    if code == "store_mismatch":
        return ["store"] if response.get("store") is not False else []
    if code == "tools_boundary_mismatch":
        return [name for name, expected in boundary[1:3] if response.get(name) != expected]
    return []


def failure_details(exc, phase):
    if isinstance(exc, TimeoutError):
        return "deadline_exceeded", phase
    if isinstance(exc, UnicodeError):
        return "invalid_utf8", "sse"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json", "parsing"
    if isinstance(exc, OSError) and not isinstance(exc, frozen.RouteBlocked):
        return "transport_io_error", phase
    # The text is used solely as an exact lookup, never retained or interpolated.
    return GUARDS.get(str(exc), ("unclassified_local_rejection", phase))


def _summary(chunks, code, secrets, deadline):
    summary = {"event_types": [], "events_inspected": 0}
    if deadline <= time.monotonic():
        summary["inspection_deadline_expired"] = True
        return summary
    try:
        for event in frozen._events(iter(chunks), deadline):
            if summary["events_inspected"] >= MAX_SUMMARY_EVENTS:
                summary["inspection_limit_reached"] = True
                break
            summary["events_inspected"] += 1
            kind = "stream_terminator" if event is None else _value(event.get("type"), secrets)
            if kind not in summary["event_types"] and len(summary["event_types"]) < MAX_EVENT_TYPES:
                summary["event_types"].append(kind)
            if event is None:
                continue
            if "response" in event:
                summary["last_metadata"] = _metadata(event["response"], secrets)
                summary["last_metadata_event"] = kind
                mismatch = _mismatches(event["response"], code)
                if mismatch:
                    summary["mismatch_fields"] = mismatch
                    summary["mismatch_event"] = kind
                    break
            item = event.get("item")
            if isinstance(item, dict):
                summary["last_output_item_type"] = _value(item.get("type"), secrets)
    except Exception:
        summary["inspection_stopped_at_invalid_frame"] = True
    return summary


def _http_tap(payload, token, deadline, observation, secrets):
    """The frozen HTTP recipe plus safe status/content-type observations."""
    request = urllib.request.Request(frozen.ENDPOINT, data=canonical_json(payload).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                 "Accept": "text/event-stream"}, method="POST")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=frozen._remaining(deadline)) as reply:
            try:
                observation["http_status"] = _value(reply.getcode(), secrets)
            except Exception:
                pass  # Missing observational data cannot change acceptance.
            content_type = reply.headers.get_content_type()
            observation["content_type"] = _value(content_type, secrets)
            if content_type != "text/event-stream":
                raise frozen.RouteBlocked("Expected an SSE response")
            read_chunk = getattr(reply, "read1", reply.read)
            while True:
                frozen._remaining(deadline)
                chunk = read_chunk(8192)
                if not chunk:
                    return
                yield chunk
    except urllib.error.HTTPError as exc:
        observation["http_status"] = _value(exc.code, secrets)
        try:
            if exc.headers is not None:
                observation["content_type"] = _value(exc.headers.get_content_type(), secrets)
        except Exception:
            pass
        raise


class DiagnosticChatGPTPlanGenerator(frozen.ChatGPTPlanGenerator):
    def __init__(self, trace_root, credential_provider, transport=None):
        super().__init__(trace_root, credential_provider, transport=transport)
        self._injected_transport = transport

    def __call__(self, messages, schema):
        payload = frozen.request_payload(messages, schema)
        if not self.case_id or not re.fullmatch(r"[A-Za-z0-9_-]+", self.case_id):
            raise ValueError("A unique safe case ID is required")
        target = self.trace_root / self.case_id
        target.mkdir(parents=True, exist_ok=False)
        self.last_call = {"backend": "chatgpt_plan_public_responses_stream", "endpoint": frozen.ENDPOINT,
            "model_requested": frozen.MODEL, "reasoning_effort_requested": frozen.EFFORT,
            "tools": [], "tool_choice": "none", "store": False, "stream": True,
            "timeout_seconds": frozen.TIMEOUT_SECONDS, "max_response_bytes": frozen.MAX_RESPONSE_BYTES,
            "provider_output_token_cap": None, "retry": False, "output_repair": False,
            "fresh_context": True, "blocked": False, "timed_out": False,
            "actual_calls": 0, "tool_dispatcher_present": False, "diagnostic_version": 1}
        started = time.monotonic()
        deadline = started + frozen.TIMEOUT_SECONDS
        phase, secrets, observed, http = "credentials", (), [], {}
        buffered = 0
        try:
            token, secrets = frozen._credentials(self.credential_provider, deadline)
            if not isinstance(token, str) or not token or any(not isinstance(s, str) for s in secrets):
                secrets = tuple(s for s in secrets if isinstance(s, str))
                raise frozen.RouteBlocked("Invalid credential provider result")
            phase = "request"
            encoded = canonical_json(payload)
            if any(secret and secret in encoded for secret in secrets):
                raise frozen.RouteBlocked("Credential reflection in request")
            frozen._remaining(deadline)
            (target / "request.json").write_text(encoded, encoding="utf-8")
            self.calls += 1
            self.last_call["actual_calls"] = 1
            send = self._injected_transport or (lambda p, t, d: _http_tap(p, t, d, http, secrets))
            chunks = frozen._bounded_chunks(send, payload, token, deadline, secrets)

            def tap():
                nonlocal buffered
                for chunk in chunks:
                    if isinstance(chunk, bytes) and buffered < frozen.MAX_RESPONSE_BYTES:
                        part = chunk[:frozen.MAX_RESPONSE_BYTES - buffered]
                        observed.append(part); buffered += len(part)
                    yield chunk  # Validation always receives the original bytes.

            phase = "stream"
            try:
                response = frozen.completed_response(tap(), deadline, secrets)
            finally:
                chunks.close()
            phase = "passive_output"
            raw = frozen.passive_text(response)
            frozen._remaining(deadline)
            (target / "response.json").write_text(canonical_json(response), encoding="utf-8")
            usage = response.get("usage")
            self.last_call.update(provider_model=response["model"], provider_reasoning={"effort": frozen.EFFORT},
                status="completed", terminal_completed=True, usage={name: usage[name]
                    for name in ("input_tokens", "output_tokens", "total_tokens")
                    if isinstance(usage, dict) and type(usage.get(name)) is int and usage[name] >= 0})
            self.last_call["diagnostic"] = {"http": dict(http), "terminal_metadata": _metadata(response, secrets)}
            return raw
        except Exception as exc:
            code, stage = failure_details(exc, phase)
            self.last_call.update(blocked=True, timed_out=isinstance(exc, TimeoutError),
                failure_code=code, stage=stage)
            if isinstance(exc, frozen.ProviderBlocked):
                safe = exc.provider_error
                self.last_call["provider_error"] = frozen._safe_error(safe, secrets, status=safe.get("http_status"))
            diagnostic = _summary(observed, code, secrets, deadline)
            diagnostic["http"] = dict(http)
            self.last_call["diagnostic"] = diagnostic
            raise frozen.RouteBlocked("ChatGPT-plan diagnostic blocked: " + code + "; no retry or partial answer") from None
        finally:
            self.last_call["elapsed_seconds"] = time.monotonic() - started
            encoded = canonical_json(self.last_call)
            if len(encoded.encode()) > MAX_SUMMARY_BYTES or any(secret and secret in encoded for secret in secrets):
                # Retain a fixed local failure reason, never unsafe provider metadata.
                self.last_call.pop("diagnostic", None)
                self.last_call.pop("provider_error", None)
                self.last_call.pop("usage", None)
                self.last_call["diagnostic_summary_suppressed"] = True
                encoded = canonical_json(self.last_call)
            if any(secret and secret in encoded for secret in secrets):
                encoded = canonical_json({"blocked": True, "failure_code": "unsafe_diagnostic_suppressed",
                                          "stage": "receipt", "actual_calls": self.last_call["actual_calls"]})
                self.last_call = json.loads(encoded)
            (target / "receipt.json").write_text(encoded, encoding="utf-8")
