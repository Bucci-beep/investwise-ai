"""Isolated, passive D5 inference through the official ChatGPT-plan route.

OAuth supplies access to public Responses; it supplies no tools or engine
authority. Streamed text is private until completed inference and stream EOF.
There is no provider output-token cap on this route: local limits are 30 seconds
for the whole operation and 1 MiB of response bytes, with no inference retry.
"""
from __future__ import annotations

import codecs
from pathlib import Path
import queue
import re
import threading
import time
import urllib.error
import urllib.request

from ai.d5_tool_free_gpt_experiment import (
    EFFORT, ENDPOINT, MODEL, RouteBlocked, TIMEOUT_SECONDS,
    passive_text, request_payload as frozen_request_payload,
)
from ai.llm_shadow import canonical_json, strict_json

MAX_RESPONSE_BYTES = 1_048_576


class ProviderBlocked(RouteBlocked):
    """Safe provider error fields, never provider messages or bodies."""
    def __init__(self, provider_error):
        super().__init__("Provider rejected ChatGPT-plan inference")
        self.provider_error = provider_error


def _safe_error(error, secrets, *, status=None):
    safe = {"http_status": status} if type(status) is int else {}
    if isinstance(error, dict):
        for name in ("code", "type", "param"):
            value = error.get(name)
            if name in error and (value is None or (isinstance(value, str)
                    and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value)
                    and not any(secret and secret in value for secret in secrets))):
                safe[name] = value
    return safe


def request_payload(messages, schema):
    """Preserve the frozen interpretation, adapting only route requirements."""
    payload = frozen_request_payload(messages, schema)
    instructions, current = payload.pop("input")
    payload["instructions"] = instructions["content"]
    payload["input"] = [current]
    payload["stream"] = True
    payload.pop("max_output_tokens")  # Unsupported by ChatGPT-plan inference.
    return payload


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Whole-request deadline exceeded")
    return remaining


def _http(payload, token, deadline):
    """Yield bytes only; never follow a redirect or retry the POST."""
    request = urllib.request.Request(ENDPOINT, data=canonical_json(payload).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                 "Accept": "text/event-stream"}, method="POST")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=_remaining(deadline)) as reply:
        if reply.headers.get_content_type() != "text/event-stream":
            raise RouteBlocked("Expected an SSE response")
        read_chunk = getattr(reply, "read1", reply.read)
        while True:
            _remaining(deadline)
            chunk = read_chunk(8192)
            if not chunk:
                return
            yield chunk


def _bounded_chunks(transport, payload, token, deadline, secrets):
    """The caller's deadline also bounds a stalled socket/iterator.

    The reader owns no trace/state handles. On cancellation it closes its stream
    when the current read returns; the HTTP read itself has a finite timeout.
    """
    messages = queue.Queue(maxsize=1)
    cancelled = threading.Event()

    def publish(value):
        while not cancelled.is_set():
            try:
                messages.put(value, timeout=min(0.05, _remaining(deadline)))
                return
            except queue.Full:
                pass

    def reader():
        source = None
        try:
            _remaining(deadline)
            source = iter(transport(payload, token, deadline))
            while not cancelled.is_set():
                try:
                    chunk = next(source)
                except StopIteration:
                    publish(("end", None))
                    break
                publish(("chunk", chunk))
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                safe = _safe_error({}, secrets, status=exc.code)
                try:
                    _remaining(deadline)
                    body = exc.read(65_537)
                    if len(body) <= 65_536:
                        parsed = strict_json(body.decode("utf-8"))
                        if isinstance(parsed, dict):
                            safe = _safe_error(parsed.get("error"), secrets, status=exc.code)
                except Exception:
                    pass
                finally:
                    try:
                        exc.close()
                    except Exception:
                        pass
                exc = ProviderBlocked(safe)
            try:
                publish(("error", exc))
            except TimeoutError:
                pass
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass  # Never print a transport/credential exception in a worker.

    threading.Thread(target=reader, daemon=True, name="d5-passive-sse").start()
    try:
        while True:
            try:
                kind, value = messages.get(timeout=_remaining(deadline))
            except queue.Empty:
                raise TimeoutError("Whole-request deadline exceeded") from None
            _remaining(deadline)
            if kind == "error":
                raise value
            if kind == "end":
                return
            yield value
    finally:
        cancelled.set()


def _credentials(provider, deadline):
    """Bound token access too; a delayed provider can never start inference."""
    result = queue.Queue(maxsize=1)

    def read():
        try:
            token = provider.get_access_token()
            result.put((True, (token, tuple(provider.sensitive_values()) + (token,))))
        except Exception as exc:
            result.put((False, exc))

    threading.Thread(target=read, daemon=True, name="d5-credential-read").start()
    try:
        ok, value = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise TimeoutError("Whole-request deadline exceeded") from None
    _remaining(deadline)
    if not ok:
        raise value
    return value


def _events(chunks, deadline):
    """Parse bounded SSE, including CRLF, fragmented Unicode and data lines."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending, fields, size = "", [], 0
    for chunk in chunks:
        _remaining(deadline)
        if not isinstance(chunk, bytes) or not chunk:
            raise RouteBlocked("Invalid stream byte chunk")
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise RouteBlocked("Response exceeded local byte limit")
        pending += decoder.decode(chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line = line.removesuffix("\r")
            if line:
                if line.startswith(":"):
                    continue
                name, sep, value = line.partition(":")
                if not sep or name not in {"event", "data", "id", "retry"}:
                    raise RouteBlocked("Malformed SSE field")
                fields.append((name, value.removeprefix(" ")))
                continue
            data = "\n".join(value for name, value in fields if name == "data")
            names = [value for name, value in fields if name == "event"]
            fields = []
            if not data:
                continue
            if len(names) > 1:
                raise RouteBlocked("Duplicate SSE event field")
            if data == "[DONE]":
                yield None
                continue
            event = strict_json(data)
            if (not isinstance(event, dict) or not isinstance(event.get("type"), str)
                    or (names and names[0] != event["type"])):
                raise RouteBlocked("Malformed or conflicting SSE event")
            yield event
    pending += decoder.decode(b"", final=True)
    if pending or fields:
        raise RouteBlocked("Stream ended in an unfinished SSE frame")


def _check_item(item):
    if not isinstance(item, dict) or item.get("type") not in {"reasoning", "message"}:
        raise RouteBlocked("Tool or unsupported output item forbidden")
    if item["type"] == "message":
        if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            raise RouteBlocked("Invalid assistant output item")
        for part in item["content"]:
            _check_part(part)


def _check_part(part):
    if (not isinstance(part, dict) or part.get("type") != "output_text"
            or part.get("annotations", []) != []):
        raise RouteBlocked("Refusal or non-text content forbidden")


def completed_response(chunks, deadline, sensitive_values):
    terminal, done, response_id = None, False, None
    deltas = []
    seen_delta = False
    finished_text = None
    allowed_metadata = {"response.created", "response.in_progress",
        "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
        "response.reasoning_text.delta", "response.reasoning_text.done"}
    for event in _events(chunks, deadline):
        if event is None:
            if terminal is None or done:
                raise RouteBlocked("Unexpected or repeated stream terminator")
            done = True
            continue
        if terminal is not None or done:
            raise RouteBlocked("Event after terminal response")
        kind = event["type"]
        if kind in {"response.failed", "response.incomplete", "error"}:
            response = event.get("response")
            error = response.get("error") if isinstance(response, dict) else event.get("error", event)
            raise ProviderBlocked(_safe_error(error, sensitive_values))
        serialised = canonical_json(event)
        if any(secret and secret in serialised for secret in sensitive_values):
            raise RouteBlocked("Credential reflection in provider output")
        response = event.get("response")
        if response is not None:
            if not isinstance(response, dict):
                raise RouteBlocked("Invalid response metadata")
            for name, expected in (("model", MODEL), ("tools", []), ("tool_choice", "none"), ("store", False)):
                if name in response and response[name] != expected:
                    raise RouteBlocked("Conflicting provider boundary metadata")
            if "reasoning" in response and (not isinstance(response["reasoning"], dict)
                    or response["reasoning"].get("effort") != EFFORT):
                raise RouteBlocked("Conflicting provider reasoning effort")
            ident = response.get("id")
            if ident is not None:
                if not isinstance(ident, str) or (response_id is not None and ident != response_id):
                    raise RouteBlocked("Conflicting response identity")
                response_id = ident
            for item in response.get("output", []):
                _check_item(item)
        if kind == "response.completed":
            raw = passive_text(response)
            if seen_delta and "".join(deltas) != raw:
                raise RouteBlocked("Stream text disagrees with completed response")
            if finished_text is not None and finished_text != raw:
                raise RouteBlocked("Finished text disagrees with completed response")
            terminal = response
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            _check_item(event.get("item"))
        elif kind in {"response.content_part.added", "response.content_part.done"}:
            _check_part(event.get("part"))
        elif kind in {"response.output_text.delta", "response.output_text.done"}:
            field = "delta" if kind.endswith(".delta") else "text"
            if not isinstance(event.get(field), str):
                raise RouteBlocked("Invalid streamed output text")
            if finished_text is not None or event.get("annotations", []) != []:
                raise RouteBlocked("Repeated or annotated streamed text")
            if field == "delta":
                deltas.append(event[field]); seen_delta = True
            else:
                if seen_delta and event["text"] != "".join(deltas):
                    raise RouteBlocked("Conflicting text completion")
                finished_text = event["text"]
        elif kind not in allowed_metadata:
            raise RouteBlocked("Unexpected streaming output type")
    _remaining(deadline)
    if terminal is None:
        raise RouteBlocked("Stream ended without completed inference")
    return terminal


class ChatGPTPlanGenerator:
    """Credentials and transport only; no engine, tools, actions or save store."""
    def __init__(self, trace_root, credential_provider, *, transport=None):
        self.trace_root = Path(trace_root)
        self.credential_provider = credential_provider
        self.transport = transport or _http
        self.calls, self.case_id, self.last_call = 0, None, {}

    def __call__(self, messages, schema):
        payload = request_payload(messages, schema)
        if not self.case_id or not re.fullmatch(r"[A-Za-z0-9_-]+", self.case_id):
            raise ValueError("A unique safe case ID is required")
        target = self.trace_root / self.case_id
        target.mkdir(parents=True, exist_ok=False)
        self.last_call = {"backend": "chatgpt_plan_public_responses_stream", "endpoint": ENDPOINT,
            "model_requested": MODEL, "reasoning_effort_requested": EFFORT,
            "tools": [], "tool_choice": "none", "store": False, "stream": True,
            "timeout_seconds": TIMEOUT_SECONDS, "max_response_bytes": MAX_RESPONSE_BYTES,
            "provider_output_token_cap": None, "retry": False, "output_repair": False,
            "fresh_context": True, "blocked": False, "timed_out": False,
            "actual_calls": 0, "tool_dispatcher_present": False}
        started = time.monotonic()
        deadline = started + TIMEOUT_SECONDS
        try:
            token, secrets = _credentials(self.credential_provider, deadline)
            if not isinstance(token, str) or not token or any(not isinstance(s, str) for s in secrets):
                raise RouteBlocked("Invalid credential provider result")
            encoded = canonical_json(payload)
            if any(secret and secret in encoded for secret in secrets):
                raise RouteBlocked("Credential reflection in request")
            _remaining(deadline)
            (target / "request.json").write_text(encoded, encoding="utf-8")
            self.calls += 1
            self.last_call["actual_calls"] = 1
            chunks = _bounded_chunks(self.transport, payload, token, deadline, secrets)
            try:
                response = completed_response(chunks, deadline, secrets)
            finally:
                chunks.close()
            raw = passive_text(response)
            _remaining(deadline)
            (target / "response.json").write_text(canonical_json(response), encoding="utf-8")
            self.last_call.update(provider_model=response["model"],
                provider_reasoning=response["reasoning"], status="completed",
                terminal_completed=True, usage=response.get("usage"))
            return raw
        except Exception as exc:
            self.last_call["blocked"] = True
            self.last_call["timed_out"] = isinstance(exc, TimeoutError)
            # Do not log provider bodies, headers, token values or exception text.
            self.last_call["error_type"] = type(exc).__name__
            if isinstance(exc, ProviderBlocked):
                self.last_call["provider_error"] = exc.provider_error
            raise RouteBlocked("ChatGPT-plan transport blocked; no retry or partial answer") from None
        finally:
            self.last_call["elapsed_seconds"] = time.monotonic() - started
            (target / "receipt.json").write_text(canonical_json(self.last_call), encoding="utf-8")
