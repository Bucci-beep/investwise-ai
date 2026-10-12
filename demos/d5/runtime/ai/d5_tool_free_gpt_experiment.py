"""Isolated D5 Responses transport; deliberately not wired into the application.

One text-only request, an empty tool list and no action dispatcher. The existing
D5 interpreter owns parsing and vetoes; the conversation engine owns actions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from ai.d5_puzzle_interpreter import D5_OPTION_IDS, output_schema
from ai.llm_shadow import canonical_json, strict_json

MODEL = "gpt-6.1-sol"
EFFORT = "low"
ENDPOINT = "https://api.openai.com/v1/responses"
MAX_OUTPUT_TOKENS = 4096
TIMEOUT_SECONDS = 30


class RouteBlocked(RuntimeError):
    """An access/configuration failure stops the experiment without substitution."""


def generation_schema(value):
    # As in the previous GPT experiment, uniqueness is checked by the parser.
    if isinstance(value, dict):
        return {k: generation_schema(v) for k, v in value.items() if k != "uniqueItems"}
    if isinstance(value, list):
        return [generation_schema(v) for v in value]
    return value


def request_payload(messages, schema):
    if canonical_json(schema) != canonical_json(output_schema(D5_OPTION_IDS)):
        raise ValueError("Only the existing passive D5 schema is allowed")
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(m, dict) for m in messages)
            or [m.get("role") for m in messages] != ["system", "user"]
            or any(set(m) != {"role", "content"} or not isinstance(m["content"], str)
                   for m in messages)):
        raise ValueError("Only D5 instructions and one current reply are allowed")
    data = strict_json(messages[1]["content"])
    if (not isinstance(data, dict)
            or set(data) != {"current_question", "allowed_options", "original_reply"}
            or not isinstance(data["current_question"], dict)
            or set(data["current_question"]) != {"id", "text"}
            or data["current_question"].get("id") != "D5"
            or not isinstance(data["current_question"]["text"], str)
            or not isinstance(data["allowed_options"], list)
            or len(data["allowed_options"]) != len(D5_OPTION_IDS)
            or any(not isinstance(o, dict) or set(o) != {"id", "label", "definition"}
                   or any(not isinstance(value, str) for value in o.values())
                   for o in data["allowed_options"])
            or {o["id"] for o in data["allowed_options"]} != D5_OPTION_IDS
            or not isinstance(data["original_reply"], str)
            or not data["original_reply"].strip()):
        raise ValueError("No conversation state or non-D5 input is allowed")
    return {"model": MODEL, "reasoning": {"effort": EFFORT},
            "input": copy.deepcopy(messages), "tools": [], "tool_choice": "none",
            "store": False, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "text": {"format": {"type": "json_schema", "name": "d5_interpretation",
                                "strict": True, "schema": generation_schema(schema)}}}


def passive_text(response):
    if not isinstance(response, dict) or response.get("model") != MODEL:
        raise RouteBlocked("Provider did not report the exact requested model")
    if response.get("status") != "completed" or response.get("error"):
        raise ValueError("Response did not complete successfully")
    if response.get("tools") != [] or response.get("tool_choice") != "none":
        raise RouteBlocked("Provider did not echo the tool-free request boundary")
    if response.get("reasoning", {}).get("effort") != EFFORT:
        raise RouteBlocked("Provider did not echo low reasoning effort")
    if response.get("store") is not False:
        raise RouteBlocked("Provider did not echo store=false")
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Missing response output")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Invalid output item")
        # Reasoning metadata is inert; no tool items or execution loop exist.
        if item.get("type") == "reasoning":
            continue
        if (item.get("type") != "message" or item.get("role") != "assistant"
                or item.get("status") != "completed"):
            raise ValueError("Non-message output is forbidden")
        if not isinstance(item.get("content"), list):
            raise ValueError("Invalid message content")
        for part in item["content"]:
            if (not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str)
                    or part.get("annotations", []) != []):
                raise ValueError("Refusal, annotation or non-text output is not an interpretation")
            texts.append(part["text"])
    if len(texts) != 1 or not texts[0].strip():
        raise ValueError("Exactly one passive JSON text is required")
    strict_json(texts[0])  # No Markdown stripping, repair or retry.
    return texts[0]


def _read_key(env_file):
    path = Path(env_file)
    if path.is_symlink() or not path.is_file():
        raise RouteBlocked("Confirmed local credential file is unavailable")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "OPENAI_API_KEY":
            matches.append(value.strip().strip("\"'"))
    if len(matches) != 1 or not matches[0]:
        raise RouteBlocked("Confirmed file must contain one API key")
    return matches[0]


def _http(payload, key):
    request = urllib.request.Request(ENDPOINT, data=canonical_json(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST")
    # No redirects: a credential cannot follow an unapproved endpoint.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=TIMEOUT_SECONDS) as reply:
        body = reply.read(1_048_577)
    if len(body) > 1_048_576:
        raise ValueError("Oversized provider response")
    return strict_json(body.decode("utf-8"))


class ResponsesGenerator:
    """Transport only. It receives no engine, save store, tool or action handle."""
    def __init__(self, trace_root, *, env_file, transport=None):
        self.trace_root, self.env_file = Path(trace_root), Path(env_file)
        self.transport = transport or _http
        self.calls, self.case_id, self.last_call = 0, None, {}

    def __call__(self, messages, schema):
        payload = request_payload(messages, schema)
        if not self.case_id or not re.fullmatch(r"[A-Za-z0-9_-]+", self.case_id):
            raise ValueError("A unique safe case ID is required")
        target = self.trace_root / self.case_id
        target.mkdir(parents=True, exist_ok=False)
        (target / "request.json").write_text(canonical_json(payload), encoding="utf-8")
        self.last_call = {"backend": "public_responses_tool_free", "endpoint": ENDPOINT,
            "model_requested": MODEL, "reasoning_effort_requested": EFFORT,
            "tools": [], "tool_choice": "none", "store": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "timeout_seconds": TIMEOUT_SECONDS,
            "retry": False, "output_repair": False, "fresh_context": True,
            "blocked": False, "timed_out": False, "actual_calls": 0,
            "tool_dispatcher_present": False, "trace_directory": str(target)}
        started = time.monotonic()
        try:
            key = _read_key(self.env_file)
            self.calls += 1
            self.last_call["actual_calls"] = 1
            response = self.transport(payload, key)
            if not isinstance(response, dict):
                raise ValueError("Provider response must be an object")
            serialised = canonical_json(response)
            if key in serialised:
                raise RouteBlocked("Provider response contained credential material; nothing was logged")
            self.last_call.update(provider_model=response.get("model"),
                provider_reasoning=response.get("reasoning"), status=response.get("status"),
                usage=response.get("usage"), output_types=[o.get("type") if isinstance(o, dict) else "invalid"
                    for o in response.get("output", [])] if isinstance(response.get("output"), list) else [])
            (target / "response.json").write_text(serialised, encoding="utf-8")
            raw = passive_text(response)
            if time.monotonic() - started > TIMEOUT_SECONDS:
                raise TimeoutError("Whole-request deadline exceeded")
            return raw
        except urllib.error.HTTPError as exc:
            # Never persist headers, credential values or full error messages.
            safe = {"http_status": exc.code}
            try:
                error = json.loads(exc.read(65536)).get("error", {})
                for name in ("code", "type", "param"):
                    value = error.get(name)
                    if value is None or (isinstance(value, str) and key not in value
                            and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value)):
                        safe[name] = value
            except (ValueError, AttributeError):
                pass
            self.last_call.update(blocked=True, provider_error=safe)
            raise RouteBlocked("Public API route rejected: " + canonical_json(safe)) from None
        except RouteBlocked:
            self.last_call["blocked"] = True
            raise
        except TimeoutError:
            self.last_call["timed_out"] = True
            raise
        finally:
            self.last_call["elapsed_seconds"] = time.monotonic() - started
            (target / "receipt.json").write_text(canonical_json(self.last_call), encoding="utf-8")
