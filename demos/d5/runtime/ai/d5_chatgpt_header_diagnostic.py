"""Safe header diagnostics; frozen SSE compatibility acceptance is unchanged."""
from __future__ import annotations

import urllib.error
import urllib.request

from ai import d5_chatgpt_sse_compat as compat
from ai.llm_shadow import canonical_json

frozen = compat.frozen
MEDIA_TYPES = frozenset({"text/plain", "text/event-stream", "application/json", "text/html", "application/octet-stream"})


def value_shape(value):
    if value is None:
        return "none"
    return {str: "string", list: "list", tuple: "tuple", dict: "dict", int: "integer", bool: "boolean"}.get(type(value), "other")


def error_kind(exc):
    return {AttributeError: "AttributeError", TypeError: "TypeError", ValueError: "ValueError",
            KeyError: "KeyError", RuntimeError: "RuntimeError", OSError: "OSError",
            TimeoutError: "TimeoutError"}.get(type(exc), "other_exception")


def media_label(value):
    if type(value) is not str:
        return "non_text"
    token = value.split(";", 1)[0].strip().lower()
    return token if token in MEDIA_TYPES else "other"


def read_header_diagnostics(headers):
    """Same ordered accessor calls and fallback values as frozen observe().

    No fallback accessor, raw header, arbitrary class/exception name or message
    is persisted. Only a successful get_all list/tuple can establish absence.
    """
    detail = {"version": 1, "header_state": "unreadable", "headers_object": value_shape(headers),
              "get_content_type": {"state": "not_reached"}, "get_all": {"state": "not_reached"}}
    stage = "get_content_type.lookup"
    try:
        getter = headers.get_content_type
        stage = "get_content_type.call"
        content_type = getter()
        detail["get_content_type"] = {"state": "returned", "return_shape": value_shape(content_type),
                                      "media_type": media_label(content_type)}
        stage = "get_all.lookup"
        getter = headers.get_all
        stage = "get_all.call"
        explicit = getter("Content-Type", [])
        detail["get_all"] = {"state": "returned", "return_shape": value_shape(explicit)}
    except Exception as exc:
        accessor, phase = stage.split(".")
        detail[accessor] = {"state": phase + "_failed", "exception_type": error_kind(exc)}
        detail["failed_stage"] = stage
        return None, [], detail  # Identical acceptance fallback; no diagnostic recovery.
    if type(explicit) in (list, tuple):
        count = len(explicit)
        detail.update(header_state="missing" if count == 0 else "single" if count == 1 else "duplicate",
                      header_count=min(count, 16), header_count_capped=count > 16,
                      media_types=[media_label(item) for item in explicit[:4]])
        if any(type(item) is not str for item in explicit[:4]):
            detail["header_state"] = "unreadable"
            detail["failed_stage"] = "get_all.return_value"
    else:
        detail["failed_stage"] = "get_all.return_shape"
    return content_type, explicit, detail


class HeaderDiagnosticChatGPTPlanGenerator(compat.SSECompatibleChatGPTPlanGenerator):
    def _http_sse(self, payload, token, deadline):
        observation = self._observation
        request = urllib.request.Request(frozen.ENDPOINT, data=canonical_json(payload).encode(),
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                     "Accept": "text/event-stream"}, method="POST")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        def observe(status, headers):
            content_type, explicit, detail = read_header_diagnostics(headers)
            with observation["lock"]:
                if not observation["sealed"]:
                    observation["header_diagnostic"] = detail
            # MIME libraries default missing/malformed headers to text/plain.
            # The compatibility exception requires the literal media-type token.
            media_type = (explicit[0].split(";", 1)[0].strip().lower()
                          if len(explicit) == 1 and isinstance(explicit[0], str) else None)
            known_type = media_type if media_type in {"text/event-stream", "text/plain"} else "other"
            result = {"http_status": status if type(status) is int and 100 <= status <= 599 else None,
                      "content_type": known_type, "single_explicit_content_type": len(explicit) == 1,
                      "mislabelled_sse_candidate": status == 200 and media_type == "text/plain"}
            with observation["lock"]:
                if not observation["sealed"]:
                    observation["http"] = result
            return content_type, result

        opener = self._opener_factory(NoRedirect)
        try:
            with opener.open(request, timeout=frozen._remaining(deadline)) as reply:
                content_type, observed = observe(reply.getcode(), reply.headers)
                if content_type != "text/event-stream" and not observed["mislabelled_sse_candidate"]:
                    raise frozen.RouteBlocked("Expected an SSE response")
                read_chunk = getattr(reply, "read1", reply.read)
                while True:
                    frozen._remaining(deadline)
                    chunk = read_chunk(8192)
                    if not chunk:
                        return
                    yield chunk  # Unchanged bytes; the inherited validator owns acceptance.
        except urllib.error.HTTPError as exc:
            observe(exc.code, exc.headers)
            raise


    def __call__(self, messages, schema):
        try:
            return super().__call__(messages, schema)
        finally:
            observation = self._observation
            if observation is not None:
                with observation["lock"]:
                    detail = observation.get("header_diagnostic")
                if detail is not None:
                    self.last_call["header_diagnostic"] = detail
                    target = self.trace_root / self.case_id
                    (target / "receipt.json").write_text(canonical_json(self.last_call), encoding="utf-8")
