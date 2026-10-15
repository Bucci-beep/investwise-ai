"""Narrow HTTP compatibility for SSE mislabelled as text/plain.

Only HTTP 200 with an explicit text/plain header gains a parsing opportunity.
Every byte still passes through the unchanged bounded SSE, completed-response
and passive-output validators. No prefix can become an accepted interpretation.
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.request

from ai import d5_chatgpt_plan_diagnostic as diagnostic
from ai import d5_chatgpt_plan_transport as frozen
from ai.llm_shadow import canonical_json


class SSECompatibleChatGPTPlanGenerator(diagnostic.DiagnosticChatGPTPlanGenerator):
    def __init__(self, trace_root, credential_provider, transport=None, *, opener_factory=None):
        self._opener_factory = opener_factory or urllib.request.build_opener
        self._observation = None
        super().__init__(trace_root, credential_provider, transport=transport or self._http_sse)

    def _http_sse(self, payload, token, deadline):
        observation = self._observation
        request = urllib.request.Request(frozen.ENDPOINT, data=canonical_json(payload).encode(),
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                     "Accept": "text/event-stream"}, method="POST")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        def observe(status, headers):
            try:
                content_type = headers.get_content_type()
                explicit = headers.get_all("Content-Type", [])
            except Exception:
                content_type, explicit = None, []
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
        observation = {"http": {}, "sealed": False, "lock": threading.Lock()}
        self._observation = observation
        try:
            return super().__call__(messages, schema)
        finally:
            with observation["lock"]:
                observation["sealed"] = True
                http = dict(observation["http"])
            if http:
                # Only fixed labels, booleans and a bounded status integer are recorded.
                self.last_call.setdefault("diagnostic", {})["http"] = http
                self.last_call["sse_compatibility_version"] = 1
                target = self.trace_root / self.case_id
                (target / "receipt.json").write_text(canonical_json(self.last_call), encoding="utf-8")
