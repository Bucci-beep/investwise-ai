"""Allow positively established missing MIME to reach the frozen SSE validator."""
from __future__ import annotations

import urllib.error
import urllib.request

from ai.d5_chatgpt_header_diagnostic import (
    HeaderDiagnosticChatGPTPlanGenerator, frozen, read_header_diagnostics,
)
from ai.llm_shadow import canonical_json


class MissingMimeChatGPTPlanGenerator(HeaderDiagnosticChatGPTPlanGenerator):
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
            media_type = (explicit[0].split(";", 1)[0].strip().lower()
                          if len(explicit) == 1 and isinstance(explicit[0], str) else None)
            known_type = media_type if media_type in {"text/event-stream", "text/plain"} else "other"
            # The MIME reader defaults an absent header to text/plain. An
            # unreadable header is not absence. This only admits bytes to the
            # existing parser; a complete validated SSE response is still required.
            missing = (type(status) is int and status == 200 and content_type == "text/plain"
                       and detail["header_state"] == "missing"
                       and detail.get("header_count") == 0
                       and detail["get_content_type"].get("state") == "returned"
                       and detail["get_content_type"].get("return_shape") == "string"
                       and detail["get_all"].get("state") == "returned"
                       and detail["get_all"].get("return_shape") in {"list", "tuple"})
            result = {"http_status": status if type(status) is int and 100 <= status <= 599 else None,
                      "content_type": known_type, "single_explicit_content_type": len(explicit) == 1,
                      "mislabelled_sse_candidate": status == 200 and media_type == "text/plain",
                      "missing_content_type_candidate": missing}
            with observation["lock"]:
                if not observation["sealed"]:
                    observation["http"] = result
            return content_type, result

        opener = self._opener_factory(NoRedirect)
        try:
            with opener.open(request, timeout=frozen._remaining(deadline)) as reply:
                content_type, observed = observe(reply.getcode(), reply.headers)
                if (content_type != "text/event-stream"
                        and not observed["mislabelled_sse_candidate"]
                        and not observed["missing_content_type_candidate"]):
                    raise frozen.RouteBlocked("Expected an SSE response")
                read_chunk = getattr(reply, "read1", reply.read)
                while True:
                    frozen._remaining(deadline)
                    chunk = read_chunk(8192)
                    if not chunk:
                        return
                    yield chunk
        except urllib.error.HTTPError as exc:
            observe(exc.code, exc.headers)
            raise
