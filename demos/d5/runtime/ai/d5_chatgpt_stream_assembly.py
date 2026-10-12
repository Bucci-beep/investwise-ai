"""Assemble an empty terminal output from corroborated completed stream items."""
from __future__ import annotations

import re
import threading

from ai.d5_chatgpt_missing_mime import MissingMimeChatGPTPlanGenerator, frozen
from ai.llm_shadow import canonical_json

MAX_ITEMS = 16


class AssemblyBlocked(frozen.RouteBlocked):
    def __init__(self, code):
        self.code = code
        super().__init__("Stream assembly rejected")


def require(condition, code):
    if not condition:
        raise AssemblyBlocked(code)


def assembled_response(events, terminal):
    """Use only original done items; never manufacture or repair their text."""
    require(isinstance(terminal, dict) and terminal.get("output") == [], "not_explicit_empty_output")
    require(terminal.get("status") == "completed" and not terminal.get("error"), "terminal_not_completed")
    rid = terminal.get("id")
    require(isinstance(rid, str) and bool(rid), "missing_response_identity")
    sequences = [e.get("sequence_number") for e in events]
    require(all(type(n) is int and n >= 0 for n in sequences)
            and all(a < b for a, b in zip(sequences, sequences[1:])), "invalid_event_sequence")
    created = [e for e in events if e["type"] == "response.created"]
    require(len(created) == 1, "missing_or_duplicate_response_created")
    require(events[0]["type"] == "response.created", "response_created_out_of_order")
    for event in events:
        if "response" in event:
            response = event["response"]
            require(isinstance(response, dict) and response.get("id") == rid, "response_identity_mismatch")
            if event["type"] != "response.completed":
                require(response.get("output", []) == [], "premature_response_output")
    added, done = {}, {}
    for position, event in enumerate(events):
        kind = event["type"]
        if kind not in {"response.output_item.added", "response.output_item.done"}:
            continue
        item, index = event.get("item"), event.get("output_index")
        require(type(index) is int and 0 <= index < MAX_ITEMS, "invalid_output_index")
        require(isinstance(item, dict) and item.get("type") in {"message", "reasoning"}, "unsupported_output_item")
        require(isinstance(item.get("id"), str) and bool(item["id"]), "missing_item_identity")
        target = added if kind.endswith(".added") else done
        require(index not in target, "duplicate_item_lifecycle")
        target[index] = (position, item)
    require(bool(added) and set(added) == set(done) == set(range(len(added))), "unfinished_or_gapped_items")
    require(len({item["id"] for _, item in added.values()}) == len(added), "duplicate_item_identity")
    for index in added:
        start, first = added[index]
        finish, last = done[index]
        require(start < finish and first["id"] == last["id"] and first["type"] == last["type"], "item_identity_or_order_mismatch")
    messages = [(i, pair) for i, pair in done.items() if pair[1]["type"] == "message"]
    require(len(messages) == 1, "single_assistant_message_required")
    index, (finish, item) = messages[0]
    start, first = added[index]
    require(first.get("role") == "assistant" and first.get("status") == "in_progress"
            and first.get("content") == [], "invalid_message_start")
    require(item.get("role") == "assistant" and item.get("status") == "completed"
            and isinstance(item.get("content"), list) and len(item["content"]) == 1, "invalid_completed_message")
    part = item["content"][0]
    require(isinstance(part, dict) and part.get("type") == "output_text"
            and part.get("annotations", []) == [] and isinstance(part.get("text"), str)
            and bool(part["text"].strip()), "invalid_completed_text")
    text = part["text"]
    kinds = {"response.content_part.added", "response.content_part.done",
             "response.output_text.delta", "response.output_text.done"}
    related = [(n, e) for n, e in enumerate(events) if e["type"] in kinds]
    require(len(related) >= 4, "incomplete_text_lifecycle")
    require(related[0][1]["type"] == "response.content_part.added"
            and related[-2][1]["type"] == "response.output_text.done"
            and related[-1][1]["type"] == "response.content_part.done"
            and all(e["type"] == "response.output_text.delta" for _, e in related[1:-2]), "invalid_text_lifecycle_order")
    for position, event in related:
        require(start < position < finish and event.get("item_id") == item["id"]
                and type(event.get("output_index")) is int and event["output_index"] == index
                and type(event.get("content_index")) is int and event["content_index"] == 0, "text_identity_or_index_mismatch")
        require(event.get("annotations", []) == [], "annotated_stream_text")
    first_part, last_part = related[0][1].get("part"), related[-1][1].get("part")
    for candidate in (first_part, last_part):
        require(isinstance(candidate, dict) and candidate.get("type") == "output_text"
                and candidate.get("annotations", []) == [], "invalid_stream_content_part")
    require(first_part.get("text") == "" and last_part.get("text") == text, "content_part_text_mismatch")
    deltas = [e.get("delta") for _, e in related[1:-2]]
    require(all(isinstance(d, str) for d in deltas) and "".join(deltas) == text
            and related[-2][1].get("text") == text, "completed_text_mismatch")
    # Every completed item is unchanged. Final metadata remains the provider's.
    return {**terminal, "output": [done[i][1] for i in range(len(done))]}


def normalised_chunks(chunks, deadline, observation):
    events = []

    def publish(**values):
        with observation["lock"]:
            if not observation["sealed"]:
                observation["detail"].update(values)

    try:
        for event in frozen._events(chunks, deadline):
            if event is None:
                yield b"data: [DONE]\n\n"
                continue
            events.append(event)
            if event["type"] == "response.completed":
                response = event.get("response")
                output = response.get("output") if isinstance(response, dict) else None
                shape = "empty_list" if type(output) is list and not output else "populated_list" if type(output) is list else "missing_or_invalid"
                publish(original_terminal_output_shape=shape)
                if shape == "empty_list":
                    publish(attempted=True)
                    response = assembled_response(events, response)
                    event = {**event, "response": response}
                    publish(applied=True, completed_item_count=len(response["output"]))
                elif type(output) is list:
                    # An existing terminal output must agree with supplied done items.
                    for previous in events[:-1]:
                        if previous["type"] == "response.output_item.done":
                            index = previous.get("output_index")
                            require(type(index) is int and 0 <= index < len(output)
                                    and previous.get("item") == output[index], "populated_terminal_conflicts_with_done_item")
            yield ("data: " + canonical_json(event) + "\n\n").encode("utf-8")
        frozen._remaining(deadline)
    except AssemblyBlocked as exc:
        publish(failure_code=exc.code)
        raise
    finally:
        events.clear()


class StreamAssemblyChatGPTPlanGenerator(MissingMimeChatGPTPlanGenerator):
    def __init__(self, trace_root, credential_provider, transport=None, *, opener_factory=None):
        super().__init__(trace_root, credential_provider, transport, opener_factory=opener_factory)
        self._raw_transport = self._injected_transport

    def __call__(self, messages, schema):
        if not self.case_id or not re.fullmatch(r"[A-Za-z0-9_-]+", self.case_id):
            raise ValueError("A unique safe case ID is required")
        target = self.trace_root / self.case_id
        if target.exists():
            raise FileExistsError("Trace already exists")
        observation = {"lock": threading.Lock(), "sealed": False,
                       "detail": {"version": 1, "attempted": False, "applied": False}}
        send = self._raw_transport

        def normalise(payload, token, deadline):
            yield from normalised_chunks(send(payload, token, deadline), deadline, observation)

        self._injected_transport = normalise
        try:
            return super().__call__(messages, schema)
        finally:
            with observation["lock"]:
                observation["sealed"] = True
                detail = dict(observation["detail"])
            self.last_call["stream_assembly"] = detail
            if (target / "receipt.json").is_file():
                (target / "receipt.json").write_text(canonical_json(self.last_call), encoding="utf-8")
