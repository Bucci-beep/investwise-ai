"""Standalone process adapter for the unchanged, packaged D5 interpreter."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RUNTIME = HERE / "runtime"


def bootstrap():
    """Never mix this frozen runtime with the application's older ai package."""
    for name, module in tuple(sys.modules.items()):
        if name == "ai" or name.startswith("ai."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(RUNTIME):
                raise RuntimeError("Run this demo in a separate Python process")
    sys.path.insert(0, str(RUNTIME))


def external_path(path):
    value = Path(path).expanduser().absolute()
    if value.is_symlink() or value.resolve().is_relative_to(REPOSITORY):
        raise ValueError("Credentials and session output must stay outside the repository")
    return value


def private_directory(path):
    path = external_path(path)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def write_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class DemoStopped(RuntimeError):
    """A fixed safe error; never carry arbitrary provider or credential text."""


class NoOtherQuestions:
    source_hash = "standalone_d5_no_other_question_interpretation"

    def resolve(self, question_id, text):
        raise DemoStopped("Only D5 is available in this demonstration")


class NoProfileSave:
    def save(self, *args, **kwargs):
        raise DemoStopped("This D5 example does not save a financial profile")


class D5Session:
    def __init__(self, credentials, output_dir):
        bootstrap()
        from ai.d5_single_interpreter_transport import SingleInterpreterGenerator
        from ai.question_spec import BusinessSpec
        self.output_dir = Path(output_dir)
        self.spec = BusinessSpec.load(RUNTIME / "config/business-decision-spec.json")
        if len(self.spec.question_order) != 15:
            raise DemoStopped("The unchanged fifteen-question specification is required")
        self.backend = SingleInterpreterGenerator(self.output_dir / "traces", credentials)
        self.failed = False
        self.sequence = 0
        self.last_record = None
        self.new_example()

    def new_example(self):
        if self.failed:
            raise DemoStopped("The session stopped after a failure; no automatic retry")
        from ai.d5_single_interpreter_contract_v2 import D5SingleInterpreter
        from ai.d5_single_interpreter_engine_v2 import D5BusinessStandardEngine
        self.captured = {}

        def generate(messages, schema):
            raw = self.backend(messages, schema)
            self.captured["raw_output"] = raw
            return raw

        self.interpreter = D5SingleInterpreter(self.spec, generate)
        self.engine = D5BusinessStandardEngine(
            self.spec, d5_interpreter=self.interpreter,
            question_resolver=NoOtherQuestions(), store=NoProfileSave())
        self.engine.state.current_index = self.spec.question_order.index("D5")

    @property
    def question(self):
        return self.spec.question("D5").question

    @property
    def confirmed(self):
        return self.engine.state.answers["D5"].confirmed_version is not None

    def command(self, command):
        """UI actions only; all financial state changes stay in the old engine."""
        if self.failed:
            raise DemoStopped("The session has stopped")
        if command == "/new":
            self.new_example()
            return {"action": "new_example", "message": "A separate fictional D5 example has started."}
        if command == "/resume":
            if self.confirmed or self.engine.state.current_question_id != "D5":
                return {"action": "d5_complete", "message": "D5 has ended. Use /change, /new or /exit."}
            return self.engine.resume()
        if command in {"/confirm", "/change"}:
            if self.engine.state.stopped_for_safety:
                return {"action": "safety_stop", "message": "This example remains stopped for safety."}
            if "user_stop" in self.engine.state.active_holds:
                return {"action": "stop", "message": "This example remains stopped."}
            if self.engine.state.paused:
                return {"action": "paused", "message": "Resume before confirming or changing an answer."}
            if command == "/change":
                result = self.engine.correct_answer("D5")
                result["message"] = "The previous D5 answer is invalidated. Give the complete replacement answer."
                return result
            if self.confirmed:
                return {"action": "confirmed", "message": "D5 is already confirmed for this example."}
            if self.engine.state.answers["D5"].candidate_option_id is None:
                return {"action": "unresolved", "message": "There is no pending financial answer to confirm."}
            result = self.engine.confirm_current(True)
            if self.confirmed:
                result = dict(result, action="d5_complete", message=(
                    "D5 is confirmed in this session only. This standalone example ends here; "
                    "no profile has been saved. Use /change, /new or /exit."))
            return result
        if command in {"/help", "/pause", "/stop"}:
            return self.submit(command[1:])
        return {"action": "unknown_command", "message": "Use /help, /confirm, /change, /pause, /resume, /stop, /new or /exit."}

    def submit(self, text, *, case_id=None):
        if self.failed:
            raise DemoStopped("The session has stopped")
        if self.confirmed or self.engine.state.current_question_id != "D5":
            return {"action": "d5_complete", "message": "D5 has ended. Use /change, /new or /exit; no other question will be interpreted."}
        self.sequence += 1
        identifier = case_id or f"reply-{self.sequence:04d}-{uuid.uuid4().hex[:8]}"
        self.backend.case_id = identifier
        self.backend.last_call = {}
        self.captured.clear()
        before = self.backend.calls
        try:
            result = self.engine.submit_answer(text)
        except Exception:
            self.failed = True
            write_json(self.output_dir / f"{identifier}-failure.json", {
                "case_id": identifier, "actual_model_calls": self.backend.calls - before,
                "safe_backend_receipt": self.backend.last_call, "retry": False})
            raise DemoStopped("D5 stopped after a runtime or access failure; no retry or fallback") from None
        if self.backend.calls - before not in {0, 1}:
            self.failed = True
            raise DemoStopped("The one-interpretation-call boundary was exceeded")
        from ai.llm_shadow import strict_json
        raw = self.captured.get("raw_output")
        try:
            parsed = strict_json(raw) if raw is not None else None
        except ValueError:
            parsed = None
        decision = self.interpreter.last_decision
        answer = self.engine.state.answers["D5"]
        self.last_record = {
            "id": identifier, "input": text, "raw": parsed, "raw_output": raw,
            "result": result, "decision": decision.as_dict() if decision and raw is not None else None,
            "actual_model_calls": self.backend.calls - before,
            "backend_call": dict(self.backend.last_call), "candidate": answer.candidate_option_id,
            "selected": answer.selected_option_id, "confirmed_version": answer.confirmed_version,
            "saved_version": self.engine.state.saved_version, "answer_source": answer.source}
        write_json(self.output_dir / f"{identifier}.json", self.last_record)
        return result


def render(result):
    """Display authored engine text, with no generated actions or rendering code."""
    action = result.get("action", "unresolved")
    bucket = result.get("bucket")
    if bucket:
        print(f"\n{bucket}")
    emitted = set()
    for key in ("confirmation_text", "followup", "help_prompt", "message", "prompt", "pacing_text", "care_prompt", "support_prompt"):
        text = result.get(key)
        if isinstance(text, str) and text and text not in emitted:
            print(text)
            emitted.add(text)
    if not emitted:
        print({"pause": "Paused. Use /resume when ready.",
               "paused": "This example is paused. Use /resume when ready.",
               "resume": "Ready when you are. Give your complete current D5 answer.",
               "stop": "This example is stopped.",
               "safety_stop": "This example has stopped for safety. Any handover here is simulated; nobody has been contacted.",
               "finish_incomplete_or_pause": "This remains unresolved. You can pause or end this example."}.get(
                   action, "The answer remains unresolved. You can ask for help, pause or stop."))
    if action in {"care_pause", "pace_pause"}:
        print("You can use /resume when ready, or /stop to end this example.")
    if action == "confirm" or (result.get("candidate_option_id") and result.get("confirmation_text")):
        print("Use /confirm if correct, or /change to replace this answer.")
