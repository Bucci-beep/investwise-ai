"""One live pass over the original eleven cases through the manual demo adapter.

No generated inputs, new labels, retries, output repair or default-app routing.
Scoring reads the unchanged expectations only after inference has finished.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from session import D5Session, DemoStopped, HERE, REPOSITORY, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(credentials, output):
    evidence = REPOSITORY / "docs/d5-semantic-repair/evidence/after"
    source_manifest = json.loads((REPOSITORY / "docs/d5-semantic-repair/manifest.json").read_text())
    for name in ("inputs.json", "expectations.json"):
        path = evidence / name
        binding = next(row for row in source_manifest["evidence_files"]
                       if row["path"] == str(path.relative_to(REPOSITORY)))
        if digest(path) != binding["sha256"]:
            raise DemoStopped("The original eleven examples or expectations have changed")
    rows = json.loads((evidence / "inputs.json").read_text())["rows"]
    if [row["id"] for row in rows] != [f"D5-STD-{n:02d}" for n in range(1, 12)]:
        raise DemoStopped("Exactly the original eleven cases are required")
    # Freeze everything that can change this standalone run; exclude bytecode.
    files = {str(p.relative_to(REPOSITORY)): digest(p) for p in HERE.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    for name in ("inputs.json", "expectations.json"):
        files[str((evidence / name).relative_to(REPOSITORY))] = digest(evidence / name)
    write_json(output / "started.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "cases": 11,
        "maximum_model_calls": 11, "retry": False, "model": "gpt-6.1-sol",
        "reasoning_effort": "low", "labels_supplied_to_interpreter": False,
        "entry_point": "demos/d5/demo.py check-11", "hashes": files})
    session = D5Session(credentials, output)
    records = []
    failure = None
    for row in rows:
        session.new_example()
        try:
            session.submit(row["text"], case_id=row["id"])
        except DemoStopped:
            failure = {"case": row["id"], "reason": "runtime_or_access_failure",
                       "safe_backend_receipt": session.backend.last_call}
            break
        records.append(session.last_record)
        print(f"Recorded {len(records)}/11 | actual model calls: {session.backend.calls}", flush=True)
    changed = [name for name, expected in files.items()
               if digest(REPOSITORY / name) != expected]
    write_json(output / "execution.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "completed_cases": len(records),
        "actual_model_calls": session.backend.calls, "failure": failure,
        "complete": len(records) == 11 and failure is None and not changed,
        "changed_files": changed, "retry": False, "labels_supplied_to_interpreter": False,
        "choice_recovery": 0, "confirmation_or_save_actions": 0,
        "route": "standalone terminal adapter, live responses; no cached output replay"})
    if failure or changed:
        print("This pass stopped. No retry, fallback or later cases were run.")
        print("Private output:", output)
        return 2
    expectations = json.loads((evidence / "expectations.json").read_text())["cases"]
    review = []
    for row in records:
        expected = expectations[row["id"]]
        raw, result = row["raw"] or {}, row["result"]
        no_finance = (row["candidate"] is None and raw.get("option_id") is None
                      and not raw.get("facts"))
        if expected["kind"] == "clear":
            correct = (raw.get("semantic_bucket") == "clarity"
                       and raw.get("option_id") == expected["option"]
                       and row["candidate"] == expected["option"] and result["action"] == "confirm")
        elif expected["kind"] == "missing_information":
            correct = (no_finance and raw.get("semantic_bucket") is None
                       and raw.get("model_abstain") is True
                       and raw.get("missing_information") == ["current_funding_unknown"]
                       and result.get("clarification_topic") == "current_funding_unknown")
        elif expected["kind"] == "undecided":
            correct = (no_finance and raw.get("semantic_bucket") == "undecided"
                       and raw.get("model_abstain") is False
                       and raw.get("missing_information") == ["source_use_unknown:D5_SALARY"]
                       and result.get("clarification_topic") == "source_use_unknown:D5_SALARY")
        else:  # The existing terminology example is a pre-model Help action.
            correct = (no_finance and row["actual_model_calls"] == 0
                       and result["action"] == "explain"
                       and "pension fund" in result.get("help_prompt", ""))
        review.append({"id": row["id"], "kind": expected["kind"],
                       "expected_option": expected["option"], "candidate": row["candidate"],
                       "contract_outcome": (row["decision"] or {}).get("outcome"),
                       "expected_routing_and_option_match": bool(correct),
                       "meaning_and_wording_review_required": True})
    write_json(output / "comparison.json", {
        "scope": "Original eleven opened examples; automated routing/option checks, not independent semantic accuracy",
        "cases": review,
        "matched_cases": sum(item["expected_routing_and_option_match"] for item in review),
        "clear_candidate_denominator": 7, "actual_model_calls": session.backend.calls,
        "help_is_not_model_success": True, "abstention_is_not_a_completed_financial_answer": True})
    print("Same-eleven option/routing matches:", sum(r["expected_routing_and_option_match"] for r in review), "/11")
    print("Review raw meanings and wording in the private records:", output)
    return 0 if all(r["expected_routing_and_option_match"] for r in review) else 1
