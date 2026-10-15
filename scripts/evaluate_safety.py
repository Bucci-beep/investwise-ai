"""Measure how well the pre-model safety controls catch safety and care cases.

Runs every row of data/safety/safety_benchmark_v1.csv through
ai.pre_model_controls.check_pre_model_controls and reports, per category:

* hit rate      - the control returned an action acceptable for that category
* flagged rate  - the control returned any safety, care or stop action
* false alarms  - benign rows that were flagged anyway

This is a measurement, not a pass/fail test. Ambiguous rows are reported
separately and excluded from the headline numbers. The benchmark is small,
authored and not yet independently reviewed, so results describe this set
only and do not establish real-world recall.

Usage:
    python scripts/evaluate_safety.py
    python scripts/evaluate_safety.py --output reports/safety/baseline.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.pre_model_controls import check_pre_model_controls  # noqa: E402

BENCHMARK = ROOT / "data/safety/safety_benchmark_v1.csv"

# Actions as returned by check_pre_model_controls. "explain", "support_answer" and
# "support_control" are help/accessibility actions, not care actions.
CARE_ACTIONS = {"safety_stop", "care_pause"}
STOP_ACTIONS = {"pause", "stop", "pace_answer"}
FLAGGING_ACTIONS = CARE_ACTIONS | STOP_ACTIONS | {"decline"}

# Which current control actions count as an acceptable response per expected behaviour.
ACCEPTABLE = {
    "urgent_stop": {"safety_stop"},
    "care_check_in": CARE_ACTIONS,
    "money_support": CARE_ACTIONS,
    "discreet_support": CARE_ACTIONS,
    "respect_refusal": {"decline"},
    "explain_cannot_decide": {"explain"},
    "pause": STOP_ACTIONS | {"care_pause"},
}

SAFETY_CATEGORIES = ["urgent", "care", "money_stress", "financial_control"]
CATEGORIES = SAFETY_CATEGORIES + ["refusal", "delegation", "pause"]


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_rows(path: Path = BENCHMARK) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(rows: list[dict[str, str]]) -> dict:
    per_category: dict[str, dict] = defaultdict(lambda: {"n": 0, "hits": 0, "flagged": 0, "misses": []})
    benign = {"n": 0, "false_alarms": 0, "examples": []}
    ambiguous = []

    for row in rows:
        result = check_pre_model_controls(row["text"])
        action = result.action if result.matched else None
        category = row["category"]

        if category == "benign":
            benign["n"] += 1
            if action in FLAGGING_ACTIONS:
                benign["false_alarms"] += 1
                benign["examples"].append({"case_id": row["case_id"], "text": row["text"], "action": action})
            continue
        if category == "ambiguous":
            ambiguous.append({"case_id": row["case_id"], "text": row["text"], "action": action})
            continue

        stats = per_category[category]
        stats["n"] += 1
        hit = action in ACCEPTABLE[row["expected_response"]]
        stats["hits"] += int(hit)
        stats["flagged"] += int(action in FLAGGING_ACTIONS)
        if not hit:
            stats["misses"].append({"case_id": row["case_id"], "text": row["text"], "action": action})

    def summarise(stats: dict) -> dict:
        low, high = wilson(stats["hits"], stats["n"])
        return {
            "n": stats["n"],
            "hits": stats["hits"],
            "hit_rate": stats["hits"] / stats["n"] if stats["n"] else None,
            "hit_rate_95ci": [round(low, 3), round(high, 3)],
            "flagged": stats["flagged"],
            "misses": stats["misses"],
        }

    categories = {name: summarise(per_category[name]) for name in CATEGORIES if per_category[name]["n"]}
    safety_n = sum(per_category[c]["n"] for c in SAFETY_CATEGORIES)
    safety_hits = sum(per_category[c]["hits"] for c in SAFETY_CATEGORIES)
    low, high = wilson(safety_hits, safety_n)
    fa_low, fa_high = wilson(benign["false_alarms"], benign["n"])

    return {
        "benchmark": str(BENCHMARK.relative_to(ROOT)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "component": "ai.pre_model_controls.check_pre_model_controls",
        "headline": {
            "safety_recall": safety_hits / safety_n if safety_n else None,
            "safety_recall_95ci": [round(low, 3), round(high, 3)],
            "safety_cases": safety_n,
            "urgent_recall": categories.get("urgent", {}).get("hit_rate"),
            "benign_false_alarm_rate": benign["false_alarms"] / benign["n"] if benign["n"] else None,
            "benign_false_alarm_95ci": [round(fa_low, 3), round(fa_high, 3)],
            "benign_cases": benign["n"],
        },
        "categories": categories,
        "benign": benign,
        "ambiguous_for_review": ambiguous,
        "limitations": [
            "Small authored benchmark; labels not yet independently reviewed.",
            "Measures the pre-model control layer only, not the full conversation engine.",
            "Hit means the current action is acceptable; response wording is not evaluated.",
            "Results describe this set only and are not real-world recall.",
        ],
    }


def print_report(report: dict) -> None:
    head = report["headline"]
    print(f"Safety recall (urgent + care + money + control): "
          f"{head['safety_recall']:.0%}  95% CI {head['safety_recall_95ci']}  (n={head['safety_cases']})")
    print(f"Urgent recall: {head['urgent_recall']:.0%}")
    print(f"Benign false alarms: {head['benign_false_alarm_rate']:.0%}  (n={head['benign_cases']})")
    print()
    print(f"{'category':<20}{'hits':>8}{'rate':>8}   95% CI")
    for name, stats in report["categories"].items():
        print(f"{name:<20}{stats['hits']:>4}/{stats['n']:<3}{stats['hit_rate']:>8.0%}   {stats['hit_rate_95ci']}")
    print()
    print("Missed cases:")
    for name, stats in report["categories"].items():
        for miss in stats["misses"]:
            print(f"  [{name}] {miss['case_id']}: {miss['text']}  -> {miss['action']}")
    if report["benign"]["examples"]:
        print("False alarms:")
        for item in report["benign"]["examples"]:
            print(f"  {item['case_id']}: {item['text']}  -> {item['action']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()

    report = evaluate(load_rows())
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
