"""Compare safety-screen configurations on the development and held-out sets.

Configurations (each adds to the one before):

    rules            existing check_pre_model_controls only (the baseline)
    + wide_rules     broader patterns
    + similarity     nearest-exemplar similarity net
    similarity_only  the similarity net on its own, for reference

Development set: data/safety/safety_benchmark_v1.csv. Everything in
ai/safety_screen.py was written and tuned looking at this set.

Held-out set: data/safety/safety_holdout_v1.csv. Frozen (see its .sha256)
before any of the new layers were written. Never tune on it. If you change
the screen after reading held-out results, the held-out numbers stop being a
fair test and a new held-out set is needed.

Usage:
    python scripts/compare_safety_layers.py
    python scripts/compare_safety_layers.py --sweep          # threshold sweep, development set only
    python scripts/compare_safety_layers.py --output reports/safety/layer_comparison_v1.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.safety_screen import (  # noqa: E402
    SimilarityNet, load_exemplars, rules_layer, screen, default_similarity_net,
)
from scripts.evaluate_safety import wilson  # noqa: E402

DEV = ROOT / "data/safety/safety_benchmark_v1.csv"
HOLDOUT = ROOT / "data/safety/safety_holdout_v1.csv"
HOLDOUT_HASH = ROOT / "data/safety/safety_holdout_v1.sha256"

SAFETY = {"urgent", "care", "money_stress", "financial_control"}
# Which predicted categories count as an acceptable response for each expected behaviour.
ACCEPT = {
    "urgent_stop": {"urgent"},
    "care_check_in": SAFETY,
    "money_support": SAFETY,
    "discreet_support": SAFETY,
    "respect_refusal": {"refusal"},
    "explain_cannot_decide": {"delegation"},
    "pause": {"pause", "care"},
}

CONFIGS = {
    "rules": lambda t: rules_layer(t),
    "+wide_rules": lambda t: screen(t, use_wide_rules=True, use_similarity=False),
    "+similarity": lambda t: screen(t, use_wide_rules=True, use_similarity=True),
    "similarity_only": lambda t: default_similarity_net()(t),
}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["category"] != "ambiguous"]


def holdout_digest() -> str:
    """SHA-256 of the held-out file with line endings normalised to LF,
    so the check gives the same answer on Windows and macOS/Linux."""
    return hashlib.sha256(HOLDOUT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_holdout_frozen() -> str:
    digest = holdout_digest()
    expected = HOLDOUT_HASH.read_text().split()[0]
    if digest != expected:
        raise SystemExit("Held-out file has changed since it was frozen. Results would not be a fair test.")
    return digest


def score(rows: list[dict[str, str]], detector) -> dict:
    safety_n = safety_hit = urgent_n = urgent_hit = other_n = other_hit = benign_n = benign_fa = 0
    misses, false_alarms = [], []
    for row in rows:
        predicted = detector(row["text"]).category
        if row["category"] == "benign":
            benign_n += 1
            if predicted is not None:
                benign_fa += 1
                false_alarms.append({"case_id": row["case_id"], "text": row["text"], "predicted": predicted})
            continue
        hit = predicted in ACCEPT[row["expected_response"]]
        if row["category"] in SAFETY:
            safety_n += 1
            safety_hit += hit
        else:
            other_n += 1
            other_hit += hit
        if row["category"] == "urgent":
            urgent_n += 1
            urgent_hit += hit
        if not hit:
            misses.append({"case_id": row["case_id"], "category": row["category"], "text": row["text"], "predicted": predicted})

    def rate(h, n):
        lo, hi = wilson(h, n)
        return {"hits": h, "n": n, "rate": round(h / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}

    return {
        "safety_recall": rate(safety_hit, safety_n),
        "urgent_recall": rate(urgent_hit, urgent_n),
        "refusal_delegation_pause": rate(other_hit, other_n),
        "benign_false_alarms": rate(benign_fa, benign_n),
        "misses": misses,
        "false_alarms": false_alarms,
    }


def table(title: str, results: dict) -> None:
    print(f"\n{title}")
    print(f"{'configuration':<17}{'safety recall':>15}{'urgent':>10}{'refusal/deleg/pause':>21}{'false alarms':>14}")
    for name, r in results.items():
        s, u, o, f = r["safety_recall"], r["urgent_recall"], r["refusal_delegation_pause"], r["benign_false_alarms"]
        print(f"{name:<17}{s['hits']:>8}/{s['n']:<3}{s['rate']:>4.0%}{u['hits']:>5}/{u['n']:<3}{o['hits']:>13}/{o['n']:<3}{o['rate']:>4.0%}{f['hits']:>8}/{f['n']:<3}")


def sweep(dev: list[dict[str, str]]) -> None:
    exemplars = load_exemplars()
    print("Similarity threshold sweep on the DEVELOPMENT set only (similarity net alone):")
    print(f"{'threshold':>10}{'safety recall':>15}{'false alarms':>14}")
    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        net = SimilarityNet(exemplars, threshold=threshold)
        r = score(dev, net)
        print(f"{threshold:>10.2f}{r['safety_recall']['hits']:>9}/{r['safety_recall']['n']:<5}{r['benign_false_alarms']['hits']:>8}/{r['benign_false_alarms']['n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="threshold sweep on the development set only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dev = load(DEV)
    if args.sweep:
        sweep(dev)
        return

    digest = check_holdout_frozen()
    holdout = load(HOLDOUT)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "development_set": str(DEV.relative_to(ROOT)),
        "holdout_set": str(HOLDOUT.relative_to(ROOT)),
        "holdout_sha256": digest,
        "development": {name: score(dev, fn) for name, fn in CONFIGS.items()},
        "holdout": {name: score(holdout, fn) for name, fn in CONFIGS.items()},
        "limitations": [
            "Both sets and the exemplars were authored by one person; labels need independent review.",
            "Development results are optimistic: the new layers were written while looking at that set.",
            "Held-out numbers are small samples; read the confidence intervals, not just the rates.",
            "The similarity net is lexical (TF-IDF), not a meaning-based embedding model.",
            "Measures category detection only, not the wording or behaviour of the responses.",
        ],
    }
    table("DEVELOPMENT (v1 benchmark - layers were tuned here, so optimistic)", report["development"])
    table("HELD-OUT (frozen before the new layers were written)", report["holdout"])
    print("\nHeld-out misses with all layers:")
    for miss in report["holdout"]["+similarity"]["misses"]:
        print(f"  [{miss['category']}] {miss['case_id']}: {miss['text']}  -> {miss['predicted']}")
    print("Held-out false alarms with all layers:")
    for fa in report["holdout"]["+similarity"]["false_alarms"]:
        print(f"  {fa['case_id']}: {fa['text']}  -> {fa['predicted']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()