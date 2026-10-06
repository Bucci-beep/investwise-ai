"""Focused acceptance tests for the isolated D5 v2 augmentation."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_d5_v2_augmentation.py"
SPEC = importlib.util.spec_from_file_location("build_d5_v2_augmentation", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


def test_authored_rows_have_required_schema_counts_and_no_v1_overlap():
    rows = generator.build_rows()
    summary = generator.validate(rows)

    assert summary == {
        "rows": 64,
        "class_counts": {"clarity": 48, "undecided": 8, "confusion": 8},
        "option_counts": generator.OPTION_COUNTS,
    }
    assert all(tuple(row) == generator.FIELDS for row in rows)
    assert all(row["question_id"] == "D5" and row["split"] == "train" for row in rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "row count"),
        (lambda rows: rows[0].update(question_id="D4"), "question_id"),
        (lambda rows: rows[0].update(split="test"), "split"),
        (lambda rows: rows[0].update(option_id="D5_WITHDRAWALS"), "option counts"),
        (lambda rows: rows[48].update(option_id="D5_SALARY"), "blank option_id"),
        (lambda rows: rows[0].update(option_id=""), "clarity rows"),
        (lambda rows: rows[1].update(id=rows[0]["id"]), "IDs"),
        (lambda rows: rows[1].update(text="  " + rows[0]["text"].upper() + "  "), "text duplicated"),
    ],
)
def test_validation_rejects_contract_violations(mutation, message):
    rows = generator.build_rows()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        generator.validate(rows, existing_texts=set())


def test_validation_rejects_exact_normalized_v1_overlap():
    rows = generator.build_rows()
    with pytest.raises(ValueError, match="overlaps v1 train"):
        generator.validate(rows, existing_texts={generator.normalize(rows[0]["text"])})


def test_generated_files_and_provenance_match(monkeypatch, tmp_path):
    output = tmp_path / "d5_training_additions.csv"
    provenance = tmp_path / "provenance.json"
    monkeypatch.setattr(generator, "DESTINATION", tmp_path)
    monkeypatch.setattr(generator, "OUTPUT", output)
    monkeypatch.setattr(generator, "PROVENANCE", provenance)

    generator.main()

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        assert tuple(rows[0]) == generator.FIELDS
        assert len(rows) == 64
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    assert metadata["creation_date"] == "2026-10-05"
    assert metadata["fictional"] is True
    assert metadata["row_counts"] == {"total": 64, "train": 64}
    assert metadata["sha256"][output.name] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert "were not modified" in metadata["v1_data_statement"]
    assert "not evidence of real-customer performance" in metadata["limitations"][0]
