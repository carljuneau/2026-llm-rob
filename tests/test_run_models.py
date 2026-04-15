"""Tests for pure (non-API) functions in run_models.py."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_models import (
    RunModelsError,
    build_condition_bcd_schema_example,
    build_criteria_block,
    load_criterion_definitions,
    load_study_ids_from_gold_csv,
)
from schema import CRITERION_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gold_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["Study first author", "Overall risk of bias"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_criteria_csv(path: Path) -> None:
    fieldnames = ["Criterion (yes condition)", "Code key"]
    rows = [
        {"Code key": key, "Criterion (yes condition)": f"Yes if {key}"}
        for key in CRITERION_KEYS
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# load_study_ids_from_gold_csv()
# ---------------------------------------------------------------------------

def test_load_study_ids_returns_ordered_list(tmp_path: Path) -> None:
    csv_path = tmp_path / "gold.csv"
    _write_gold_csv(csv_path, [
        {"Study first author": "Alpha", "Overall risk of bias": "serious"},
        {"Study first author": "Beta", "Overall risk of bias": "serious"},
        {"Study first author": "Gamma", "Overall risk of bias": "low"},
    ])
    ids = load_study_ids_from_gold_csv(csv_path)
    assert ids == ["Alpha", "Beta", "Gamma"]


def test_load_study_ids_skips_note_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "gold.csv"
    _write_gold_csv(csv_path, [
        {"Study first author": "Alpha", "Overall risk of bias": "serious"},
        {"Study first author": "Note: see methods", "Overall risk of bias": ""},
        {"Study first author": "Table 1. Something", "Overall risk of bias": ""},
    ])
    ids = load_study_ids_from_gold_csv(csv_path)
    assert ids == ["Alpha"]


def test_load_study_ids_skips_blank_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "gold.csv"
    _write_gold_csv(csv_path, [
        {"Study first author": "Alpha", "Overall risk of bias": "serious"},
        {"Study first author": "", "Overall risk of bias": ""},
    ])
    ids = load_study_ids_from_gold_csv(csv_path)
    assert ids == ["Alpha"]


def test_load_study_ids_raises_on_duplicate(tmp_path: Path) -> None:
    csv_path = tmp_path / "gold.csv"
    _write_gold_csv(csv_path, [
        {"Study first author": "Alpha", "Overall risk of bias": "serious"},
        {"Study first author": "Alpha", "Overall risk of bias": "serious"},
    ])
    with pytest.raises(RunModelsError, match="Duplicate"):
        load_study_ids_from_gold_csv(csv_path)


def test_load_study_ids_raises_on_missing_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", newline="") as f:
        f.write("Wrong column\nAlpha\n")
    with pytest.raises(RunModelsError, match="missing required column"):
        load_study_ids_from_gold_csv(csv_path)


def test_load_study_ids_raises_if_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_study_ids_from_gold_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# load_criterion_definitions()
# ---------------------------------------------------------------------------

def test_load_criterion_definitions_returns_all_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteria.csv"
    _write_criteria_csv(csv_path)
    defs = load_criterion_definitions(csv_path)
    assert [d.code_key for d in defs] == list(CRITERION_KEYS)


def test_load_criterion_definitions_preserves_criterion_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteria.csv"
    # Write in reverse order — output should match CRITERION_KEYS order regardless
    fieldnames = ["Criterion (yes condition)", "Code key"]
    rows = [
        {"Code key": key, "Criterion (yes condition)": f"Yes if {key}"}
        for key in reversed(CRITERION_KEYS)
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    defs = load_criterion_definitions(csv_path)
    assert [d.code_key for d in defs] == list(CRITERION_KEYS)


def test_load_criterion_definitions_raises_on_missing_key(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteria.csv"
    fieldnames = ["Criterion (yes condition)", "Code key"]
    # Omit the last criterion key
    rows = [
        {"Code key": key, "Criterion (yes condition)": f"Yes if {key}"}
        for key in CRITERION_KEYS[:-1]
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RunModelsError, match="missing required code key"):
        load_criterion_definitions(csv_path)


def test_load_criterion_definitions_raises_if_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_criterion_definitions(Path("/nonexistent/criteria.csv"))


# ---------------------------------------------------------------------------
# build_criteria_block()
# ---------------------------------------------------------------------------

def test_build_criteria_block_numbers_lines(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteria.csv"
    _write_criteria_csv(csv_path)
    defs = load_criterion_definitions(csv_path)
    block = build_criteria_block(defs)
    lines = block.splitlines()
    assert len(lines) == 8
    assert lines[0].startswith("1.")
    assert lines[-1].startswith("8.")


def test_build_criteria_block_contains_code_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "criteria.csv"
    _write_criteria_csv(csv_path)
    defs = load_criterion_definitions(csv_path)
    block = build_criteria_block(defs)
    for key in CRITERION_KEYS:
        assert key in block


# ---------------------------------------------------------------------------
# build_condition_bcd_schema_example()
# ---------------------------------------------------------------------------

def test_build_schema_example_has_all_criterion_keys() -> None:
    example = build_condition_bcd_schema_example("TestStudy")
    assert example["study_id"] == "TestStudy"
    assert set(example["criteria"].keys()) == set(CRITERION_KEYS)


def test_build_schema_example_criterion_values_are_dicts() -> None:
    example = build_condition_bcd_schema_example("TestStudy")
    for key in CRITERION_KEYS:
        assert "judgment" in example["criteria"][key]
        assert "quote" in example["criteria"][key]
