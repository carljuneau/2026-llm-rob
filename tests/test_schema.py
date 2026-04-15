"""Tests for schema.py — validation and RoB derivation logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schema import (
    CRITERION_KEYS,
    SchemaValidationError,
    derive_overall_rob,
    validate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_criteria(judgment: str = "yes") -> dict:
    return {
        key: {"judgment": judgment, "quote": "supporting text"}
        for key in CRITERION_KEYS
    }


def _condition_bcd_output(judgment: str = "yes", overall: str = "low") -> dict:
    return {
        "study_id": "AuthorYear",
        "criteria": _full_criteria(judgment),
        "overall_rob": overall,
    }


# ---------------------------------------------------------------------------
# validate() — Condition A
# ---------------------------------------------------------------------------

def test_validate_condition_a_valid() -> None:
    result = validate({"study_id": "AuthorYear", "overall_rob": "serious"}, "A")
    assert result["study_id"] == "AuthorYear"
    assert result["overall_rob"] == "serious"


def test_validate_condition_a_normalises_case() -> None:
    result = validate({"study_id": "AuthorYear", "overall_rob": "  SERIOUS  "}, "A")
    assert result["overall_rob"] == "serious"


def test_validate_condition_a_missing_key_raises() -> None:
    with pytest.raises(SchemaValidationError, match="missing required key"):
        validate({"study_id": "AuthorYear"}, "A")


def test_validate_condition_a_bad_overall_rob_raises() -> None:
    with pytest.raises(SchemaValidationError, match="must be one of"):
        validate({"study_id": "AuthorYear", "overall_rob": "unknown"}, "A")


def test_validate_condition_a_extra_key_raises() -> None:
    with pytest.raises(SchemaValidationError, match="unexpected key"):
        validate({"study_id": "AuthorYear", "overall_rob": "low", "extra": "x"}, "A")


# ---------------------------------------------------------------------------
# validate() — Condition B/C/D
# ---------------------------------------------------------------------------

def test_validate_condition_b_valid() -> None:
    result = validate(_condition_bcd_output(), "B")
    assert result["study_id"] == "AuthorYear"
    assert result["overall_rob"] == "low"
    assert set(result["criteria"].keys()) == set(CRITERION_KEYS)


def test_validate_condition_c_valid() -> None:
    result = validate(_condition_bcd_output(judgment="no", overall="serious"), "C")
    assert result["overall_rob"] == "serious"
    for key in CRITERION_KEYS:
        assert result["criteria"][key]["judgment"] == "no"


def test_validate_condition_d_valid() -> None:
    result = validate(_condition_bcd_output(judgment="unclear", overall="moderate"), "D")
    assert result["overall_rob"] == "moderate"


def test_validate_condition_b_missing_criterion_key_raises() -> None:
    output = _condition_bcd_output()
    del output["criteria"][CRITERION_KEYS[0]]
    with pytest.raises(SchemaValidationError, match="missing required key"):
        validate(output, "B")


def test_validate_condition_b_bad_judgment_raises() -> None:
    output = _condition_bcd_output()
    output["criteria"][CRITERION_KEYS[0]]["judgment"] = "maybe"
    with pytest.raises(SchemaValidationError, match="must be one of"):
        validate(output, "B")


def test_validate_condition_b_extra_criterion_key_raises() -> None:
    output = _condition_bcd_output()
    output["criteria"]["extra_key"] = {"judgment": "yes", "quote": "x"}
    with pytest.raises(SchemaValidationError, match="unexpected key"):
        validate(output, "B")


def test_validate_unknown_condition_raises() -> None:
    with pytest.raises(SchemaValidationError):
        validate({"study_id": "x", "overall_rob": "low"}, "Z")


# ---------------------------------------------------------------------------
# derive_overall_rob()
# ---------------------------------------------------------------------------

def test_derive_all_yes_gives_low() -> None:
    assert derive_overall_rob(_full_criteria("yes")) == "low"


def test_derive_any_unclear_no_no_gives_moderate() -> None:
    criteria = _full_criteria("yes")
    criteria[CRITERION_KEYS[0]] = {"judgment": "unclear", "quote": ""}
    assert derive_overall_rob(criteria) == "moderate"


def test_derive_any_no_gives_serious() -> None:
    criteria = _full_criteria("yes")
    criteria[CRITERION_KEYS[2]] = {"judgment": "no", "quote": ""}
    assert derive_overall_rob(criteria) == "serious"


def test_derive_no_beats_unclear() -> None:
    criteria = _full_criteria("unclear")
    criteria[CRITERION_KEYS[0]] = {"judgment": "no", "quote": ""}
    assert derive_overall_rob(criteria) == "serious"


def test_derive_accepts_plain_judgment_strings() -> None:
    plain = {key: "yes" for key in CRITERION_KEYS}
    assert derive_overall_rob(plain) == "low"
