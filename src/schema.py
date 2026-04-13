"""Schema and validation helpers for the LLM RoB pipeline.

This module keeps the core contract for model outputs in one place:

- the 8 canonical criterion keys
- strict validation for each supported output shape
- pure-Python derivation of overall risk of bias from criterion judgments

It uses only the Python standard library.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# These keys are the shared contract across the pipeline.
# Keep them in one place so other modules can import them.
CRITERION_KEYS = (
    "study_group_representative",
    "intervention_and_participants_defined",
    "outcome_assessed_for_60pct",
    "follow_up_length_reported",
    "outcome_assessors_blinded",
    "outcome_definition_objective_precise",
    "important_prognostic_factors_accounted_for",
    "analysis_described_and_effect_quantified",
)

ALLOWED_JUDGMENTS = ("yes", "no", "unclear")
ALLOWED_OVERALL_ROB = ("low", "moderate", "serious")
ALLOWED_CONDITIONS = ("A", "B", "C", "D")
CONDITIONS_WITH_CRITERIA = ("B", "C", "D")


Judgment = Literal["yes", "no", "unclear"]
OverallROB = Literal["low", "moderate", "serious"]


class SchemaValidationError(ValueError):
    """Raised when a model output does not match the expected schema."""


class CriterionResult(TypedDict):
    judgment: Judgment
    quote: str


class CriteriaResult(TypedDict):
    study_group_representative: CriterionResult
    intervention_and_participants_defined: CriterionResult
    outcome_assessed_for_60pct: CriterionResult
    follow_up_length_reported: CriterionResult
    outcome_assessors_blinded: CriterionResult
    outcome_definition_objective_precise: CriterionResult
    important_prognostic_factors_accounted_for: CriterionResult
    analysis_described_and_effect_quantified: CriterionResult


class ConditionAResult(TypedDict):
    study_id: str
    overall_rob: OverallROB


class ConditionBCDResult(TypedDict):
    study_id: str
    criteria: CriteriaResult
    overall_rob: OverallROB


def validate(output: dict[str, Any], condition: str) -> ConditionAResult | ConditionBCDResult:
    """Validate one parsed model output.

    Parameters
    ----------
    output:
        A Python dict, usually produced by ``json.loads(...)``.
    condition:
        One of ``A``, ``B``, ``C``, or ``D``.

    Returns
    -------
    dict
        A normalized dict with the same shape as the expected schema.
        Controlled vocabulary fields are lower-cased and stripped.

    Raises
    ------
    SchemaValidationError
        If the structure, keys, types, or allowed values are wrong.
    """
    normalized_condition = _normalize_condition(condition)

    if normalized_condition == "A":
        return _validate_condition_a(output)

    return _validate_condition_bcd(output)


def derive_overall_rob(criteria: dict[str, Any]) -> OverallROB:
    """Derive overall risk of bias from the 8 criterion judgments.

    Rule:
    - all 8 yes -> low
    - any unclear, with no no -> moderate
    - any no -> serious

    This function accepts either:
    - the full validated ``criteria`` dict where each value is
      ``{"judgment": ..., "quote": ...}``, or
    - a simpler dict where each value is just ``"yes" | "no" | "unclear"``.
    """
    criteria_dict = _expect_dict(criteria, path="criteria")
    _check_exact_keys(criteria_dict, expected_keys=CRITERION_KEYS, path="criteria")

    judgments: list[Judgment] = []

    for key in CRITERION_KEYS:
        value = criteria_dict[key]

        if isinstance(value, dict):
            _check_exact_keys(value, expected_keys=("judgment", "quote"), path=f"criteria.{key}")
            judgment = _normalize_judgment(value["judgment"], path=f"criteria.{key}.judgment")
        else:
            judgment = _normalize_judgment(value, path=f"criteria.{key}")

        judgments.append(judgment)

    if any(judgment == "no" for judgment in judgments):
        return "serious"
    if any(judgment == "unclear" for judgment in judgments):
        return "moderate"
    return "low"


def _validate_condition_a(output: dict[str, Any]) -> ConditionAResult:
    output_dict = _expect_dict(output, path="output")
    _check_exact_keys(output_dict, expected_keys=("study_id", "overall_rob"), path="output")

    return {
        "study_id": _normalize_study_id(output_dict["study_id"], path="output.study_id"),
        "overall_rob": _normalize_overall_rob(output_dict["overall_rob"], path="output.overall_rob"),
    }


def _validate_condition_bcd(output: dict[str, Any]) -> ConditionBCDResult:
    output_dict = _expect_dict(output, path="output")
    _check_exact_keys(
        output_dict,
        expected_keys=("study_id", "criteria", "overall_rob"),
        path="output",
    )

    criteria_dict = _expect_dict(output_dict["criteria"], path="output.criteria")
    _check_exact_keys(criteria_dict, expected_keys=CRITERION_KEYS, path="output.criteria")

    normalized_criteria: CriteriaResult = {
        key: _validate_single_criterion(criteria_dict[key], path=f"output.criteria.{key}")
        for key in CRITERION_KEYS
    }

    return {
        "study_id": _normalize_study_id(output_dict["study_id"], path="output.study_id"),
        "criteria": normalized_criteria,
        "overall_rob": _normalize_overall_rob(output_dict["overall_rob"], path="output.overall_rob"),
    }


def _validate_single_criterion(value: Any, path: str) -> CriterionResult:
    criterion_dict = _expect_dict(value, path=path)
    _check_exact_keys(criterion_dict, expected_keys=("judgment", "quote"), path=path)

    return {
        "judgment": _normalize_judgment(criterion_dict["judgment"], path=f"{path}.judgment"),
        "quote": _expect_string(criterion_dict["quote"], path=f"{path}.quote", allow_empty=True),
    }


def _normalize_condition(condition: str) -> str:
    if not isinstance(condition, str):
        raise SchemaValidationError(
            f"condition must be a string in {ALLOWED_CONDITIONS}; got {type(condition).__name__}"
        )

    normalized = condition.strip().upper()
    if normalized not in ALLOWED_CONDITIONS:
        allowed = ", ".join(ALLOWED_CONDITIONS)
        raise SchemaValidationError(f"condition must be one of {allowed}; got {condition!r}")

    return normalized


def _normalize_judgment(value: Any, path: str) -> Judgment:
    return _normalize_choice(value, allowed=ALLOWED_JUDGMENTS, path=path)  # type: ignore[return-value]


def _normalize_overall_rob(value: Any, path: str) -> OverallROB:
    return _normalize_choice(value, allowed=ALLOWED_OVERALL_ROB, path=path)  # type: ignore[return-value]


def _normalize_choice(value: Any, allowed: tuple[str, ...], path: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{path} must be a string; got {type(value).__name__}")

    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(allowed)
        raise SchemaValidationError(f"{path} must be one of {allowed_text}; got {value!r}")

    return normalized


def _normalize_study_id(value: Any, path: str) -> str:
    study_id = _expect_string(value, path=path, allow_empty=False)
    return study_id.strip()


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be an object/dict; got {type(value).__name__}")
    return value


def _expect_string(value: Any, path: str, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{path} must be a string; got {type(value).__name__}")

    if not allow_empty and not value.strip():
        raise SchemaValidationError(f"{path} must be a non-empty string")

    return value


def _check_exact_keys(value: dict[str, Any], expected_keys: tuple[str, ...], path: str) -> None:
    actual_keys = set(value.keys())
    expected = set(expected_keys)

    missing = [key for key in expected_keys if key not in actual_keys]
    unexpected = sorted(actual_keys - expected)

    if missing:
        raise SchemaValidationError(f"{path} is missing required key(s): {', '.join(missing)}")

    if unexpected:
        raise SchemaValidationError(f"{path} has unexpected key(s): {', '.join(unexpected)}")


__all__ = [
    "ALLOWED_CONDITIONS",
    "ALLOWED_JUDGMENTS",
    "ALLOWED_OVERALL_ROB",
    "CONDITIONS_WITH_CRITERIA",
    "CRITERION_KEYS",
    "ConditionAResult",
    "ConditionBCDResult",
    "CriteriaResult",
    "CriterionResult",
    "SchemaValidationError",
    "derive_overall_rob",
    "validate",
]
