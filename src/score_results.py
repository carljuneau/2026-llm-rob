"""Score LLM risk-of-bias outputs against expert gold labels.

This script reads:
- gold labels from ``data/public/Table - RoB_observational_studies.csv``
- parsed model outputs from ``results/parsed/*.json``

It writes:
- ``results/scored_summary.csv``

It also prints a human-readable report to stdout.

The parsed JSON files are expected to follow this naming pattern:
    <study_id>_<model>_<condition>.json

Condition A returns only an overall label.
Conditions B/C/D return 8 criterion judgments plus an overall label.

This file uses the Python standard library plus SciPy for t-test p-values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy import stats

from schema import (
    CRITERION_KEYS,
    CONDITIONS_WITH_CRITERIA,
    SchemaValidationError,
    derive_overall_rob,
    validate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_CSV = PROJECT_ROOT / "data" / "public" / "Table 1 - RoB_observational_studies.csv"
DEFAULT_PARSED_DIR = PROJECT_ROOT / "results" / "parsed"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "scored_summary.csv"

# These defaults match the current study protocol. They can be overridden from the CLI.
DEFAULT_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
)
DEFAULT_CONDITIONS = ("A", "B", "C", "D")

JUDGMENT_LABELS = ("yes", "no", "unclear")
OVERALL_LABELS = ("low", "moderate", "serious")

# 95% two-sided t critical values for small-sample confidence intervals.
# We only need small n here; after 30 df we fall back to 1.96.
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

GOLD_CSV_COLUMN_MAP = {
    "Selection bias (representative study group: yes/no)": "study_group_representative",
    "Attrition bias (complete follow-up assessment: yes/no)": "outcome_assessed_for_60pct",
    "Detection bias (blinded outcome assessor: yes/no)": "outcome_assessors_blinded",
    "Confounding (adjustment for important confounders: yes/no)": "important_prognostic_factors_accounted_for",
    "Reporting bias (well defined study group: yes/no)": "intervention_and_participants_defined",
    "Reporting bias (well defined follow-up: yes/no)": "follow_up_length_reported",
    "Reporting bias (well defined outcome: yes/no)": "outcome_definition_objective_precise",
    "Analyses (well defined: yes/no)": "analysis_described_and_effect_quantified",
}

SUMMARY_CSV_COLUMNS = [
    "study_id",
    "model",
    "condition",
    "criterion_agreement",
    "model_overall_rob",
    "gold_overall_rob",
    "overall_correct",
    "derived_overall_rob",
    "derived_overall_correct",
    "parse_failure",
]


@dataclass(frozen=True)
class GoldStudy:
    study_id: str
    criteria: dict[str, str]
    overall_rob: str


@dataclass
class ScoredResult:
    study_id: str
    model: str
    condition: str
    gold_overall_rob: str
    gold_criteria: dict[str, str]
    parse_failure: bool
    criterion_agreement: float | None = None
    model_overall_rob: str | None = None
    overall_correct: bool | None = None
    derived_overall_rob: str | None = None
    derived_overall_correct: bool | None = None
    model_criteria: dict[str, str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class MeanCI:
    n: int
    mean: float | None
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class Contrast:
    name: str
    left_model: str
    left_condition: str
    right_model: str
    right_condition: str


class ScoreResultsError(ValueError):
    """Raised for bad inputs or inconsistent scoring setup."""


def load_gold_labels(csv_path: Path) -> dict[str, GoldStudy]:
    """Load expert gold labels from the public CSV.

    The last row is a note row starting with ``Note:`` and is skipped.
    Values are normalized to lowercase.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Gold-label CSV not found: {csv_path}")

    gold_by_study: dict[str, GoldStudy] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required_headers = {
            "Study first author",
            "Overall risk of bias",
            *GOLD_CSV_COLUMN_MAP.keys(),
        }
        missing_headers = [header for header in required_headers if header not in (reader.fieldnames or [])]
        if missing_headers:
            raise ScoreResultsError(
                "Gold-label CSV is missing required column(s): " + ", ".join(sorted(missing_headers))
            )

        for row_number, row in enumerate(reader, start=2):
            raw_study_id = (row.get("Study first author") or "").strip()
            if not raw_study_id:
                continue
            if raw_study_id.startswith("Note:") or raw_study_id.startswith("Table "):
                continue
            if raw_study_id in gold_by_study:
                raise ScoreResultsError(f"Duplicate study_id in gold CSV: {raw_study_id!r}")

            criteria: dict[str, str] = {}
            for column_name, criterion_key in GOLD_CSV_COLUMN_MAP.items():
                raw_value = row.get(column_name, "")
                criteria[criterion_key] = _normalize_choice(
                    raw_value,
                    allowed=JUDGMENT_LABELS,
                    field_name=f"gold CSV row {row_number} column {column_name!r}",
                )

            gold_by_study[raw_study_id] = GoldStudy(
                study_id=raw_study_id,
                criteria=criteria,
                overall_rob=_normalize_choice(
                    row.get("Overall risk of bias", ""),
                    allowed=OVERALL_LABELS,
                    field_name=f"gold CSV row {row_number} column 'Overall risk of bias'",
                ),
            )

    if not gold_by_study:
        raise ScoreResultsError(f"No gold-study rows were loaded from {csv_path}")

    return gold_by_study


def load_validated_result(json_path: Path, condition: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load one parsed output file and validate it against ``schema.py``.

    Returns ``(parsed_dict, None)`` on success and ``(None, error_message)`` on failure.
    Missing files are treated as parse failures, not fatal errors.
    """
    if not json_path.exists():
        return None, f"missing file: {json_path.name}"

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {json_path.name}: {exc}"
    except OSError as exc:
        return None, f"could not read {json_path.name}: {exc}"

    try:
        validated = validate(payload, condition)
    except SchemaValidationError as exc:
        return None, f"schema validation failed for {json_path.name}: {exc}"

    return validated, None


def _strip_markdown_fences(text: str) -> str:
    """Strip outer markdown code fences from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def load_condition_a_raw_overall_rob(
    raw_dir: Path,
    study_ids: list[str] | tuple[str, ...],
    model: str,
) -> dict[str, str]:
    """Load model-reported overall RoB labels from raw condition A outputs."""
    if not raw_dir.exists():
        print(f"Warning: condition A raw directory not found: {raw_dir}", file=sys.stderr)
        return {}

    accepted_keys = {
        "risk_of_bias",
        "risk of bias",
        "risk of bias assessment",
    }
    overall_by_study: dict[str, str] = {}

    for study_id in study_ids:
        raw_path = raw_dir / f"{study_id}_{model}_A_raw.txt"
        if not raw_path.exists():
            print(
                f"Warning: missing condition A raw file for {study_id} / {model}: {raw_path.name}",
                file=sys.stderr,
            )
            continue

        try:
            raw_text = raw_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(
                f"Warning: could not read condition A raw file for {study_id} / {model}: {exc}",
                file=sys.stderr,
            )
            continue

        try:
            payload = json.loads(_strip_markdown_fences(raw_text))
        except json.JSONDecodeError as exc:
            print(
                f"Warning: invalid JSON in condition A raw file for {study_id} / {model}: {exc}",
                file=sys.stderr,
            )
            continue

        if not isinstance(payload, dict):
            print(
                f"Warning: condition A raw file for {study_id} / {model} did not contain a JSON object",
                file=sys.stderr,
            )
            continue

        matched_value: Any | None = None
        for key, value in payload.items():
            if isinstance(key, str) and key.strip().lower() in accepted_keys:
                matched_value = value
                break

        if matched_value is None:
            print(
                f"Warning: no overall RoB key found in condition A raw file for {study_id} / {model}",
                file=sys.stderr,
            )
            continue

        try:
            overall_by_study[study_id] = _normalize_choice(
                matched_value,
                allowed=OVERALL_LABELS,
                field_name=f"condition A overall_rob for {study_id} / {model}",
            )
        except ScoreResultsError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            continue

    return overall_by_study


def score_results(
    gold_by_study: dict[str, GoldStudy],
    parsed_dir: Path,
    models: tuple[str, ...],
    conditions: tuple[str, ...],
) -> list[ScoredResult]:
    """Score every expected study × model × condition combination.

    A missing or invalid parsed JSON file becomes ``parse_failure=True``.
    No values are imputed.
    """
    if not parsed_dir.exists():
        raise FileNotFoundError(f"Parsed-results directory not found: {parsed_dir}")

    scored: list[ScoredResult] = []

    for model in models:
        for condition in conditions:
            for study_id, gold in gold_by_study.items():
                json_path = parsed_dir / f"{study_id}_{model}_{condition}.json"
                parsed, error = load_validated_result(json_path, condition)

                if parsed is None:
                    scored.append(
                        ScoredResult(
                            study_id=study_id,
                            model=model,
                            condition=condition,
                            gold_overall_rob=gold.overall_rob,
                            gold_criteria=gold.criteria,
                            parse_failure=True,
                            error=error,
                        )
                    )
                    continue

                if parsed["study_id"] != study_id:
                    scored.append(
                        ScoredResult(
                            study_id=study_id,
                            model=model,
                            condition=condition,
                            gold_overall_rob=gold.overall_rob,
                            gold_criteria=gold.criteria,
                            parse_failure=True,
                            error=(
                                f"study_id mismatch in {json_path.name}: "
                                f"expected {study_id!r}, got {parsed['study_id']!r}"
                            ),
                        )
                    )
                    continue

                if condition in CONDITIONS_WITH_CRITERIA:
                    model_criteria = {
                        key: parsed["criteria"][key]["judgment"]
                        for key in CRITERION_KEYS
                    }
                    criterion_agreement = compute_criterion_agreement(gold.criteria, model_criteria)
                    derived_overall = derive_overall_rob(parsed["criteria"])
                else:
                    model_criteria = None
                    criterion_agreement = None
                    derived_overall = None

                model_overall = parsed["overall_rob"]
                scored.append(
                    ScoredResult(
                        study_id=study_id,
                        model=model,
                        condition=condition,
                        gold_overall_rob=gold.overall_rob,
                        gold_criteria=gold.criteria,
                        parse_failure=False,
                        criterion_agreement=criterion_agreement,
                        model_overall_rob=model_overall,
                        overall_correct=(model_overall == gold.overall_rob),
                        derived_overall_rob=derived_overall,
                        derived_overall_correct=(derived_overall == gold.overall_rob)
                        if derived_overall is not None
                        else None,
                        model_criteria=model_criteria,
                    )
                )

    return scored


def compute_criterion_agreement(gold_criteria: dict[str, str], model_criteria: dict[str, str]) -> float:
    """Return the proportion of the 8 criteria that match exactly."""
    matches = sum(
        1 for key in CRITERION_KEYS
        if gold_criteria[key] == model_criteria[key]
    )
    return matches / len(CRITERION_KEYS)


def mean_and_t_ci(values: list[float]) -> MeanCI:
    """Compute mean and a simple 95% t-based confidence interval."""
    n = len(values)
    if n == 0:
        return MeanCI(n=0, mean=None, lower=None, upper=None)

    mean_value = statistics.fmean(values)
    if n == 1:
        return MeanCI(n=1, mean=mean_value, lower=None, upper=None)

    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(n)
    t_crit = _t_critical_95(df=n - 1)
    margin = t_crit * standard_error
    return MeanCI(n=n, mean=mean_value, lower=mean_value - margin, upper=mean_value + margin)


def paired_t_p_value(differences: list[float]) -> float | None:
    """Return a two-sided paired t-test p-value for H0: mean(differences) = 0."""
    if len(differences) < 2:
        return None

    if all(math.isclose(value, differences[0], abs_tol=1e-12) for value in differences[1:]):
        if math.isclose(differences[0], 0.0, abs_tol=1e-12):
            return 1.0
        return 0.0

    result = stats.ttest_1samp(differences, popmean=0.0)
    p_value = float(result.pvalue)
    if math.isnan(p_value):
        return None
    return p_value


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Apply the Holm step-down correction to a family of p-values."""
    indexed = [(index, p_value) for index, p_value in enumerate(p_values) if p_value is not None]
    if not indexed:
        return [None for _ in p_values]

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted_sorted: list[float] = []
    running_max = 0.0
    for rank, (_, p_value) in enumerate(indexed, start=1):
        adjusted = min(1.0, p_value * (m - rank + 1))
        running_max = max(running_max, adjusted)
        adjusted_sorted.append(running_max)

    adjusted_by_index: list[float | None] = [None] * len(p_values)
    for (original_index, _), adjusted in zip(indexed, adjusted_sorted):
        adjusted_by_index[original_index] = adjusted
    return adjusted_by_index


def unweighted_cohen_kappa(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> float | None:
    """Compute plain Cohen's kappa for categorical labels.

    Returns ``None`` when kappa is undefined, for example because the expected
    agreement is exactly 1.0.
    """
    if not pairs:
        return None

    matrix = build_confusion_matrix(pairs, labels)
    n = len(pairs)
    observed = sum(matrix[label][label] for label in labels) / n

    gold_marginals = {
        gold_label: sum(matrix[gold_label][model_label] for model_label in labels) / n
        for gold_label in labels
    }
    model_marginals = {
        model_label: sum(matrix[gold_label][model_label] for gold_label in labels) / n
        for model_label in labels
    }
    expected = sum(gold_marginals[label] * model_marginals[label] for label in labels)

    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None

    return (observed - expected) / denominator


def weighted_cohen_kappa_linear(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> float | None:
    """Compute linearly weighted Cohen's kappa for ordinal labels.

    The protocol asks for weighted kappa on the ordinal overall label but does
    not specify linear versus quadratic weights. This implementation uses simple
    linear agreement weights because they are easy to inspect and explain.
    """
    if not pairs:
        return None

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = build_confusion_matrix(pairs, labels)
    n = len(pairs)
    k = len(labels)
    if k < 2:
        return None

    gold_counts = {
        gold_label: sum(matrix[gold_label][model_label] for model_label in labels)
        for gold_label in labels
    }
    model_counts = {
        model_label: sum(matrix[gold_label][model_label] for gold_label in labels)
        for model_label in labels
    }

    observed_weighted = 0.0
    expected_weighted = 0.0
    for gold_label in labels:
        for model_label in labels:
            weight = 1.0 - (
                abs(label_to_index[gold_label] - label_to_index[model_label]) / (k - 1)
            )
            observed_weighted += weight * matrix[gold_label][model_label] / n
            expected_weighted += weight * (gold_counts[gold_label] / n) * (model_counts[model_label] / n)

    denominator = 1.0 - expected_weighted
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None

    return (observed_weighted - expected_weighted) / denominator


def build_confusion_matrix(
    pairs: list[tuple[str, str]],
    labels: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Build a square confusion matrix as nested dicts.

    Rows are gold labels. Columns are model labels.
    """
    matrix = {
        gold_label: {model_label: 0 for model_label in labels}
        for gold_label in labels
    }
    for gold_label, model_label in pairs:
        matrix[gold_label][model_label] += 1
    return matrix


def build_majority_class_baseline(gold_by_study: dict[str, GoldStudy]) -> tuple[dict[str, str], MeanCI]:
    """Return majority-label predictions per criterion and the mean agreement.

    The baseline predicts, for each criterion, the most common gold label across
    studies. Ties are broken in the fixed order yes, no, unclear for determinism.
    """
    majority_prediction: dict[str, str] = {}
    for criterion_key in CRITERION_KEYS:
        counts = Counter(gold.criteria[criterion_key] for gold in gold_by_study.values())
        majority_prediction[criterion_key] = max(
            JUDGMENT_LABELS,
            key=lambda label: (counts[label], -JUDGMENT_LABELS.index(label)),
        )

    per_study_agreement = [
        compute_criterion_agreement(gold.criteria, majority_prediction)
        for gold in gold_by_study.values()
    ]
    return majority_prediction, mean_and_t_ci(per_study_agreement)


def build_default_contrasts(
    models: tuple[str, ...],
    conditions: tuple[str, ...],
) -> list[Contrast]:
    """Return the default paired contrasts for B/C/D criterion agreement.

    With two models, we compare:
    - within-model prompt changes: B vs C, C vs D, B vs D
    - between-model same-condition differences: weak vs strong within B, C, D

    Only contrasts whose conditions are present in the requested run are included.
    """
    available = {condition for condition in conditions if condition in CONDITIONS_WITH_CRITERIA}
    contrasts: list[Contrast] = []

    def add_if_available(name: str, left_model: str, left_condition: str, right_model: str, right_condition: str) -> None:
        if left_condition in available and right_condition in available:
            contrasts.append(Contrast(name, left_model, left_condition, right_model, right_condition))

    if not models:
        return contrasts

    for model in models:
        add_if_available(f"{model} B - {model} C", model, "B", model, "C")
        add_if_available(f"{model} C - {model} D", model, "C", model, "D")
        add_if_available(f"{model} B - {model} D", model, "B", model, "D")

    if len(models) >= 2:
        weak_model, strong_model = models[0], models[1]
        for condition in ("B", "C", "D"):
            add_if_available(
                f"{weak_model} {condition} - {strong_model} {condition}",
                weak_model,
                condition,
                strong_model,
                condition,
            )

    return contrasts


def collect_contrast_differences(
    *,
    gold_by_study: dict[str, GoldStudy],
    results_by_key: dict[tuple[str, str, str], ScoredResult],
    contrast: Contrast,
) -> list[float]:
    """Collect paired per-study criterion-agreement differences for one contrast."""
    differences: list[float] = []
    for study_id in gold_by_study:
        left = results_by_key.get((study_id, contrast.left_model, contrast.left_condition))
        right = results_by_key.get((study_id, contrast.right_model, contrast.right_condition))
        if left is None or right is None:
            continue
        if left.parse_failure or right.parse_failure:
            continue
        if left.criterion_agreement is None or right.criterion_agreement is None:
            continue
        differences.append(left.criterion_agreement - right.criterion_agreement)
    return differences


def write_scored_summary_csv(scored_results: list[ScoredResult], output_csv: Path) -> None:
    """Write the study-level summary CSV requested by the protocol."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for result in scored_results:
            writer.writerow(
                {
                    "study_id": result.study_id,
                    "model": result.model,
                    "condition": result.condition,
                    "criterion_agreement": _format_float(result.criterion_agreement),
                    "model_overall_rob": result.model_overall_rob or "",
                    "gold_overall_rob": result.gold_overall_rob,
                    "overall_correct": _format_bool(result.overall_correct),
                    "derived_overall_rob": result.derived_overall_rob or "",
                    "derived_overall_correct": _format_bool(result.derived_overall_correct),
                    "parse_failure": _format_bool(result.parse_failure),
                }
            )


def build_report(
    gold_by_study: dict[str, GoldStudy],
    scored_results: list[ScoredResult],
    models: tuple[str, ...],
    conditions: tuple[str, ...],
    contrasts: list[Contrast],
    raw_dir: Path,
) -> str:
    """Build the human-readable stdout report."""
    results_by_key = {(r.study_id, r.model, r.condition): r for r in scored_results}
    lines: list[str] = []

    lines.append("LLM RoB scoring report")
    lines.append("=" * 80)
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Studies in gold CSV: {len(gold_by_study)}")
    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Conditions: {', '.join(conditions)}")
    lines.append("")

    lines.append("Parse failures")
    lines.append("-" * 80)
    for model in models:
        for condition in conditions:
            group = [r for r in scored_results if r.model == model and r.condition == condition]
            failure_count = sum(1 for r in group if r.parse_failure)
            lines.append(f"{model} / {condition}: {failure_count}/{len(group)} parse failures")
    lines.append("")

    lines.append("Primary criterion-level analysis (Conditions B/C/D only)")
    lines.append("-" * 80)
    for model in models:
        for condition in conditions:
            if condition not in CONDITIONS_WITH_CRITERIA:
                continue
            group = [r for r in scored_results if r.model == model and r.condition == condition]
            valid_group = [r for r in group if not r.parse_failure]
            criterion_values = [r.criterion_agreement for r in valid_group if r.criterion_agreement is not None]
            criterion_summary = mean_and_t_ci(criterion_values)
            criterion_pairs = pooled_criterion_pairs(valid_group)
            criterion_kappa = unweighted_cohen_kappa(criterion_pairs, JUDGMENT_LABELS)

            overall_pairs = [
                (r.gold_overall_rob, r.model_overall_rob)
                for r in valid_group
                if r.model_overall_rob is not None
            ]
            reported_overall_agreement = percent_agreement(overall_pairs)
            reported_overall_kappa = weighted_cohen_kappa_linear(overall_pairs, OVERALL_LABELS)

            derived_pairs = [
                (r.gold_overall_rob, r.derived_overall_rob)
                for r in valid_group
                if r.derived_overall_rob is not None
            ]
            derived_overall_agreement = percent_agreement(derived_pairs)
            derived_overall_kappa = weighted_cohen_kappa_linear(derived_pairs, OVERALL_LABELS)

            lines.append(f"{model} / {condition}")
            lines.append(
                "  criterion agreement: "
                + format_mean_ci(criterion_summary)
            )
            lines.append(
                f"  criterion-level kappa: {format_metric(criterion_kappa)}"
            )
            lines.append(
                "  model-reported overall vs gold: "
                f"agreement={format_percent(reported_overall_agreement)}, "
                f"weighted kappa={format_metric(reported_overall_kappa)}"
            )
            lines.append(
                "  Python-derived overall vs gold: "
                f"agreement={format_percent(derived_overall_agreement)}, "
                f"weighted kappa={format_metric(derived_overall_kappa)}"
            )
    lines.append("")

    lines.append("Condition A overall-label baseline (descriptive only)")
    lines.append("-" * 80)
    for model in models:
        condition_a_overall_by_study = load_condition_a_raw_overall_rob(
            raw_dir=raw_dir,
            study_ids=tuple(gold_by_study.keys()),
            model=model,
        )
        overall_pairs = [
            (gold_by_study[study_id].overall_rob, condition_a_overall_by_study[study_id])
            for study_id in gold_by_study
            if study_id in condition_a_overall_by_study
        ]
        overall_agreement = percent_agreement(overall_pairs)
        overall_kappa = weighted_cohen_kappa_linear(overall_pairs, OVERALL_LABELS)
        lines.append(
            f"{model} / A: agreement={format_percent(overall_agreement)}, "
            f"weighted kappa={format_metric(overall_kappa)}, n={len(overall_pairs)}"
        )
    lines.append("")

    lines.append("Paired differences in per-study criterion agreement")
    lines.append("-" * 80)
    if not contrasts:
        lines.append("No paired contrasts available for the requested models/conditions.")
        contrast_differences: list[list[float]] = []
    else:
        contrast_differences = [
            collect_contrast_differences(
                gold_by_study=gold_by_study,
                results_by_key=results_by_key,
                contrast=contrast,
            )
            for contrast in contrasts
        ]
        for contrast, differences in zip(contrasts, contrast_differences):
            diff_summary = mean_and_t_ci(differences)
            lines.append(f"{contrast.name}: {format_mean_ci(diff_summary)}")
    lines.append("")

    lines.append("Holm sensitivity analysis (all contrasts)")
    lines.append("-" * 80)
    if not contrasts:
        lines.append("No paired contrasts available for the requested models/conditions.")
    else:
        raw_p_values = [paired_t_p_value(differences) for differences in contrast_differences]
        holm_p_values = holm_adjust(raw_p_values)
        family_size = sum(1 for p_value in raw_p_values if p_value is not None)
        lines.append(f"Family size (k={family_size})")
        for contrast, differences, raw_p_value, holm_p_value in zip(
            contrasts, contrast_differences, raw_p_values, holm_p_values
        ):
            diff_summary = mean_and_t_ci(differences)
            lines.append(
                f"{contrast.name}: mean={format_signed_metric(diff_summary.mean)}, "
                f"95% CI {format_short_ci(diff_summary)}, "
                f"p={format_metric(raw_p_value)}, "
                f"Holm p={format_metric(holm_p_value)}, "
                f"survives={format_survival_flag(holm_p_value)}"
            )
    lines.append("")

    majority_prediction, majority_summary = build_majority_class_baseline(gold_by_study)
    lines.append("Majority-class baseline")
    lines.append("-" * 80)
    lines.append(
        "Per-criterion majority label: "
        + ", ".join(f"{key}={majority_prediction[key]}" for key in CRITERION_KEYS)
    )
    lines.append(f"Criterion agreement if always predicting those labels: {format_mean_ci(majority_summary)}")
    lines.append("")

    lines.append("Criterion-level confusion matrices (gold rows × model columns)")
    lines.append("-" * 80)
    for model in models:
        for condition in conditions:
            if condition not in CONDITIONS_WITH_CRITERIA:
                continue
            group = [r for r in scored_results if r.model == model and r.condition == condition]
            valid_group = [r for r in group if not r.parse_failure]
            lines.append(f"{model} / {condition} (valid studies={len(valid_group)}, parse failures={len(group) - len(valid_group)})")
            for criterion_key in CRITERION_KEYS:
                pairs = [
                    (r.gold_criteria[criterion_key], r.model_criteria[criterion_key])
                    for r in valid_group
                    if r.model_criteria is not None
                ]
                matrix = build_confusion_matrix(pairs, JUDGMENT_LABELS)
                lines.append(f"  {criterion_key}")
                lines.extend(_format_confusion_matrix_lines(matrix, labels=JUDGMENT_LABELS, indent="    "))
            lines.append("")

    warnings = [r for r in scored_results if r.parse_failure and r.error]
    if warnings:
        lines.append("Parse-failure details")
        lines.append("-" * 80)
        for result in warnings:
            lines.append(
                f"{result.study_id} / {result.model} / {result.condition}: {result.error}"
            )

    return "\n".join(lines)


def pooled_criterion_pairs(valid_results: list[ScoredResult]) -> list[tuple[str, str]]:
    """Pool all valid gold/model criterion pairs across studies."""
    pairs: list[tuple[str, str]] = []
    for result in valid_results:
        if result.model_criteria is None:
            continue
        for key in CRITERION_KEYS:
            pairs.append((result.gold_criteria[key], result.model_criteria[key]))
    return pairs


def percent_agreement(pairs: list[tuple[str, str]]) -> float | None:
    """Return exact agreement as a proportion in [0, 1]."""
    if not pairs:
        return None
    agreement_count = sum(1 for gold, model in pairs if gold == model)
    return agreement_count / len(pairs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LLM RoB outputs against gold labels.")
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=DEFAULT_GOLD_CSV,
        help=f"Path to the gold-label CSV (default: {DEFAULT_GOLD_CSV})",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help=f"Directory containing validated JSON outputs (default: {DEFAULT_PARSED_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path to write scored_summary.csv (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model IDs to score. Missing files for these models count as parse failures.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Conditions to score (default: A B C D).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = tuple(args.models)
    conditions = tuple(_normalize_condition(condition) for condition in args.conditions)

    gold_by_study = load_gold_labels(args.gold_csv)
    scored_results = score_results(
        gold_by_study=gold_by_study,
        parsed_dir=args.parsed_dir,
        models=models,
        conditions=conditions,
    )
    write_scored_summary_csv(scored_results, args.output_csv)

    raw_dir = args.parsed_dir.parent / "raw"
    report = build_report(
        gold_by_study=gold_by_study,
        scored_results=scored_results,
        models=models,
        conditions=conditions,
        contrasts=build_default_contrasts(models, conditions),
        raw_dir=raw_dir,
    )
    print(report)
    return 0


def _normalize_choice(value: Any, allowed: tuple[str, ...], field_name: str) -> str:
    if not isinstance(value, str):
        raise ScoreResultsError(f"{field_name} must be a string; got {type(value).__name__}")

    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(allowed)
        raise ScoreResultsError(
            f"{field_name} must be one of {allowed_text}; got {value!r}"
        )
    return normalized


def _normalize_condition(condition: str) -> str:
    normalized = condition.strip().upper()
    if normalized not in {"A", "B", "C", "D"}:
        raise ScoreResultsError(f"condition must be one of A, B, C, D; got {condition!r}")
    return normalized


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    return 1.96


def format_mean_ci(summary: MeanCI) -> str:
    if summary.mean is None:
        return "n=0, mean=NA, 95% CI=NA"
    if summary.lower is None or summary.upper is None:
        return f"n={summary.n}, mean={summary.mean:.3f}, 95% CI=NA"
    return (
        f"n={summary.n}, mean={summary.mean:.3f}, "
        f"95% CI [{summary.lower:.3f}, {summary.upper:.3f}]"
    )


def format_metric(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.3f}"


def format_percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value * 100:.1f}%"


def format_signed_metric(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:+.3f}"


def format_short_ci(summary: MeanCI) -> str:
    if summary.mean is None or summary.lower is None or summary.upper is None:
        return "[NA, NA]"
    return f"[{summary.lower:+.3f}, {summary.upper:+.3f}]"


def format_survival_flag(adjusted_p_value: float | None) -> str:
    if adjusted_p_value is None or math.isnan(adjusted_p_value):
        return "NA"
    return "Yes" if adjusted_p_value < 0.05 else "No"


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _format_confusion_matrix_lines(
    matrix: dict[str, dict[str, int]],
    labels: tuple[str, ...],
    indent: str,
) -> list[str]:
    header = indent + "gold\\model" + " | " + " | ".join(f"{label:>7}" for label in labels)
    separator = indent + "-" * max(40, len(header) - len(indent))
    lines = [header, separator]
    for gold_label in labels:
        row = " | ".join(f"{matrix[gold_label][model_label]:>7d}" for model_label in labels)
        lines.append(indent + f"{gold_label:>10} | {row}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
