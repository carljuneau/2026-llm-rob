"""Run LLM risk-of-bias assessments with the Gemini API.

This script:
- loads study IDs from the public gold-label CSV
- loads criterion definitions from the public criteria CSV
- finds the target study PDFs in ``data/private/observational/``
- loads committed prompt template files from ``prompts/``
- assembles prompt conditions cumulatively from those template files
- sends requests to Gemini models
- saves raw response text to ``results/raw/``
- parses and validates JSON outputs with ``schema.validate``
- saves validated JSON to ``results/parsed/``
- retries once on parse/validation failure with the identical prompt
- records unrecovered failures in ``results/parse_failures.csv``

The substantive prompt text lives in:
- ``prompts/condition_a.txt``
- ``prompts/condition_b.txt``
- ``prompts/condition_c.txt``
- ``prompts/condition_d.txt``

Condition assembly is cumulative:
- A: [study_pdf, A]
- B: [study_pdf, A + "\\n\\n" + B]
- C: [study_pdf, A + "\\n\\n" + B, C, training_material]
- D: [study_pdf, A + "\\n\\n" + B, C, training_material, D, example_input, example_output]

It uses the official Google GenAI SDK and the Python standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - depends on local environment
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from schema import CRITERION_KEYS, CONDITIONS_WITH_CRITERIA, SchemaValidationError, validate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DIR = DATA_DIR / "public"
PRIVATE_DIR = DATA_DIR / "private"
OBSERVATIONAL_PDF_DIR = PRIVATE_DIR / "observational"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PARSED_DIR = RESULTS_DIR / "parsed"
DEBUG_DIR = RESULTS_DIR / "debug"
PARSE_FAILURES_CSV = RESULTS_DIR / "parse_failures.csv"

GOLD_CSV_PATH = PUBLIC_DIR / "Table 1 - RoB_observational_studies.csv"
CRITERIA_CSV_PATH = PUBLIC_DIR / "Table 2 - RoB_criteria.csv"

DEFAULT_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
)
DEFAULT_CONDITIONS = ("A", "B", "C", "D")

TEMPERATURE = 0
MAX_TOKENS = 32768

PROMPT_TEMPLATE_FILENAMES = {
    "condition_a.txt",
    "condition_b.txt",
    "condition_c.txt",
    "condition_d.txt",
}

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}
SUPPORTED_PROMPT_MATERIAL_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}

GOLD_STUDY_ID_HEADER = "Study first author"
CRITERIA_YES_CONDITION_HEADER = "Criterion (yes condition)"
CRITERIA_CODE_KEY_HEADER = "Code key"

class RunModelsError(ValueError):
    """Raised for bad inputs or inconsistent run setup."""


@dataclass(frozen=True)
class StudyInput:
    study_id: str
    pdf_path: Path


@dataclass(frozen=True)
class CriterionDefinition:
    code_key: str
    yes_condition: str


@dataclass(frozen=True)
class PromptMaterial:
    kind: str  # "pdf" or "text"
    path: Path


@dataclass(frozen=True)
class WorkedExample:
    input_material: PromptMaterial
    output_json_text: str
    output_json_path: Path


@dataclass
class PromptAssetResolver:
    prompts_dir: Path
    _condition_c_material: PromptMaterial | None = None
    _condition_c_checked: bool = False
    _condition_d_example: WorkedExample | None = None
    _condition_d_checked: bool = False

    def get_condition_c_material(self) -> PromptMaterial:
        if not self._condition_c_checked:
            self._condition_c_material = _discover_condition_c_material(self.prompts_dir)
            self._condition_c_checked = True
        if self._condition_c_material is None:
            raise RunModelsError(
                "Condition C requires training material in prompts/. "
                "Add a PDF or text file for the Mulder reference material."
            )
        return self._condition_c_material

    def get_condition_d_example(self) -> WorkedExample:
        if not self._condition_d_checked:
            self._condition_d_example = _discover_condition_d_example(self.prompts_dir)
            self._condition_d_checked = True
        if self._condition_d_example is None:
            raise RunModelsError(
                "Condition D requires a worked example in prompts/examples/ "
                "or another unambiguous example file pair under prompts/."
            )
        return self._condition_d_example


@dataclass
class RunCounters:
    successful: int = 0
    skipped_existing: int = 0
    failures: int = 0
    dry_run_printed: int = 0


def load_study_ids_from_gold_csv(csv_path: Path) -> list[str]:
    """Load study IDs from the public gold-label CSV.

    The last row is a note row starting with ``Note:`` and is skipped.
    Order is preserved from the CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {csv_path}")

    study_ids: list[str] = []
    seen: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if GOLD_STUDY_ID_HEADER not in (reader.fieldnames or []):
            raise RunModelsError(f"Gold CSV is missing required column: {GOLD_STUDY_ID_HEADER!r}")

        for row_number, row in enumerate(reader, start=2):
            raw_study_id = (row.get(GOLD_STUDY_ID_HEADER) or "").strip()
            if not raw_study_id:
                continue
            if raw_study_id.startswith("Note:"):
                continue
            if raw_study_id.startswith("Table "):
                continue
            if raw_study_id in seen:
                raise RunModelsError(f"Duplicate study_id in gold CSV on row {row_number}: {raw_study_id!r}")
            seen.add(raw_study_id)
            study_ids.append(raw_study_id)

    if not study_ids:
        raise RunModelsError(f"No study IDs were loaded from {csv_path}")

    return study_ids


def load_studies(gold_csv_path: Path, observational_pdf_dir: Path, selected_studies: list[str] | None) -> list[StudyInput]:
    """Build the ordered study list and map each study ID to its local PDF."""
    ordered_study_ids = load_study_ids_from_gold_csv(gold_csv_path)
    available_study_ids = set(ordered_study_ids)

    if selected_studies:
        unknown = [study_id for study_id in selected_studies if study_id not in available_study_ids]
        if unknown:
            raise RunModelsError(
                "Unknown study_id(s) in --studies: " + ", ".join(sorted(unknown))
            )
        selected_set = set(selected_studies)
        ordered_study_ids = [study_id for study_id in ordered_study_ids if study_id in selected_set]

    if not observational_pdf_dir.exists():
        raise FileNotFoundError(f"Study PDF directory not found: {observational_pdf_dir}")

    studies: list[StudyInput] = []
    for study_id in ordered_study_ids:
        pdf_path = observational_pdf_dir / f"{study_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing study PDF for {study_id!r}: {pdf_path}")
        studies.append(StudyInput(study_id=study_id, pdf_path=pdf_path))

    if not studies:
        raise RunModelsError("No studies selected for the run")

    return studies


def load_criterion_definitions(csv_path: Path) -> list[CriterionDefinition]:
    """Load criterion definitions from the public criteria CSV.

    The output order matches ``CRITERION_KEYS`` exactly.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Criteria CSV not found: {csv_path}")

    by_code_key: dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_headers = {CRITERIA_YES_CONDITION_HEADER, CRITERIA_CODE_KEY_HEADER}
        missing_headers = [header for header in required_headers if header not in (reader.fieldnames or [])]
        if missing_headers:
            raise RunModelsError(
                "Criteria CSV is missing required column(s): " + ", ".join(sorted(missing_headers))
            )

        for row_number, row in enumerate(reader, start=2):
            code_key = (row.get(CRITERIA_CODE_KEY_HEADER) or "").strip()
            yes_condition = (row.get(CRITERIA_YES_CONDITION_HEADER) or "").strip()
            if not code_key:
                continue
            if code_key in by_code_key:
                raise RunModelsError(f"Duplicate code key in criteria CSV on row {row_number}: {code_key!r}")
            by_code_key[code_key] = yes_condition

    missing_keys = [key for key in CRITERION_KEYS if key not in by_code_key]
    unexpected_keys = sorted(set(by_code_key) - set(CRITERION_KEYS))
    if missing_keys:
        raise RunModelsError(
            "Criteria CSV is missing required code key(s): " + ", ".join(missing_keys)
        )
    if unexpected_keys:
        raise RunModelsError(
            "Criteria CSV has unexpected code key(s): " + ", ".join(unexpected_keys)
        )

    definitions = [
        CriterionDefinition(code_key=key, yes_condition=by_code_key[key])
        for key in CRITERION_KEYS
    ]
    return definitions


def build_condition_bcd_schema_example(study_id: str) -> dict[str, Any]:
    """Build the JSON output template shown in B/C/D prompts."""
    criteria = {
        key: {"judgment": "yes|no|unclear", "quote": "verbatim supporting quote"}
        for key in CRITERION_KEYS
    }
    return {
        "study_id": study_id,
        "criteria": criteria,
        "overall_rob": "low|moderate|serious",
    }


def build_criteria_block(definitions: list[CriterionDefinition]) -> str:
    lines = []
    for index, definition in enumerate(definitions, start=1):
        lines.append(f"{index}. {definition.code_key}: {definition.yes_condition}")
    return "\n".join(lines)


def _load_prompt_text(filename: str, prompts_dir: Path) -> str:
    """Load one committed prompt template from ``prompts/``."""
    path = prompts_dir / filename
    if not path.exists():
        raise RunModelsError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_request_content(
    *,
    study: StudyInput,
    condition: str,
    criteria_definitions: list[CriterionDefinition],
    prompt_assets: PromptAssetResolver,
) -> list[dict[str, Any]]:
    """Build the Gemini request content blocks for one study × condition."""
    normalized_condition = condition.strip().upper()
    prompts_dir = prompt_assets.prompts_dir

    prompt_a = _load_prompt_text("condition_a.txt", prompts_dir).format(study_id=study.study_id)
    study_block = _pdf_document_block(study.pdf_path)

    if normalized_condition == "A":
        return [study_block, _text_block(prompt_a)]

    criteria_block = build_criteria_block(criteria_definitions)
    schema_text = json.dumps(build_condition_bcd_schema_example(study.study_id), indent=2, ensure_ascii=False)
    prompt_b = _load_prompt_text("condition_b.txt", prompts_dir).format(
        criteria_block=criteria_block,
        schema=schema_text,
    )
    prompt_ab = prompt_a + "\n\n" + prompt_b

    if normalized_condition == "B":
        return [study_block, _text_block(prompt_ab)]

    prompt_c = _load_prompt_text("condition_c.txt", prompts_dir)
    training_material = prompt_assets.get_condition_c_material()

    if normalized_condition == "C":
        return [
            study_block,
            _text_block(prompt_ab),
            _text_block(prompt_c),
            *_material_to_content_blocks(training_material),
        ]

    if normalized_condition == "D":
        prompt_d = _load_prompt_text("condition_d.txt", prompts_dir)
        worked_example = prompt_assets.get_condition_d_example()
        return [
            study_block,
            _text_block(prompt_ab),
            _text_block(prompt_c),
            *_material_to_content_blocks(training_material),
            _text_block(prompt_d),
            *_material_to_content_blocks(worked_example.input_material),
            _text_block("Worked example expected JSON output:\n" + worked_example.output_json_text),
        ]

    raise RunModelsError(f"Unsupported condition: {condition!r}")


def extract_response_text(response: Any) -> str | None:
    """Extract text from a Gemini response object."""
    try:
        response_text = getattr(response, "text", None)
    except Exception:  # pragma: no cover - defensive around SDK properties
        response_text = None

    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list):
        return None

    text_parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None)
        if not isinstance(parts, list):
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                text_parts.append(part_text)

    if not text_parts:
        return None

    return "\n".join(text_parts).strip()


def serialize_response_for_debug(response: Any) -> str:
    """Serialize the full SDK response when no text is directly available."""
    if hasattr(response, "model_dump_json"):
        try:
            return response.model_dump_json(indent=2)
        except TypeError:
            return response.model_dump_json()
    if hasattr(response, "model_dump"):
        try:
            return json.dumps(response.model_dump(), indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return json.dumps(response.model_dump(), ensure_ascii=False, default=str)
    if hasattr(response, "to_json_dict"):
        return json.dumps(response.to_json_dict(), indent=2, ensure_ascii=False, default=str)
    return repr(response)


def _extract_response_diagnostics(response: Any, assistant_text: str | None) -> dict[str, Any]:
    """Extract Gemini response metadata useful for diagnosing stop reasons."""
    diag: dict[str, Any] = {}

    # Top-level metadata
    diag["response_id"] = getattr(response, "response_id", None)
    diag["model_version"] = getattr(response, "model_version", None)

    # Usage metadata
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        try:
            diag["usage_metadata"] = usage.model_dump() if hasattr(usage, "model_dump") else repr(usage)
        except Exception:
            diag["usage_metadata"] = repr(usage)

    # Prompt feedback (blocking info)
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        try:
            diag["prompt_feedback"] = prompt_feedback.model_dump() if hasattr(prompt_feedback, "model_dump") else repr(prompt_feedback)
        except Exception:
            diag["prompt_feedback"] = repr(prompt_feedback)

    # Per-candidate stop info
    candidates = getattr(response, "candidates", None) or []
    diag["candidates"] = []
    for candidate in candidates:
        c: dict[str, Any] = {}
        c["finish_reason"] = str(getattr(candidate, "finish_reason", None))
        c["finish_message"] = getattr(candidate, "finish_message", None)
        safety_ratings = getattr(candidate, "safety_ratings", None)
        if safety_ratings is not None:
            try:
                c["safety_ratings"] = [r.model_dump() if hasattr(r, "model_dump") else repr(r) for r in safety_ratings]
            except Exception:
                c["safety_ratings"] = repr(safety_ratings)
        citation_metadata = getattr(candidate, "citation_metadata", None)
        if citation_metadata is not None:
            try:
                c["citation_metadata"] = citation_metadata.model_dump() if hasattr(citation_metadata, "model_dump") else repr(citation_metadata)
            except Exception:
                c["citation_metadata"] = repr(citation_metadata)
        diag["candidates"].append(c)

    # Assistant text summary
    if assistant_text:
        diag["assistant_text_length"] = len(assistant_text)
        diag["assistant_text_head"] = assistant_text[:200]
        diag["assistant_text_tail"] = assistant_text[-200:]
    else:
        diag["assistant_text_length"] = 0
        diag["assistant_text_head"] = None
        diag["assistant_text_tail"] = None

    return diag


def make_debug_output_path(debug_dir: Path, study_id: str, model: str, condition: str, attempt: int) -> Path:
    if attempt == 1:
        filename = f"{study_id}_{model}_{condition}_debug.json"
    else:
        filename = f"{study_id}_{model}_{condition}_attempt{attempt}_debug.json"
    return debug_dir / filename


def write_response_debug(
    *,
    debug_dir: Path,
    study_id: str,
    model: str,
    condition: str,
    attempt: int,
    response: Any,
    assistant_text: str | None,
    error: str,
) -> None:
    """Write a debug JSON file capturing Gemini response metadata for a failed attempt."""
    diag = _extract_response_diagnostics(response, assistant_text)
    diag["error"] = error
    path = make_debug_output_path(debug_dir, study_id, model, condition, attempt)
    write_json(path, diag)
    finish_reasons = [c.get("finish_reason") for c in diag.get("candidates", [])]
    print(f"[diag] {study_id} | {model} | {condition} | attempt {attempt} | finish_reason={finish_reasons} | chars={diag['assistant_text_length']}")


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def validate_model_output(parsed_payload: Any, condition: str, expected_study_id: str) -> dict[str, Any]:
    """Validate the parsed JSON and enforce the expected study ID."""
    validated = validate(parsed_payload, condition)
    actual_study_id = validated["study_id"]
    if actual_study_id != expected_study_id:
        raise SchemaValidationError(
            f"output.study_id must equal {expected_study_id!r}; got {actual_study_id!r}"
        )
    return dict(validated)


def run_one_combination(
    *,
    client: Any,
    study: StudyInput,
    model: str,
    condition: str,
    criteria_definitions: list[CriterionDefinition],
    prompt_assets: PromptAssetResolver,
    raw_dir: Path,
    parsed_dir: Path,
    debug_dir: Path,
    parse_failures_csv: Path,
    dry_run: bool,
) -> str:
    """Run one study × model × condition combination.

    Returns one of: ``success``, ``skipped_existing``, ``failure``, ``dry_run_printed``.
    """
    parsed_path = parsed_dir / f"{study.study_id}_{model}_{condition}.json"

    if parsed_path.exists() and not dry_run:
        print(f"[skip] Parsed result already exists: {parsed_path.name}")
        return "skipped_existing"

    content_blocks = build_request_content(
        study=study,
        condition=condition,
        criteria_definitions=criteria_definitions,
        prompt_assets=prompt_assets,
    )

    if dry_run:
        print_request_preview(study=study, model=model, condition=condition, content_blocks=content_blocks)
        return "dry_run_printed"

    errors: list[tuple[int, str]] = []

    for attempt in (1, 2):
        try:
            response = _call_model_with_backoff(client=client, model=model, content_blocks=content_blocks)
        except Exception as exc:  # pragma: no cover - depends on SDK/runtime behavior
            error_msg = f"API call failed: {exc}"
            errors.append((attempt, error_msg))
            break

        assistant_text = extract_response_text(response)
        raw_text_to_save = assistant_text if assistant_text is not None else serialize_response_for_debug(response)
        raw_path = make_raw_output_path(raw_dir, study.study_id, model, condition, attempt)
        write_text(raw_path, raw_text_to_save)

        if assistant_text is None or not assistant_text.strip():
            error_msg = "Response did not contain any assistant text blocks"
            errors.append((attempt, error_msg))
            write_response_debug(debug_dir=debug_dir, study_id=study.study_id, model=model, condition=condition, attempt=attempt, response=response, assistant_text=assistant_text, error=error_msg)
            if attempt == 1:
                continue
            break

        try:
            parsed_payload = json.loads(_strip_markdown_fences(assistant_text))
        except json.JSONDecodeError as exc:
            error_msg = f"Invalid JSON: {exc}"
            errors.append((attempt, error_msg))
            write_response_debug(debug_dir=debug_dir, study_id=study.study_id, model=model, condition=condition, attempt=attempt, response=response, assistant_text=assistant_text, error=error_msg)
            if attempt == 1:
                continue
            break

        try:
            validated_output = validate_model_output(
                parsed_payload=parsed_payload,
                condition=condition,
                expected_study_id=study.study_id,
            )
        except SchemaValidationError as exc:
            error_msg = f"Schema validation failed: {exc}"
            errors.append((attempt, error_msg))
            write_response_debug(debug_dir=debug_dir, study_id=study.study_id, model=model, condition=condition, attempt=attempt, response=response, assistant_text=assistant_text, error=error_msg)
            if attempt == 1:
                continue
            break

        write_json(parsed_path, validated_output)
        print(f"[ok] Saved parsed result: {parsed_path.name}")
        return "success"

    append_parse_failures(
        csv_path=parse_failures_csv,
        study_id=study.study_id,
        model=model,
        condition=condition,
        failures=errors,
    )
    print(f"[fail] {study.study_id} | {model} | {condition}")
    return "failure"


def run_pipeline(
    *,
    models: tuple[str, ...],
    conditions: tuple[str, ...],
    selected_studies: list[str] | None,
    dry_run: bool,
) -> RunCounters:
    """Run the requested grid of study × model × condition combinations."""
    studies = load_studies(
        gold_csv_path=GOLD_CSV_PATH,
        observational_pdf_dir=OBSERVATIONAL_PDF_DIR,
        selected_studies=selected_studies,
    )
    criteria_definitions = load_criterion_definitions(CRITERIA_CSV_PATH)
    prompt_assets = PromptAssetResolver(prompts_dir=PROMPTS_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    client = None if dry_run else create_gemini_client()
    counters = RunCounters()

    for study in studies:
        for model in models:
            for condition in conditions:
                status = run_one_combination(
                    client=client,
                    study=study,
                    model=model,
                    condition=condition,
                    criteria_definitions=criteria_definitions,
                    prompt_assets=prompt_assets,
                    raw_dir=RAW_DIR,
                    parsed_dir=PARSED_DIR,
                    debug_dir=DEBUG_DIR,
                    parse_failures_csv=PARSE_FAILURES_CSV,
                    dry_run=dry_run,
                )
                if status == "success":
                    counters.successful += 1
                elif status == "skipped_existing":
                    counters.skipped_existing += 1
                elif status == "failure":
                    counters.failures += 1
                elif status == "dry_run_printed":
                    counters.dry_run_printed += 1
                else:  # pragma: no cover - defensive guard
                    raise RunModelsError(f"Unexpected run status: {status!r}")
                if status in {"success", "failure"}:
                    time.sleep(13)

    return counters


def create_gemini_client() -> Any:
    """Create the Gemini SDK client.

    The SDK reads ``GEMINI_API_KEY`` from the environment.
    """
    if genai is None:
        raise RunModelsError(
            "The google-genai package is not installed. Install it with: pip install google-genai"
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RunModelsError("GEMINI_API_KEY is not set in the environment")
    return genai.Client(api_key=api_key)


def make_raw_output_path(raw_dir: Path, study_id: str, model: str, condition: str, attempt: int) -> Path:
    """Build a raw-output path.

    Attempt 1 keeps the simple base name requested in the protocol.
    Attempt 2 gets an explicit suffix so both attempts are preserved.
    """
    if attempt == 1:
        filename = f"{study_id}_{model}_{condition}_raw.txt"
    else:
        filename = f"{study_id}_{model}_{condition}_attempt{attempt}_raw.txt"
    return raw_dir / filename


def append_parse_failures(
    *,
    csv_path: Path,
    study_id: str,
    model: str,
    condition: str,
    failures: list[tuple[int, str]],
) -> None:
    """Append unrecovered failures to ``parse_failures.csv``.

    We only write rows if the combination failed overall. If attempt 2 succeeds,
    no row is written.
    """
    if not failures:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["study_id", "model", "condition", "attempt", "error"],
        )
        if not file_exists:
            writer.writeheader()
        for attempt, error in failures:
            writer.writerow(
                {
                    "study_id": study_id,
                    "model": model,
                    "condition": condition,
                    "attempt": attempt,
                    "error": error,
                }
            )


def print_request_preview(
    *,
    study: StudyInput,
    model: str,
    condition: str,
    content_blocks: list[dict[str, Any]],
) -> None:
    """Print a human-readable preview of the request content blocks."""
    divider = "=" * 80
    print(divider)
    print(f"DRY RUN | study={study.study_id} | model={model} | condition={condition}")
    print(divider)

    for index, block in enumerate(content_blocks, start=1):
        if "text" in block and "inline_data" not in block:
            print(f"[block {index}] type=text")
            text = str(block.get("text", ""))
            print(_truncate_for_preview(text))
        elif "inline_data" in block:
            print(f"[block {index}] type=document")
            inline_data = block.get("inline_data", {})
            path_hint = block.get("_local_path", "(path not recorded)")
            media_type = inline_data.get("mime_type") if isinstance(inline_data, dict) else None
            print(f"local_path: {path_hint}")
            print(f"media_type: {media_type}")
        else:
            print(f"[block {index}] type=unknown")
            print(repr(block))
        print("-" * 80)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pdf_document_block(pdf_path: Path) -> dict[str, Any]:
    return {
        "inline_data": {
            "mime_type": "application/pdf",
            "data": pdf_path.read_bytes(),
        },
        # Local-only debug hint. This is stripped before the API call.
        "_local_path": str(pdf_path),
    }


def _text_block(text: str) -> dict[str, Any]:
    return {"text": text}


def _material_to_content_blocks(material: PromptMaterial) -> list[dict[str, Any]]:
    if material.kind == "pdf":
        return [_pdf_document_block(material.path)]
    if material.kind == "text":
        text = material.path.read_text(encoding="utf-8")
        return [_text_block(text)]
    raise RunModelsError(f"Unsupported prompt material kind: {material.kind!r}")


def _discover_condition_c_material(prompts_dir: Path) -> PromptMaterial | None:
    """Find the condition C training material at runtime.

    The file was not guaranteed in the uploaded tree, so we detect it lazily.
    We avoid assuming an exact filename.
    """
    if not prompts_dir.exists():
        return None

    candidates = []
    for path in prompts_dir.rglob("*"):
        if not path.is_file():
            continue
        if "examples" in {part.lower() for part in path.parts}:
            continue
        if path.suffix.lower() not in SUPPORTED_PROMPT_MATERIAL_EXTENSIONS:
            continue
        lowered_name = path.name.lower()
        if path.name in PROMPT_TEMPLATE_FILENAMES:
            continue
        if any(token in lowered_name for token in ("prompt", "template", "schema", "example")):
            continue
        candidates.append(path)

    if not candidates:
        return None

    chosen_path = _pick_best_path(
        candidates,
        scorer=_score_condition_c_candidate,
        description="condition C training material",
    )
    return _material_from_path(chosen_path)


def _discover_condition_d_example(prompts_dir: Path) -> WorkedExample | None:
    """Find the condition D worked example input/output pair."""
    if not prompts_dir.exists():
        return None

    examples_dir = prompts_dir / "examples"
    search_root = examples_dir if examples_dir.exists() else prompts_dir

    input_candidates: list[Path] = []
    output_candidates: list[Path] = []

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in PROMPT_TEMPLATE_FILENAMES:
            continue
        lowered_name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix == ".json":
            output_candidates.append(path)
            continue
        if suffix in SUPPORTED_PROMPT_MATERIAL_EXTENSIONS and "prompt" not in lowered_name and "template" not in lowered_name:
            input_candidates.append(path)

    if not examples_dir.exists():
        input_candidates = [path for path in input_candidates if _score_condition_d_candidate(path) > 0]
        output_candidates = [path for path in output_candidates if _score_condition_d_candidate(path) > 0]

    if not input_candidates or not output_candidates:
        return None

    chosen_input = _pick_best_path(
        input_candidates,
        scorer=_score_condition_d_candidate,
        description="condition D example input",
    )
    chosen_output = _pick_best_path(
        output_candidates,
        scorer=_score_condition_d_candidate,
        description="condition D example output JSON",
    )

    with chosen_output.open("r", encoding="utf-8") as handle:
        example_payload = json.load(handle)
    validated_example = validate(example_payload, "D")
    output_json_text = json.dumps(validated_example, indent=2, ensure_ascii=False)

    return WorkedExample(
        input_material=_material_from_path(chosen_input),
        output_json_text=output_json_text,
        output_json_path=chosen_output,
    )


def _material_from_path(path: Path) -> PromptMaterial:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PromptMaterial(kind="pdf", path=path)
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return PromptMaterial(kind="text", path=path)
    raise RunModelsError(f"Unsupported prompt material file type: {path}")


def _pick_best_path(candidates: list[Path], scorer: Any, description: str) -> Path:
    if not candidates:
        raise RunModelsError(f"No candidates available for {description}")
    if len(candidates) == 1:
        return candidates[0]

    scored_candidates = [(scorer(path), path) for path in candidates]
    scored_candidates.sort(key=lambda item: (item[0], item[1].name.lower()), reverse=True)

    top_score = scored_candidates[0][0]
    top_paths = [path for score, path in scored_candidates if score == top_score]

    if top_score > 0 and len(top_paths) == 1:
        return top_paths[0]

    candidate_text = ", ".join(path.name for path in candidates)
    raise RunModelsError(
        f"Could not choose an unambiguous {description}. Candidates: {candidate_text}"
    )


def _score_condition_c_candidate(path: Path) -> int:
    lowered = path.name.lower()
    score = 0
    if "mulder" in lowered:
        score += 10
    if "higgins" in lowered:
        score += 8
    if "chapter" in lowered:
        score += 6
    if "training" in lowered:
        score += 4
    if path.suffix.lower() == ".pdf":
        score += 2
    return score


def _score_condition_d_candidate(path: Path) -> int:
    lowered = path.name.lower()
    score = 0
    if "green" in lowered and "2019" in lowered:
        score += 10
    if "green" in lowered:
        score += 6
    if "worked" in lowered:
        score += 4
    if "example" in lowered:
        score += 4
    if path.suffix.lower() == ".pdf":
        score += 2
    return score


def _truncate_for_preview(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated for dry-run preview]"


def _sanitize_content_blocks_for_api(content_blocks: list[dict[str, Any]]) -> list[Any]:
    if types is None:
        raise RunModelsError(
            "The google-genai package is not installed. Install it with: pip install google-genai"
        )

    sanitized_blocks: list[Any] = []
    for block in content_blocks:
        if "text" in block and "inline_data" not in block:
            sanitized_blocks.append(str(block["text"]))
            continue

        if "inline_data" in block:
            inline_data = block["inline_data"]
            if not isinstance(inline_data, dict):
                raise RunModelsError("Document content block must contain an inline_data mapping")
            data = inline_data.get("data")
            mime_type = inline_data.get("mime_type")
            if not isinstance(data, (bytes, bytearray)):
                raise RunModelsError("Document inline_data.data must be raw bytes")
            if not isinstance(mime_type, str) or not mime_type:
                raise RunModelsError("Document inline_data.mime_type must be a non-empty string")
            sanitized_blocks.append(
                types.Part.from_bytes(
                    data=bytes(data),
                    mime_type=mime_type,
                )
            )
            continue

        raise RunModelsError(f"Unsupported content block for Gemini API: {block!r}")

    return sanitized_blocks


def _backoff_seconds(retry_index: int) -> float:
    """Capped exponential backoff with jitter: 15s, 30s, 60s, 120s."""
    base = 15 * (2 ** retry_index)
    return min(base, 120) + random.uniform(0, 3)


def _call_model_with_backoff(client: Any, model: str, content_blocks: list[dict[str, Any]]) -> Any:
    """Call the model with exponential backoff on retryable transport errors (429, 500, 503, 504)."""
    try:
        from google.genai import errors as genai_errors
        retryable_codes = {429, 500, 503, 504}
    except ImportError:
        genai_errors = None  # type: ignore[assignment]
        retryable_codes = set()

    max_transport_attempts = 4
    for transport_attempt in range(1, max_transport_attempts + 1):
        try:
            return call_model(client=client, model=model, content_blocks=content_blocks)
        except Exception as exc:
            is_retryable = False
            if genai_errors is not None and isinstance(exc, genai_errors.APIError):
                is_retryable = getattr(exc, "code", None) in retryable_codes
            elif any(str(code) in str(exc) for code in retryable_codes):
                is_retryable = True

            if is_retryable and transport_attempt < max_transport_attempts:
                sleep_s = _backoff_seconds(transport_attempt - 1)
                print(f"[retry] API error on transport attempt {transport_attempt} (sleeping {sleep_s:.1f}s): {exc}")
                time.sleep(sleep_s)
                continue
            raise


def call_model(client: Any, model: str, content_blocks: list[dict[str, Any]]) -> Any:
    config: dict[str, Any] = {
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_TOKENS,
    }
    if model == "gemini-3.1-pro-preview":
        config["thinking_config"] = {"thinking_level": "high"}

    return client.models.generate_content(
        model=model,
        contents=_sanitize_content_blocks_for_api(content_blocks),
        config=config,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model IDs to run. Default: both protocol models.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Prompt conditions to run. Default: A B C D.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional subset of study IDs. Default: all studies from the gold CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print request content blocks without calling the API.",
    )
    return parser.parse_args(argv)


def normalize_conditions(raw_conditions: list[str]) -> tuple[str, ...]:
    normalized = tuple(condition.strip().upper() for condition in raw_conditions)
    invalid = [condition for condition in normalized if condition not in {"A", "B", "C", "D"}]
    if invalid:
        raise RunModelsError("Invalid --conditions value(s): " + ", ".join(sorted(set(invalid))))
    return normalized


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    args = parse_args(argv)

    try:
        conditions = normalize_conditions(args.conditions)
        models = tuple(args.models)
        counters = run_pipeline(
            models=models,
            conditions=conditions,
            selected_studies=args.studies,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, RunModelsError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"Dry run complete. Printed {counters.dry_run_printed} study × model × condition request(s)."
        )
    else:
        print(
            "Run complete. "
            f"Successes: {counters.successful}. "
            f"Skipped existing: {counters.skipped_existing}. "
            f"Failures: {counters.failures}."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
