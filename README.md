# Can LLMs assess risk of bias in medical research?

Code and data for: Juneau C-E. *Can LLMs assess risk of bias in medical research?* 2026.

**Pre-registration:** Open Science Framework, https://osf.io/8grbe (registered April 7, 2026).

## Overview

This study tests whether two large language models (Gemini Flash and Gemini Pro) can reproduce expert risk-of-bias labels for 14 observational COVID-19 studies, and whether agreement improves as prompts cumulatively add criteria definitions, training material, and a worked example.

**Four prompt conditions:**

| Condition | What the model receives |
|-----------|------------------------|
| A — baseline | Study PDF + minimal task instruction |
| B — criteria definitions | A + 8 criterion definitions, derivation rule, output schema |
| C — training material | B + full text of Mulder et al. (2019) |
| D — worked example | C + a worked example (Green et al., 2019) |

**Two models:**

| Label | Model ID |
|-------|----------|
| Flash (weaker) | `gemini-3-flash-preview` |
| Pro (stronger) | `gemini-3.1-pro-preview` (thinking_level: high) |

## Repo structure

```
data/
  public/
    Table 1 - RoB_observational_studies.csv   # 14 studies with expert gold labels
    Table 2 - RoB_criteria.csv                # 8 criterion definitions
    Table 3 - RoB_criteria_mapping.csv        # Mapping from Mulder et al. (2019) criteria
  private/                                    # Not in repo — add study PDFs here
prompts/
  condition_a.txt                            # Prompt template for Condition A
  condition_b.txt                            # Additional text for Condition B
  condition_c.txt                            # Separator text for Condition C
  condition_d.txt                            # Separator text for Condition D
  examples/
    green2019_condition_d_example.json       # Worked example output for Condition D
src/
  run_models.py     # Calls the Gemini API; writes raw + parsed outputs
  score_results.py  # Scores parsed outputs against gold labels
  schema.py         # Output schema, validation, and RoB derivation logic
tests/
  test_score_results.py   # Tests for score_results.py
  test_schema.py          # Tests for schema.py
  test_run_models.py      # Tests for run_models.py pure functions
results/                  # Outputs from the main run (missingness rule removed)
  raw/                    # Raw model responses (.txt)
  parsed/                 # Validated JSON outputs
  parse_failures.csv      # Calls that failed to produce valid JSON
  scored_summary.csv      # Per-study accuracy metrics
results_with_missingness/ # Outputs from the earlier run that included a missingness rule
  raw/
  parsed/
  parse_failures.csv
  scored_summary.csv
```

**Private data (not in this repo):** Study PDFs are required to run `run_models.py`. Place each PDF as `data/private/observational/<study_id>.pdf`, where `<study_id>` matches the "Study first author" column in Table 1.

**`results/` vs `results_with_missingness/`:** A post hoc sensitivity analysis found that a missingness rule originally in the protocol markedly depressed agreement for both models. That rule was removed, and `results/` contains the operative outputs (rule removed). `results_with_missingness/` preserves the earlier run with the rule in place for comparison. Both are described in the paper.

## Setup

Python 3.11+ is recommended.

Install dependencies:

```bash
pip install google-genai scipy
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

## Running the pipeline

### Step 1: Run the models

```bash
cd src
python run_models.py
```

This sends API requests for all 14 studies × 2 models × 4 conditions and writes outputs to `results/raw/` and `results/parsed/`. Re-running skips files that already exist.

Selected options:

```
--studies <id> [<id> ...]   Run a subset of studies by ID
--models <id> [<id> ...]    Run a subset of models
--conditions A B C D        Run a subset of conditions
--dry-run                   Print prompts without calling the API
--force                     Overwrite existing output files
```

### Step 2: Score the results

```bash
cd src
python score_results.py
```

This reads `results/parsed/` and writes `results/scored_summary.csv`. It also prints a human-readable report to stdout.

### Step 3: Run the tests

```bash
pytest tests/
```

## Data

**Table 1** (`data/public/Table 1 - RoB_observational_studies.csv`): Expert risk-of-bias labels for the 14 test studies. Source: Table 3 in Juneau CE et al. (2023). *Effective contact tracing for COVID-19: A systematic review.* Global Epidemiology. https://doi.org/10.1016/j.gloepi.2023.100103

**Table 2** (`data/public/Table 2 - RoB_criteria.csv`): The 8 criteria used for assessment. Source: Table 1 in Juneau et al. (2023), adapted from Mulder et al. (2019).

**Table 3** (`data/public/Table 3 - RoB_criteria_mapping.csv`): Maps each criterion used here to its antecedent in Mulder et al. (2019) and documents any adaptations. Corresponds to Appendix A in the paper.

## Citation

Juneau C-E. Can LLMs assess risk of bias in medical research? 2026. Pre-registration: https://osf.io/8grbe.

## Known limitations

**Condition A validation.** The schema validator (`schema.py`) requires a JSON object with `study_id` and `overall_rob` keys for all conditions, including Condition A. In practice, the minimal Condition A prompt does not reliably produce that exact structure, so all Condition A API outputs fail validation and are recorded as parse failures in `results/parse_failures.csv`. `score_results.py` handles this by reading raw response files directly for Condition A rather than the validated parsed JSON. The Condition A agreement numbers reported in the paper come from that raw-file path. This behavior was present in the original run and is preserved as-is for fidelity to the published study.

**Condition A retry recovery.** The raw-file parser in `score_results.py` reads only the first-attempt file (`<study>_<model>_A_raw.txt`). If a first attempt fails and a retry recovers, the retry output (`_attempt2_raw.txt`) is not used in Condition A scoring.
