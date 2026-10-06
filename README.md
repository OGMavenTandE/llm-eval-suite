# LLM Eval Suite

Offline AI evaluation workbench engine for local LLM benchmarking. Built for operational users and AI Test & Evaluation (T&E) teams who need auditable, reproducible assessments of LLM performance, especially on air-gapped or edge deployments.

## Overview

This repo is the backend core of an **Offline AI Evaluation Workbench**. It runs YAML-driven evaluations against local or API-backed models, produces timestamped artifacts, and exposes a structured service layer that a local API and guided UI can build on in later milestones.

Milestone 1 stabilized the engine: installable packaging, typed schemas, filesystem storage/indexing, audit JSON per run, and a thin service layer over the existing evaluation runner.

## Current capabilities

- YAML-driven evaluations with no code changes required
- Pluggable model adapters (Ollama, OpenAI-compatible)
- Five built-in evaluators: correctness, latency, robustness, consistency, cost
- Model comparison mode with side-by-side reports
- Dry-run validation without inference
- Structured run results, per-run audit JSON, and a filesystem run index
- Service-layer accessors for listing runs, loading results, and retrieving artifacts

## Install

```bash
pip install -e .

# With test dependencies
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: `pydantic`, `pyyaml`, `requests`.

## Quick start (Windows)

The click-through app does not need Node. From the repo folder, double-click `run_app.bat` or run it in a terminal. The script creates `.venv`, installs the API extra, and opens http://127.0.0.1:8765.

```bat
run_app.bat
```

The same command by hand, after `pip install -e ".[api]"`:

```bat
python -m llm_eval_suite.app
```

Ollama example: start Ollama, choose type Ollama, base URL `http://127.0.0.1:11434`, and a model such as `llama3.2:3b`. Use Test connection, save the profile, pick the Quick preset, and click Run. The API key can stay blank.

Hugging Face folder example: install the optional extra in the same venv with `pip install -e ".[hf]"`. Point the folder field at a local model directory that already has `config.json` and a tokenizer (a GPT-2 Medium export is the expected shape). Set max context to `1024` for GPT-2. Base models should use completions mode.

A nanoGPT `ckpt.pt` (a file that contains `model_args`) can be converted with the button on the Connect screen, or:

```python
from llm_eval.models.nanogpt_convert import convert_nanogpt_to_hf
convert_nanogpt_to_hf(r"C:\models\ckpt.pt", r"C:\models\gpt2-export")
```

Copy a local GPT-2 tokenizer into the export folder before connecting it. The converter does not download tokenizer files.

Suggested local council judges, listed in the Judges screen: `qwen2.5:3b-instruct` and `llama3.2:3b`. If only one of those is installed, peer ranking is skipped. The model you are testing is not used as a judge.

`garak` runs live against the connected OpenAI-compatible endpoint when it is installed (`pip install garak` is optional and not part of the default install). Otherwise the Quick preset shows the fixture and labels it as a fixture. RAMPART stays a smoke check.

Runs are written under `runs/` and connection profiles under `data/`. Those directories are gitignored. Do not commit customer data, model weights, or real run outputs.

## CLI usage

```bash
# Validate config and dataset (no inference)
python -m llm_eval --config config/example_eval.yaml --dry-run

# Full evaluation (requires a running model, e.g. Ollama)
python -m llm_eval --config config/example_eval.yaml

# Compare multiple models
python -m llm_eval --config config/example_eval.yaml --compare

# Console script entry point (same behavior)
llm-eval --config config/example_eval.yaml --dry-run
```

For a step-by-step walkthrough, see [docs/TUTORIAL.md](docs/TUTORIAL.md).

## Local API (Milestone 2)

```bash
pip install -e ".[api]"
llm-eval-api
```

The API wraps the service layer with typed JSON endpoints for health checks, profile/dataset/model discovery, and run lifecycle management. See [apps/api/README.md](apps/api/README.md) for endpoint details.

## Local UI (Milestone 3)

```bash
pip install -e ".[api]"
llm-eval-api

cd frontend
npm install
npm run dev
```

Open http://localhost:5173 for the guided evaluation workbench UI. See [frontend/README.md](frontend/README.md).

## Dioptra light pilot

Offline crib of a NIST Dioptra experiment record. No Dioptra server, Docker stack, or GPU is required.

```bash
python -m llm_eval.dioptra smoke
```

Writes `results/dioptra/smoke-experiment.json`. See [docs/dioptra-light-pilot.md](docs/dioptra-light-pilot.md).

## RAMPART pytest smoke

Offline crib of a Microsoft RAMPART safety result. No RAMPART install, LLM credentials, Docker, or GPU is required.

```bash
python -m llm_eval.rampart smoke
```

Writes `results/rampart/smoke-report.json`. See [docs/rampart-pytest-smoke.md](docs/rampart-pytest-smoke.md).

## Garak fixture eval

Offline crib of an NVIDIA garak scan report. No garak install, live model, GPU, or NVIDIA service is required.

```bash
python -m llm_eval.garak smoke
```

Writes `results/garak/smoke-report.jsonl`. See [docs/garak-fixture-eval.md](docs/garak-fixture-eval.md).

## Architecture

```
llm-eval-suite/
├── apps/api/                  # Placeholder for Milestone 2 local API
├── config/                    # YAML evaluation configs
├── datasets/                  # JSONL/CSV test datasets
├── fixtures/garak/            # Canned probe/response rows for the garak smoke
├── llm_eval/
│   ├── cli.py                 # CLI (uses service layer)
│   ├── core/                  # Application services
│   │   ├── run_service.py     # Orchestration wrapper around EvalRunner
│   │   ├── config_service.py  # Config load/validate/normalize
│   │   ├── result_service.py  # Load summaries and detailed results
│   │   └── audit_service.py   # Build and persist audit metadata
│   ├── schemas/               # Pydantic contracts (RunRequest, RunStartResult, etc.)
│   ├── storage/               # Filesystem run index and artifact helpers
│   ├── runner.py              # Evaluation orchestrator
│   ├── models/                # Model adapters
│   ├── evaluators/            # Evaluation modules
│   ├── datasets/              # Dataset loader
│   ├── reporting/             # JSON/CSV reporters
│   ├── dioptra/               # Offline Dioptra-shaped experiment records
│   ├── rampart/               # Offline RAMPART-shaped pytest smoke records
│   └── garak/                 # Offline garak-shaped fixture reports
└── tests/
    ├── unit/
    └── integration/
```

### Service layer

| Service | Role |
|---------|------|
| `RunService` | Validate, dry-run, start runs; list/get runs, results, and artifacts |
| `ConfigService` | Load YAML, validate structure, normalize to `RunRequest` |
| `ResultService` | Load summary CSV and detailed JSON from artifact paths |
| `AuditService` | Build and persist structured audit records per run |

Example (Python):

```python
from llm_eval.core.run_service import RunService
from llm_eval.core.config_service import ConfigService

config = ConfigService().load_yaml("config/example_eval.yaml")
result = RunService().run_dry_run(config)
print(result.status, result.audit.audit_path)
```

## Run outputs and artifacts

Each evaluation run writes outputs under the configured `output_dir` (default `results/`).

**Per-model run directory** (`{output_dir}/{run_name}_{timestamp}/`):

- `results_detailed.json` — per-sample traces
- `results_summary.csv` — aggregated metrics

**Comparison directory** (multi-model runs):

- `comparison_summary.csv`
- `comparison_detailed.json`

**Run index and audit** (Milestone 1):

- `{output_dir}/.llm_eval_runs.json` — lightweight index of all runs
- `{output_dir}/audit/{run_id}.json` — structured audit metadata per run

Each audit JSON includes run ID, status, config hash, dataset path and sample count, model names/providers, evaluator names, timestamps, and `error_message` when a run fails.

Service-layer accessors:

- `list_runs(output_dir)`
- `get_run(run_id, output_dir)`
- `get_run_audit(run_id, output_dir)`
- `get_run_artifacts(run_id, output_dir)`
- `get_run_results(run_id, output_dir)`

Run statuses: `validated`, `completed`, `failed_validation`, `failed_runtime`.

## Development and tests

```bash
pip install -e ".[dev]"
python3 -m pytest -v
```

Tests cover dataset loader validation, dry-run via the service layer, reporter outputs, run index behavior, audit persistence, and runtime failure handling. Full runs with live inference are not required for the test suite.

## Next milestone

**Milestone 2** will add a local FastAPI layer in `apps/api/` exposing `RunService` endpoints, followed by a guided local web UI for non-technical operational users.

## Configuration

See [config/example_eval.yaml](config/example_eval.yaml) for a full multi-model config. Required keys: `dataset`, `models`, `evaluators`.

Dataset format (JSONL):

```json
{"prompt": "What is 2+2?", "expected_answer": "4", "category": "math"}
```

## License

MIT
