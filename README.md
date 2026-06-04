# LLM Eval Suite

Offline AI evaluation workbench engine for local LLM benchmarking. Built for operational users and AI Test & Evaluation (T&E) teams who need auditable, reproducible assessments of LLM performance, especially on air-gapped or edge deployments.

## Overview

This repo is the backend core of an **Offline AI Evaluation Workbench**. It runs YAML-driven evaluations against local or API-backed models, produces timestamped artifacts, and exposes a structured service layer that a local API and guided UI can build on in later milestones.

Milestone 1 stabilized the engine: installable packaging, typed schemas, filesystem storage/indexing, and a thin service layer over the existing evaluation runner.

## Current capabilities

- YAML-driven evaluations with no code changes required
- Pluggable model adapters (Ollama, OpenAI-compatible)
- Five built-in evaluators: correctness, latency, robustness, consistency, cost
- Model comparison mode with side-by-side reports
- Dry-run validation without inference
- Structured run results, audit metadata, and a filesystem run index
- Service layer ready for CLI, API, and UI consumers

## Install

```bash
pip install -e .

# With test dependencies
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: `pydantic`, `pyyaml`, `requests`.

## Run from CLI

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

## Architecture

```
llm-eval-suite/
├── apps/api/                  # Placeholder for Milestone 2 local API
├── config/                    # YAML evaluation configs
├── datasets/                  # JSONL/CSV test datasets
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
│   └── reporting/             # JSON/CSV reporters
└── tests/
    ├── unit/
    └── integration/
```

### Service layer

The service layer is the stable contract for future UI and API work:

| Service | Role |
|---------|------|
| `RunService` | Validate, dry-run, start runs; list/get runs and results |
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

## Output and run artifacts

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

Use `RunService.list_runs()`, `get_run()`, `get_run_audit()`, and `get_run_results()` to retrieve indexed runs programmatically.

## Development and tests

```bash
pip install -e ".[dev]"
python3 -m pytest -v
```

Tests cover dataset loader validation, dry-run via the service layer, reporter outputs, storage/index behavior, and audit persistence. Full runs with live inference are not required for the test suite.

## Roadmap / next step

**Milestone 2** will add a local FastAPI layer in `apps/api/` exposing `RunService` endpoints, followed by a guided local web UI for non-technical operational users.

## Configuration

See [config/example_eval.yaml](config/example_eval.yaml) for a full multi-model config. Required keys: `dataset`, `models`, `evaluators`.

Dataset format (JSONL):

```json
{"prompt": "What is 2+2?", "expected_answer": "4", "category": "math"}
```

## License

MIT
