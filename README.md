# LLM Eval Suite

A local FastAPI app for point-and-click test and evaluation of language models, at http://127.0.0.1:8765.
It is for test and evaluation teams who need a repeatable check of a model on a machine they control.

Built by [AI Eval Corp](https://aieval.org), with test and evaluation built into development.

## Quick start

Python 3.10 or newer. From a fresh clone:

```bash
python -m pip install -e ".[dev,api]"
python -m llm_eval --config config/example_eval.yaml --dry-run
python -m llm_eval_suite.app
```

The dry-run checks `config/example_eval.yaml` and its dataset and does not call a model. The app listens on http://127.0.0.1:8765.

On Windows, `run_app.bat` creates `.venv-eval`, installs pinned `torch==2.13.0`, `garak==0.17.0`, and `transformers==5.18.0`, and starts the same app. The app opens the browser once. Use that script for live garak or a local Hugging Face model. If `nvidia-smi` is on PATH, torch comes from the PyTorch CUDA 12.6 index. Otherwise the script warns and installs the CPU build. If a GPU is present and `.venv-eval` already has a CPU torch build, the script reinstalls the CUDA wheel. Torch is installed before garak so pip does not replace it with a CPU wheel.

The YAML engine API is a different process: `llm-eval-api` listens on http://127.0.0.1:8000. See [apps/api/README.md](apps/api/README.md).

## Sample result

Higher is better. A garak pass means the reply did not show the failure that probe looks for.

Measured on 6 October 2026 on a Windows test PC (RTX 4070 SUPER) with garak 0.17.0. The fine-tune is GPT-2 Medium trained on 895 public DVIDS / Department of War news stories (`hf-dow-news`), compared with base `gpt2-medium`. These counts are from that run, not from `datasets/sample_factcheck_50.jsonl`.

| Test | Base gpt2-medium | Fine-tune (hf-dow-news) |
| --- | --- | --- |
| Fact-check, 50 questions (keyword) | 21/50 (42%) | 23/50 (46%) |
| Same, after dropping the Newport News false pass | 20/50 (40%) | 22/50 (44%) |
| Unseen articles (28) | 8/28 (29%) | 7/28 (25%) |
| garak FalseAssertion, MustRefuteClaimKW | 167/450 (37.1%) | 142/450 (31.6%) |
| garak FalseAssertion, MustRefuteClaimModel | 121/450 (26.9%) | 76/450 (16.9%) |
| garak FalseAssertion, MustContradictNLI | 69/450 (15.3%) | 37/450 (8.2%) |
| garak LiteratureCloze | 762/768 (99.2%) | 681/690 (98.7%), CUDA crash, not 768 |

The keyword headline is slightly higher for the fine-tune (23/50, or 22/50 after removing a shared false pass: both said "Newport News" and the keyword Newport matched, but the Naval War College is in Newport, Rhode Island). That is not a factuality gain. On 28 questions from unseen articles the fine-tune is 7/28 against 8/28 for the base, and garak FalseAssertion is worse on all three detectors. The LiteratureCloze fine-tune cell is 681 passes out of 690 attempts from a run that crashed CUDA, not out of 768.

## Overview

The repo has three ways to run an evaluation. The click-through app is `python -m llm_eval_suite.app` or `run_app.bat`. The YAML engine is `python -m llm_eval`. The local API and React workbench sit on top of the same engine.

The engine runs YAML configs against local or API-backed models, writes timestamped artifacts, and keeps an audit JSON per run. The click-through app adds presets, a scorecard, compare, an HTML report, a local-judge council, and live garak when that package is installed.

## Current capabilities

- YAML-driven evaluations with no code changes required
- Model connectors: Ollama, OpenAI-compatible, and a local Hugging Face folder
- nanoGPT `ckpt.pt` to a Hugging Face GPT-2 folder, with a logits check
- Five built-in evaluators: correctness, latency, robustness, consistency, cost
- Presets: Quick, Standard, Full, and Government T&E
- Model comparison, a category scorecard, and an HTML report for browser print
- Council summary from local judge models. The model under test is not a judge
- Live garak when installed. A missing install shows a labeled fixture. A failed live scan is INVALID, not a fixture
- Dry-run validation without inference
- Structured run results, per-run audit JSON, and a filesystem run index

## Click-through app

Ollama example: start Ollama, choose type Ollama, base URL `http://127.0.0.1:11434`, and a model such as `llama3.2:3b`. Use Test connection, save the profile, pick the Quick preset, and click Run. The API key can stay blank for a local server. The Run screen shows probe count, prompt count, and a time estimate before you start. That estimate is reused only after a measured run of the same model, endpoint, and suite. Otherwise it stays on Estimating until this run has timed a few prompts. While garak is running, the progress line shows the attempt count.

Hugging Face folder example: point the folder field at a local model directory that already has `config.json` and a tokenizer (a GPT-2 Medium export is the expected shape). Set max context to `1024` for GPT-2. Base models should use completions mode. Prompt plus new tokens are clamped to that window (the prompt is shortened from the left).

A nanoGPT `ckpt.pt` (a file that contains `model_args`) can be converted with the button on the Connect screen, or:

```python
from llm_eval.models.nanogpt_convert import convert_nanogpt_to_hf
convert_nanogpt_to_hf("ckpt.pt", "gpt2-export")
```

The converter transposes Conv1D weights, copies tied embeddings when `lm_head` is missing, writes vocab size plus `n_ctx` and `n_positions`, sets `activation_function` to exact `gelu` (not `gelu_new`), and checks logits against the source checkpoint on a fixed prompt. If the max absolute difference is above `1e-4`, conversion raises and still writes `conversion_report.json`. Copy a local GPT-2 tokenizer into the export folder before connecting it. The converter does not download tokenizer files.

Suggested local council judges, listed in the Judges screen: `qwen2.5:3b-instruct` and `llama3.2:3b`. If only one of those is installed, peer ranking is skipped. The model you are testing is not used as a judge.

When garak is installed it runs live against the connected endpoint. Pass rate in the UI is `1 - ASR` (one minus garak's attack success rate). A higher pass rate means fewer successful attacks. An attempt passes when no detector hit reaches the threshold. The Security card, that `1 - ASR` line, and the HTML report use the same live rows and the same percent. Toxicity stays on its own card. Fixture and smoke rows, including the RAMPART check, are not mixed into the live percent. Garak writes each attempt twice. Only the copy with detector results is scored. An attempt with no detector results is not a pass. Progress counts one row per attempt and keeps that count when leakreplay starts its own process. The planned total is the prompts those probes will send, after the cap, when garak can report the prompt lists. The live ETA starts its per-prompt clock at the first completed attempt, so process startup is not priced as a prompt, then blends that rate with the pre-run per-prompt estimate. The live rate gains weight as more of the suite finishes. A finished suite row shows done and the elapsed time, and that elapsed time freezes once the suite reaches its planned total. Garak's own reports stay under the user home directory at `.local/share/garak/garak_runs`. Console output goes to `runs/<id>/run.log`, not the results page. leakreplay runs in its own process so a CUDA fault does not empty later probes. A run with too many empty generations is marked INVALID. If garak is installed and still writes no report, the suite records a failed live scan and does not substitute the fixture. If garak is missing, the Quick preset shows the fixture and labels it as a fixture. RAMPART stays a smoke check.

A category meets the pass bar only when its own live rate is at or above 80%. The Results headline is that verdict, not an average of the category rates. The 80% bar comes from the scorecard API. An invalid run shows "Score withheld" on the headline and on every category, and omits those percents from the run list. Compare marks an invalid run and does not treat a delta against it as a score. Fixture rows stay a small grey label and are left out of the failing-prompt count. Half or more empty live fact-check answers mark the run INVALID as well. Ollama calls send `think: false`, and `<think>` blocks are stripped before scoring, so a thinking model does not spend the whole reply inside the trace. Council judges on Ollama use the native chat endpoint instead, still with think turned off, and a 1200 token reply cap unless the Judges screen changes it. The keyword scorer applies Unicode NFKC, so a subscript digit matches the plain digit, and it reads compound number words through 9999, including "and" and hyphens. A thousands separator in a digit (`1,200`) matches the same number in words. A comma between number words ends the phrase. Council summaries may quote ranking points, the complement of an input percent, a failed count when the input gives both the total and the passed count, and a count from that same object's total times its pass rate or failing share, within half a count plus that rate's rounding step. Other sums and differences are rejected. Ollama judges use the native chat endpoint with think turned off and a context sized to the prompt. An empty chairman reply is handed to the other judge before the template. A summary that stops mid-sentence is finished with more room or cut back to the last complete sentence. Notes about omitted numbers are removed from the summary. Compare orders the two runs by time and labels each with its model and the same local timestamp as the run dropdown. Garak prompts are paired by probe and prompt text. When some prompts cannot be paired, Compare says how many in each run, instead of adding the two runs together. It says that no matching prompts changed only when no paired prompt changed. Prompt rows with no score change are left out. The pre-run estimate counts each fact-check and each garak prompt once, then scales a measured total by the observed actual-to-estimate ratio.

A Hugging Face folder uses CUDA when the installed torch can see a GPU, in float16, and the run record stores that device. The keyword scorer treats number words as the digits they name, from a single word such as "Eight" through a compound such as "one hundred and fifty-six", up to 9999. An uploaded fact-check file may use `expected` or `answer` in place of `expected_answer`. Export HTML report opens the page. Download HTML report saves a file with the fonts embedded. Council judges wait 90 seconds by default and may write up to 1200 tokens. The Judges screen can change both. The Run screen ETA uses the current suite's item count. After the suite finishes, that row shows done and the elapsed time.

The Run screen can run the Quick preset on an editable local Hugging Face folder and on a local `gpt2-medium` folder (`models/gpt2-medium`, or `LLM_EVAL_DEMO_BASE_FOLDER`), then open Compare on that pair. The demo does not download weights. See [docs/AIRGAP_INSTALL.md](docs/AIRGAP_INSTALL.md) for a closed-network install.

Long runs are a background process with a per-run log. Cancel still works. Resume skips completed fact-check cases and completed garak probes. The progress line shows a live ETA.

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

For a step-by-step walkthrough, see [docs/TUTORIAL.md](docs/TUTORIAL.md). A single-file viewer for engine JSON is [dashboard.html](dashboard.html). See [docs/DASHBOARD.md](docs/DASHBOARD.md).

## Local API

```bash
pip install -e ".[api]"
llm-eval-api
```

The API wraps the service layer with typed JSON endpoints for health checks, profile/dataset/model discovery, and run lifecycle management. It listens on http://127.0.0.1:8000. See [apps/api/README.md](apps/api/README.md).

## Local UI

```bash
pip install -e ".[api]"
llm-eval-api

cd frontend
npm install
npm run dev
```

Open http://localhost:5173 for the guided evaluation workbench UI. See [frontend/README.md](frontend/README.md). This UI is separate from the click-through app on port 8765. `run_app.bat` does not start it.

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

## Garak

The click-through app runs garak live when the package is installed. This command is the offline fixture. No garak install, live model, GPU, or NVIDIA service is required.

```bash
python -m llm_eval.garak smoke
```

Writes `results/garak/smoke-report.jsonl`. See [docs/garak-fixture-eval.md](docs/garak-fixture-eval.md). How empty generations are flagged is in [docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md).

## Architecture

```
llm-eval-suite/
├── apps/api/                  # Local FastAPI shell (port 8000)
├── llm_eval_suite/            # Click-through app (port 8765), presets, council, HTML report
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
│   ├── models/                # Ollama, OpenAI-compatible, HF folder, nanoGPT convert
│   ├── evaluators/            # Evaluation modules
│   ├── datasets/              # Dataset loader
│   ├── reporting/             # JSON/CSV reporters
│   ├── dioptra/               # Offline Dioptra-shaped experiment records
│   ├── rampart/               # Offline RAMPART-shaped pytest smoke records
│   └── garak/                 # Live garak scans and offline fixture reports
├── frontend/                  # React workbench (optional; not the port 8765 app)
├── dashboard.html             # Single-file viewer for engine JSON
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

- `results_detailed.json`: per-sample traces
- `results_summary.csv`: aggregated metrics

**Comparison directory** (multi-model runs):

- `comparison_summary.csv`
- `comparison_detailed.json`

**Run index and audit**:

- `{output_dir}/.llm_eval_runs.json`: lightweight index of all runs
- `{output_dir}/audit/{run_id}.json`: structured audit metadata per run

Each audit JSON includes run ID, status, config hash, dataset path and sample count, model names/providers, evaluator names, timestamps, and `error_message` when a run fails.

Service-layer accessors:

- `list_runs(output_dir)`
- `get_run(run_id, output_dir)`
- `get_run_audit(run_id, output_dir)`
- `get_run_artifacts(run_id, output_dir)`
- `get_run_results(run_id, output_dir)`

Run statuses: `validated`, `completed`, `failed_validation`, `failed_runtime`.

Click-through runs are separate. They live under `runs/<id>/` as `run.json`, `items.jsonl`, and `manifest.json`.

## Development and tests

GitHub Actions runs this on Ubuntu and Windows, Python 3.11:

```bash
python -m pip install -e ".[dev,api]"
python -m pytest
```

Tests cover dataset loader validation, dry-run via the service layer, reporter outputs, run index behavior, audit persistence, runtime failure handling, and the click-through app with torch and garak stubbed. Full runs with live inference are not required. Tests that need torch skip when it is not installed.

## Configuration

See [config/example_eval.yaml](config/example_eval.yaml) for a full multi-model config. Required keys: `dataset`, `models`, `evaluators`.

Dataset format (JSONL):

```json
{"prompt": "What is 2+2?", "expected_answer": "4", "category": "math"}
```

YAML model providers are `ollama` and `openai`. The Hugging Face folder connector is on the click-through app, not in the YAML provider list.

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
