# AGENTS.md

## Overview

LLM Eval Suite is a local test and evaluation app for language models, built by AI Eval Corp. The click-through FastAPI app listens on 127.0.0.1:8765. The YAML engine, a JSON API on port 8000, and an optional React workbench sit beside it.

Read [docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md) before changing garak scoring or the nanoGPT converter.

## Repo map

- `llm_eval_suite/`: click-through app, presets, runs, council, HTML report
- `llm_eval_suite/static/`: side-rail shell (Connect, Run, Results, Compare, Judges). Public Sans is bundled in `static/fonts/` (OFL) so the app renders offline. `tokens.css` is the shared palette
- `llm_eval_suite/static/app.js`: a live category passes only when it is at or above the pass bar from the scorecard API. The headline is not an average. An invalid run shows "Score withheld". Fixture meters stay small and grey. The ETA uses that suite's item count and hides when the suite is finished. Compare lists the largest score changes first
- `llm_eval_suite/report_html.py`: standalone HTML report. It inlines `tokens.css` and embeds the font files. The page is `class="report-doc"` (`color-scheme: light`) so print stays light
- `llm_eval_suite/presets.json`: Quick, Standard, Full, Government T&E, and the demo pair
- `llm_eval/`: YAML engine, CLI, runner, storage, reporting
- `llm_eval/models/`: Ollama, OpenAI-compatible, Hugging Face folder, nanoGPT convert
- `llm_eval/garak/live.py`: live garak, empty-generation INVALID flag, pass rate = 1 - ASR
- `llm_eval/evaluators/`: correctness, latency, robustness, consistency, cost
- `apps/api/`: JSON API on 127.0.0.1:8000 (`llm-eval-api`)
- `frontend/`: React workbench (Vite). `run_app.bat` does not start it
- `config/example_eval.yaml`: example multi-model YAML config
- `datasets/`: sample JSONL. The 50-row fact-check file is general knowledge, not the Department of War set
- `fixtures/garak/`: canned rows for the garak smoke
- `tests/`: pytest suite that GitHub Actions runs
- `docs/`: tutorial, dashboard, smoke notes, lessons
- `dashboard.html`: single-file viewer for engine JSON. It embeds Public Sans and uses the same palette as the exported report
- `run_app.bat`: Windows launcher. Creates `.venv-eval` and pins torch, garak, and transformers
- `pyproject.toml`: package metadata. Version 0.1.0

## Setup

Python 3.10 or newer. CI uses 3.11.

```bash
python -m pip install -e ".[dev,api]"
```

Optional Hugging Face weights: `python -m pip install -e ".[hf]"`. On Windows, `run_app.bat` is the path that also installs `torch==2.13.0`, `garak==0.17.0`, and `transformers==5.18.0` into `.venv-eval`. Pins live in `llm_eval_suite/eval_env.py`.

## Build

There is no compile step in CI. The install above is the build.

The optional React UI, not run by CI:

```bash
cd frontend
npm install
npm run build
```

## Test

GitHub Actions (`.github/workflows/ci.yml`) runs on `ubuntu-latest` and `windows-latest`, Python 3.11:

```bash
python -m pip install -e ".[dev,api]"
python -m pytest
```

Ubuntu CI also runs `python -m playwright install --with-deps chromium` before pytest. `tests/ui/test_screens.py` loads each screen and checks an invalid run withholds its score. It skips when Chromium cannot launch, so the Windows job stays green without a browser install.

That install does not include torch or garak. Tests that need torch use `pytest.importorskip` and skip. Do not add a test that needs a GPU, a live model, or a network call.

`cd frontend && npm test` runs Vitest. CI does not run it.

Offline smokes, not the CI gate:

```bash
python -m llm_eval --config config/example_eval.yaml --dry-run
python -m llm_eval.garak smoke
python -m llm_eval.dioptra smoke
python -m llm_eval.rampart smoke
```

## Conventions

Match the surrounding Python: type hints, Pydantic models where the module already uses them, and tests next to the behavior you change.

Branch names: `cursor/<topic>-<suffix>`.

Pull requests: short plan, then a BEFORE/AFTER of what a reader or agent sees. Do not commit secrets, API keys, model weights, customer data, or real run outputs. `runs/`, `data/`, `.env`, `.venv/`, `.venv-eval/`, `*.pt`, `*.safetensors`, `*.bin`, and `*.gguf` are gitignored. Leave them that way.

## Extending

New YAML evaluator: subclass `BaseEvaluator` in `llm_eval/evaluators/`, add it to `EVALUATOR_REGISTRY` in `llm_eval/evaluators/__init__.py`, and add the name to `SUPPORTED_EVALUATORS` in `llm_eval/core/config_service.py`.

New YAML model provider: add the class to `MODEL_REGISTRY` in `llm_eval/models/__init__.py` and the name to `SUPPORTED_PROVIDERS` in `config_service.py`. Today those names are `ollama` and `openai`. The Hugging Face folder path is a click-through connection type in `llm_eval_suite/connections.py` (`openai`, `ollama`, `hf`, `nanogpt`), not a YAML provider.

New preset: add an object to `llm_eval_suite/presets.json`. Keys the runner understands are `garak`, `factcheck`, `robustness`, `consistency`, `rampart`, and `dioptra`.

New garak probe list: edit the preset's `garak.probes`. A probe name containing `leakreplay` is run in its own process.

## Guardrails

- Do not score empty garak text as a pass or as an attack. Empty rows are excluded. Too many empties, or a failed probe process, marks the run INVALID. See `llm_eval/garak/live.py`. Half or more empty live fact-check answers also mark the run INVALID, in `llm_eval_suite/runs.py`.
- Garak writes each attempt twice. Score the copy that has detector results (status 2). The earlier copy is not a pass. Progress and empty-generation counts use one row per attempt.
- Pass rate is `1 - ASR`. Do not relabel attack success as a pass rate. The UI string is `Pass rate (1 - ASR)`. For a garak category that line and the category meter use the same rate. Do not average category rates into one headline score. A run passes only when every live category is at or above the pass bar.
- An invalid run withholds every category score, in Results and in Compare. Do not show a delta against an invalid run as a real score.
- Reuse a timing rate only for the same model, endpoint, and suite. Otherwise the estimate stays Estimating until a few prompts of this run are timed.
- Council judge calls send `think: false` and use 1200 tokens unless the Judges screen or `LLM_EVAL_JUDGE_MAX_TOKENS` sets another cap. Strip `<think>` blocks from the reply.
- A missing garak install is the labeled fixture. A garak install that writes no report is a failed live scan, not the fixture.
- Do not drop the nanoGPT logits check or raise `LOGITS_TOLERANCE` (`1e-4`) to hide a mismatch. The converter does not download tokenizer files.
- RAMPART stays a smoke check. Dioptra stays an offline record. The garak smoke fixture must stay labeled as a fixture.
- Do not commit weights, `runs/`, or `data/`. Do not put a personal machine path in docs or in a new default.
- Do not treat `datasets/sample_factcheck_50.jsonl` as the Department of War fact-check set.
