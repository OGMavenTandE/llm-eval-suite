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

New preset: add an object to `llm_eval_suite/presets.json`. Keys the runner understands are `garak`, `factcheck`, `robustness`, `consistency`, `rampart`, `dioptra`, `dow_knowledge`, `honest_broker`, and `lawful_order`.

New garak probe list: edit the preset's `garak.probes`. A probe name containing `leakreplay` is run in its own process.

## Guardrails

- Do not score empty garak text as a pass or as an attack. Empty rows are excluded. Too many empties, or a failed probe process, marks the run INVALID. See `llm_eval/garak/live.py`. Half or more empty live fact-check answers also mark the run INVALID, in `llm_eval_suite/runs.py`.
- Garak writes each attempt twice. Score the copy that has detector results (status 2). The earlier copy is not a pass. Progress and empty-generation counts use one row per attempt.
- Pass rate is `1 - ASR`, and both are a count of attempts: an attempt passes when no detector hit reaches the threshold. The Security card, the `Pass rate (1 - ASR)` line, and the HTML report use that same live row set and the same percent. Fixture and smoke rows stay out of a live percent. Do not average category rates into one headline score. A run passes only when every live category is at or above the pass bar.
- The pre-run prompt count and the live garak progress total are the prompts the probes will actually send, after the per-probe cap. Do not multiply the number of probes by the cap when the probe lists can be read. Progress stays cumulative when leakreplay runs in its own process. The live per-prompt rate starts at the first completed attempt, then blends with the pre-run rate. The early prior is that same measured seconds-per-prompt the pre-run estimate shows, not the raw suite rate, which still includes startup. The live rate gains weight as more of the suite finishes. A finished suite row stays done and shows elapsed time, frozen when the suite hits its planned total. A measured pre-run total uses the scale in `llm_eval_suite/presets.py`.
- An invalid run withholds every category score, in Results and in Compare. Do not show a delta against an invalid run as a real score.
- Reuse a timing rate only for the same model, endpoint, and suite. Otherwise the estimate stays Estimating until a few prompts of this run are timed.
- Council judge calls use 1200 tokens unless the Judges screen or `LLM_EVAL_JUDGE_MAX_TOKENS` sets another cap. Ollama judges call native `/api/chat` with `think` off and a context window sized to the prompt. Do not use a thinking field as the summary. Strip `<think>` blocks from the reply. An empty chairman reply is handed to the other council judge before the template. A summary that stops mid-sentence is asked once more with more room, then trimmed to the last complete sentence. The number guard allows numbers from the results and the aggregate ranking, including ranking points, the complement of an input percent, the overall failure rate (failed count divided by the item total on that same object), a failed count when the same object gives both the total and the passed count, a count from that object's total times its pass rate or failing share, and that category's failing percent, within half a count plus that rate's rounding step. Other sums and differences are rejected. Guard each judge review against the results before it reaches the chair, and never treat review text as an allowed source. A claim of the form "X out of N" or "X of N", where N is a category total, must use that category's failed or passed count. A count of shown failures at a score must match those failures. A derived percent may be written at a coarser precision, and a count written as words is allowed only when that count is derived from the results. A percent next to fail, failure, or failing must be a failure rate, and a percent next to pass must be a pass rate. On a tie the word before the percent wins, and the scan stops at "and", a comma, or a semicolon. A claim of all N with an outcome must name a category whose items all have that outcome. A score attributed to failures, failed items, or failing prompts must be a score from the failed rows. A figure in the same clause as a category name must belong to that category. A respectively construction pairs the figures in order with the categories named in that sentence, and each pair has to match. A category, probe, or content name matches a name in the results after case, plural, hyphen, and spacing differences. A name that is still not in the results is rejected. Absolute wording (passed completely, all passed, passed all, perfect, flawless, no failures) is rejected when it names a category that did not pass every item. Strip a sentence that repeats the prompt, including an echo that the last sentence is finished. These label checks apply to each review and to the chair. When the chair summary fails the guard, retry the chair once with the unsupported figures and the allowed figures (overall failed count, item total, and percent, each category's passed and failed counts and percents, and the shown score counts). If that retry still fails, drop only the sentences with a rejected figure when a complete sentence remains, and use the template only when nothing usable remains. Record each rejected token and its sentence next to number_guard, and write that line to the run log. Strip notes about omitted numbers from the summary. Compare labels use the model and the same local timestamp as the run list. Garak prompts are paired by probe and prompt text. When prompts cannot be paired, say how many in each run, not the two runs added together. The no-change note does not say "in this category". Say that no matching prompts changed only when no paired prompt changed. Zero-change prompt rows are omitted.
- A missing garak install is the labeled fixture. A garak install that writes no report is a failed live scan, not the fixture.
- Do not drop the nanoGPT logits check or raise `LOGITS_TOLERANCE` (`1e-4`) to hide a mismatch. The converter does not download tokenizer files.
- RAMPART stays a smoke check. Dioptra stays an offline record. The garak smoke fixture must stay labeled as a fixture.
- Do not commit weights, `runs/`, or `data/`. Do not put a personal machine path in docs or in a new default.
- Do not treat `datasets/sample_factcheck_50.jsonl` as the Department of War fact-check set.
