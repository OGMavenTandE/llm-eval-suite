# LLM Eval Suite

A modular, offline-capable Python evaluation suite for large language models. Built for AI Test & Evaluation (T&E) personnel who need auditable, reproducible assessments of LLM performance — especially quantized and open-weight models running on edge platforms.

## Features

- **YAML-driven evaluations** — define what to test without writing code
- **Vendor-agnostic** — pluggable model adapters (Ollama, OpenAI-compatible, extensible)
- **Offline-first** — runs fully air-gapped with local models via Ollama
- **Auditable output** — timestamped per-sample JSON traces and summary CSVs
- **5 built-in evaluators** — correctness, latency, robustness, consistency, cost
- **Model comparison mode** — run multiple models side-by-side with automatic comparison reports
- **Scalable** — add new models or evaluators by subclassing a base class

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run an evaluation (requires Ollama running locally)
python -m llm_eval --config config/example_eval.yaml

# Dry run (validate config without running inference)
python -m llm_eval --config config/example_eval.yaml --dry-run

# Compare multiple models
python -m llm_eval --config config/example_eval.yaml --compare
```

For a detailed step-by-step walkthrough, see **[docs/TUTORIAL.md](docs/TUTORIAL.md)**.

## Project Structure

```
llm-eval-suite/
├── config/                    # YAML evaluation configs
│   └── example_eval.yaml
├── datasets/                  # Test datasets (JSONL/CSV)
│   └── sample_correctness.jsonl
├── docs/
│   └── TUTORIAL.md            # Detailed beginner's tutorial
├── llm_eval/                  # Main package
│   ├── cli.py                 # CLI entry point
│   ├── runner.py              # Orchestrator
│   ├── models/                # Model adapters
│   │   ├── base.py            # Abstract interface
│   │   ├── ollama_model.py    # Local Ollama adapter
│   │   └── openai_model.py    # OpenAI-compatible adapter
│   ├── evaluators/            # Evaluation modules
│   │   ├── base.py            # Abstract interface
│   │   ├── correctness.py     # Exact, fuzzy, LLM-as-judge
│   │   ├── latency.py         # Response time benchmarking
│   │   ├── robustness.py      # Prompt perturbation testing
│   │   ├── consistency.py     # Determinism / repeatability
│   │   └── cost.py            # Token usage & cost estimation
│   ├── datasets/
│   │   └── loader.py          # JSONL/CSV dataset loader
│   └── reporting/
│       ├── reporter.py        # Per-model JSON/CSV results
│       └── comparison.py      # Multi-model comparison reports
└── results/                   # Output directory (auto-created)
```

## Evaluators

| Evaluator | What it measures | Key config |
|-----------|-----------------|------------|
| **correctness** | Answer accuracy (exact, fuzzy, or LLM-as-judge) | `mode`, `threshold` |
| **latency** | Response time | `max_ms` |
| **robustness** | Stability under prompt perturbations (typos, case, rephrasing) | `perturbations`, `threshold` |
| **consistency** | Determinism — same prompt N times, how similar are the answers? | `num_runs`, `threshold` |
| **cost** | Token usage tracking and cost estimation | `cost_per_1k_tokens`, `max_tokens_per_response` |

## Configuration

Evaluations are defined in YAML:

```yaml
run_name: "multi-model-eval"
models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"
  - name: "llama3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512

evaluators:
  - name: "correctness"
    mode: "fuzzy_match"
    threshold: 0.8
  - name: "latency"
    max_ms: 5000
  - name: "robustness"
    perturbations: [typo, case, rephrase]
    threshold: 0.7
  - name: "consistency"
    num_runs: 3
    threshold: 0.8
  - name: "cost"
    cost_per_1k_tokens: 0.0
    max_tokens_per_response: 512

dataset: "datasets/sample_correctness.jsonl"
output_dir: "results/"
```

## Dataset Format

JSONL (one JSON object per line):
```json
{"prompt": "What is 2+2?", "expected_answer": "4", "category": "math"}
```

CSV with headers: `prompt`, `expected_answer`, and optionally `category`, `difficulty`, `metadata`.

## Adding a New Model Adapter

Create a file in `llm_eval/models/` that subclasses `BaseModel`:

```python
from llm_eval.models.base import BaseModel, ModelResponse

class MyModel(BaseModel):
    def generate(self, prompt, **kwargs):
        # Your inference logic here
        return ModelResponse(text=..., latency_ms=..., tokens_used=..., metadata={})
```

Then register it in `llm_eval/models/__init__.py`.

## Adding a New Evaluator

Create a file in `llm_eval/evaluators/` that subclasses `BaseEvaluator`:

```python
from llm_eval.evaluators.base import BaseEvaluator, EvalResult

class MyEvaluator(BaseEvaluator):
    def evaluate(self, prompt, expected, response, **kwargs):
        # Your evaluation logic here
        return EvalResult(score=..., passed=..., details={}, metric_name="my_metric")
```

Then register it in `llm_eval/evaluators/__init__.py`.

## CLI Options

```
python -m llm_eval --config CONFIG   Path to YAML config (required)
                   --output-dir DIR  Override output directory
                   --verbose         Enable verbose logging
                   --dry-run         Validate config without running inference
                   --compare         Force comparison mode (auto-enabled with 2+ models)
```

## Output

Each run creates timestamped directories under `results/` containing:
- `results_detailed.json` — per-sample traces with prompts, responses, and all scores
- `results_summary.csv` — aggregated metrics per evaluator

When comparing multiple models, an additional comparison directory is created with:
- `comparison_summary.csv` — side-by-side scores per model per metric
- `comparison_detailed.json` — per-sample responses from all models

## Requirements

- Python 3.10+
- `requests` — HTTP client for model APIs
- `pyyaml` — YAML config parsing
- No ML frameworks required for core operation

## License

MIT
