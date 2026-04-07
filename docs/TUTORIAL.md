# LLM Eval Suite — Beginner's Tutorial

**For AI Test & Evaluation (T&E) Personnel**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Understanding the Configuration File](#4-understanding-the-configuration-file)
5. [Creating Your Own Test Dataset](#5-creating-your-own-test-dataset)
6. [Running Your First Evaluation](#6-running-your-first-evaluation)
7. [Understanding the Evaluators](#7-understanding-the-evaluators)
8. [Comparing Multiple Models](#8-comparing-multiple-models)
9. [Adding Custom Components](#9-adding-custom-components)
10. [Tips for T&E Professionals](#10-tips-for-te-professionals)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Introduction

### What does this tool do?

The **LLM Eval Suite** is a testing framework for large language models (LLMs). Think of it as a structured test harness: you provide a list of questions with known correct answers, point it at a model, and it automatically runs every question, collects the model's responses, and scores the results across multiple dimensions — accuracy, speed, consistency, and more.

The entire evaluation is defined in a plain-text configuration file (YAML format). You don't need to write any Python code to run standard evaluations. You describe what you want to test, run one command, and get back a structured report.

### Who is this for?

This tool is designed for:

- **AI Test & Evaluation (T&E) engineers** who need reproducible, auditable assessments of LLM performance
- **Test engineers** integrating LLMs into mission-critical or regulated applications
- **Technical evaluators** who need to compare models before a deployment decision
- Anyone who needs to validate that an LLM meets defined performance standards

You should be comfortable using a command-line terminal (navigating directories, running commands). You don't need to know Python programming to run evaluations — but if you do, the tool is easy to extend.

### What can you evaluate?

The suite measures five dimensions of LLM performance:

| Dimension | What it answers |
|-----------|-----------------|
| **Correctness** | Does the model produce accurate answers? |
| **Latency** | How fast does the model respond? |
| **Robustness** | Does the model still answer correctly when the question is slightly reworded or contains typos? |
| **Consistency** | Does the model give the same answer every time it's asked the same question? |
| **Cost** | How many tokens does the model consume per request? |

---

## 2. Prerequisites

Before you can use the LLM Eval Suite, you need two things installed on your machine: Python and Ollama.

### Python 3.10 or newer

Python is the programming language this tool is written in. You need version 3.10 or higher.

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download and install the latest stable release for your operating system
3. During installation on Windows, check the box that says **"Add Python to PATH"**

Verify the installation by opening a terminal and running:

```bash
python --version
```

You should see output like:

```
Python 3.12.3
```

If the number shown is 3.10 or higher, you're good. If you see `Python 2.x.x`, try `python3 --version` instead.

### Ollama — for running local models

Ollama is a tool that lets you download and run open-weight language models entirely on your own machine, with no internet connection required after the initial model download. This is essential for air-gapped or secure environments.

1. Go to [https://ollama.com](https://ollama.com) and download the installer for your operating system
2. Run the installer — it will install Ollama and start it as a background service

Verify Ollama is running:

```bash
ollama list
```

If Ollama is running correctly, you'll see a table of installed models (or an empty table if you haven't downloaded any yet).

### Pulling a model

Before running any evaluations, you need to download at least one model. The examples in this tutorial use **qwen3:8b**, a capable open-weight model that runs well on modern laptops and desktops:

```bash
ollama pull qwen3:8b
```

This downloads the model (~5 GB). It only needs to happen once. After that, the model is available offline.

Confirm the model downloaded successfully:

```bash
ollama list
```

Expected output:

```
NAME            ID              SIZE      MODIFIED
qwen3:8b        ...             5.2 GB    2 minutes ago
```

> **Tip:** If you're working in an air-gapped environment, the model download must happen on a connected machine first. Once downloaded, the model files can be transferred and used offline. See the [Ollama documentation](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image) for details on offline deployment.

---

## 3. Installation

### Step 1 — Clone or copy the repository

If you're working from a Git repository:

```bash
git clone https://github.com/your-org/llm-eval-suite.git
cd llm-eval-suite
```

If you received the project as a zip file, extract it and navigate into the folder:

```bash
cd llm-eval-suite
```

### Step 2 — Install dependencies

The tool requires two Python libraries: `requests` (for communicating with Ollama) and `pyyaml` (for reading YAML config files). Install both with one command:

```bash
pip install -r requirements.txt
```

You should see output confirming the packages installed successfully:

```
Successfully installed pyyaml-6.0.2 requests-2.32.3
```

> **Note:** If you're working in an environment with multiple Python versions, you may need to use `pip3` instead of `pip`. If you're in a virtual environment (recommended), make sure it's activated before running this command.

### Step 3 — Verify with a dry run

A **dry run** validates your configuration file and dataset without actually sending any requests to the model. It's a quick sanity check.

```bash
python -m llm_eval --config config/example_eval.yaml --dry-run
```

Expected output:

```
Loading dataset: datasets/sample_correctness.jsonl
Loaded 10 samples.
[dry-run] Config valid. 10 samples, 1 model(s), 2 evaluator(s). No inference will run.
```

If you see that message, your installation is working correctly.

---

## 4. Understanding the Configuration File

Every evaluation run is driven by a YAML configuration file. YAML is a plain-text format for structured data — it's designed to be human-readable. Here's the example configuration that ships with the tool, followed by a line-by-line explanation.

```yaml
run_name: "correctness-eval"

models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"

evaluators:
  - name: "correctness"
    mode: "exact_match"
    threshold: 0.8
  - name: "latency"
    max_ms: 5000

dataset: "datasets/sample_correctness.jsonl"
output_dir: "results/"
```

### `run_name`

```yaml
run_name: "correctness-eval"
```

A human-readable label for this evaluation run. This name becomes part of the output directory name, making it easy to find results later. Use something descriptive like `"correctness-qwen3-8b-v1"` or `"latency-baseline-2026-04"`.

### `models`

```yaml
models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"
```

This section defines which model(s) to evaluate. You can list more than one model — the suite will run the full evaluation for each one.

**`name`** — The exact model identifier. For Ollama models, this matches what you see in `ollama list`. For example: `qwen3:8b`, `llama3.2:3b`, `mistral:7b`.

**`provider`** — Where the model lives. Currently supported providers:
- `"ollama"` — A local model running through Ollama (no internet required during eval)
- `"openai"` — Any OpenAI-compatible API endpoint (requires API key and internet access)

Think of the provider as the "adapter" that tells the tool how to talk to the model. Ollama models use a local REST API; OpenAI-compatible models use their own API format. You don't need to know the details — just set the correct provider name.

**`params`** — Settings passed to the model for every request:

- **`temperature`** — Controls how "creative" or "random" the model's responses are. A value of `0.0` means the model will give the same answer every time for the same input (fully deterministic). A value of `1.0` means more varied and creative outputs. **For T&E evaluations, always use `0.0` unless you're specifically testing variability.**

- **`max_tokens`** — The maximum number of "tokens" (roughly, word fragments) the model can produce in a single response. `512` is a reasonable default for factual Q&A tasks. Increase this for tasks that require longer responses (e.g., `1024` or `2048` for summaries or explanations).

- **`base_url`** — The network address where Ollama is running. The default `http://localhost:11434` is correct for a standard local installation. Only change this if Ollama is running on a different machine or port.

### `evaluators`

```yaml
evaluators:
  - name: "correctness"
    mode: "exact_match"
    threshold: 0.8
  - name: "latency"
    max_ms: 5000
```

The evaluators section lists which tests to run on each model response. You can include multiple evaluators — all of them will run on every sample. Each evaluator is explained in detail in [Section 7](#7-understanding-the-evaluators).

**`name`** — Which evaluator to use. Available options: `correctness`, `latency`, `robustness`.

**Evaluator-specific settings** (these vary by evaluator):
- `mode`: For correctness, which scoring method to use (`exact_match`, `fuzzy_match`, or `llm_judge`)
- `threshold`: The minimum score to count as "passing" (0.0 to 1.0)
- `max_ms`: For latency, the maximum acceptable response time in milliseconds

### `dataset`

```yaml
dataset: "datasets/sample_correctness.jsonl"
```

Path to your test dataset file. Can be a JSONL or CSV file. Paths are relative to the directory you're running the command from. Dataset formats are covered in detail in [Section 5](#5-creating-your-own-test-dataset).

### `output_dir`

```yaml
output_dir: "results/"
```

Where to save results. The suite automatically creates a timestamped subdirectory inside this folder for each run, so results from different runs don't overwrite each other.

---

### Minimal config vs. full config

**Minimal config** — The bare minimum to run an evaluation:

```yaml
run_name: "my-first-eval"

models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0

evaluators:
  - name: "correctness"
    mode: "exact_match"

dataset: "datasets/my_questions.jsonl"
output_dir: "results/"
```

**Full config** — Using all available evaluators and both supported model types:

```yaml
run_name: "full-eval-comparison"

models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"
  - name: "llama3.2:3b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"

evaluators:
  - name: "correctness"
    mode: "fuzzy_match"
    threshold: 0.7
  - name: "latency"
    max_ms: 3000
  - name: "robustness"
    perturbations: ["typo", "case", "rephrase"]
    threshold: 0.8

dataset: "datasets/my_questions.jsonl"
output_dir: "results/"
```

---

## 5. Creating Your Own Test Dataset

The most important part of any evaluation is the dataset — the set of questions (prompts) and correct answers you're testing against. The quality of your dataset directly determines how meaningful your evaluation results are.

### JSONL format (recommended)

JSONL stands for "JSON Lines" — each line of the file is a separate JSON object. This format is easy to create, easy to extend, and handles special characters well.

**Basic structure:**

```jsonl
{"prompt": "What is 2+2?", "expected_answer": "4"}
{"prompt": "What is the capital of France?", "expected_answer": "Paris"}
{"prompt": "What gas do plants absorb during photosynthesis?", "expected_answer": "carbon dioxide"}
```

**With optional fields (recommended for organized evaluations):**

```jsonl
{"prompt": "What is 12 multiplied by 13?", "expected_answer": "156", "category": "math", "difficulty": "easy"}
{"prompt": "What is the square root of 144?", "expected_answer": "12", "category": "math", "difficulty": "easy"}
{"prompt": "What is the capital of France?", "expected_answer": "Paris", "category": "geography", "difficulty": "easy"}
{"prompt": "What is the chemical symbol for water?", "expected_answer": "H2O", "category": "science", "difficulty": "easy"}
{"prompt": "Explain the difference between TCP and UDP in one sentence.", "expected_answer": "TCP is connection-oriented and guarantees delivery; UDP is connectionless and faster but does not guarantee delivery.", "category": "networking", "difficulty": "medium"}
```

**Required fields:**
- `prompt` — The question or instruction sent to the model (must be a non-empty string)
- `expected_answer` — The correct answer you'll score against (must be a non-empty string)

**Optional fields:**
- `category` — A label for grouping samples (e.g., `"math"`, `"reasoning"`, `"policy"`)
- `difficulty` — A label for difficulty level (e.g., `"easy"`, `"medium"`, `"hard"`)
- `metadata` — Any additional information you want to store alongside the result

> **Important:** Each JSON object must be on its own line with no line breaks inside it. A blank line between entries is fine and will be ignored.

### CSV format

If you prefer working in spreadsheets, you can export your dataset as a CSV file. The headers must match the field names exactly.

**Example CSV:**

```csv
prompt,expected_answer,category,difficulty
What is 2+2?,4,math,easy
What is the capital of France?,Paris,geography,easy
What is the chemical symbol for water?,H2O,science,easy
How many sides does a hexagon have?,6,math,easy
What planet is closest to the Sun?,Mercury,astronomy,easy
```

Save this as a `.csv` file and reference it in your config's `dataset` field.

> **Tip:** When creating a CSV in Excel, use "Save As → CSV UTF-8" to preserve special characters correctly.

### Tips for writing good test prompts

**Be specific and unambiguous.** A good test prompt has exactly one correct answer (or a small set of acceptable answers).

- Poor: `"Tell me about the capital of France"`  
- Better: `"What is the capital of France?"`

**Match the expected answer format to the scoring method.** If using `exact_match`, the model's response must exactly equal your `expected_answer` (case-insensitive). So if you expect `"Paris"`, a response of `"The capital of France is Paris."` will fail. For prose-style answers, use `fuzzy_match` instead.

**Write prompts the way a user would actually phrase them.** If you're testing a system that will be used by operators in the field, write prompts using the same language and phrasing those operators would use.

**Group by category.** Using the `category` field lets you analyze performance by topic area. A model might score 95% on math questions but only 70% on ambiguous policy questions — and you'd want to know that.

### Domain-specific examples

**Math and arithmetic:**
```jsonl
{"prompt": "What is 17 multiplied by 23?", "expected_answer": "391", "category": "math"}
{"prompt": "What is 2 raised to the power of 8?", "expected_answer": "256", "category": "math"}
{"prompt": "What is 15% of 200?", "expected_answer": "30", "category": "math"}
```

**Factual knowledge:**
```jsonl
{"prompt": "In what year did World War II end?", "expected_answer": "1945", "category": "history"}
{"prompt": "What is the chemical formula for table salt?", "expected_answer": "NaCl", "category": "science"}
{"prompt": "What is the speed of light in a vacuum?", "expected_answer": "299,792,458 meters per second", "category": "science"}
```

**Reasoning and logic:**
```jsonl
{"prompt": "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops Lazzles?", "expected_answer": "Yes", "category": "reasoning"}
{"prompt": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?", "expected_answer": "5 cents", "category": "reasoning"}
```

**Instructions following:**
```jsonl
{"prompt": "List the primary colors. Respond with a comma-separated list only.", "expected_answer": "red, yellow, blue", "category": "instructions"}
{"prompt": "Convert 100 degrees Fahrenheit to Celsius. Respond with only the number.", "expected_answer": "37.78", "category": "instructions"}
```

**Policy and procedure (domain-specific example):**
```jsonl
{"prompt": "What is the NATO phonetic alphabet word for the letter 'A'?", "expected_answer": "Alpha", "category": "military-knowledge"}
{"prompt": "What does OPSEC stand for?", "expected_answer": "Operations Security", "category": "military-knowledge"}
```

### How many samples do you need?

| Use case | Recommended samples | Notes |
|----------|---------------------|-------|
| Quick sanity check | 10–20 | Just confirming the model works at all |
| Basic evaluation | 50–100 | Enough to get meaningful pass rates |
| Thorough evaluation | 200–500 | Good statistical confidence across categories |
| Rigorous T&E assessment | 500+ | Required for high-stakes deployment decisions |
| Regression testing (CI/CD) | 25–50 | Fast subset to catch regressions quickly |

More samples give you more confidence in your pass rates. With only 10 samples, a pass rate of 80% could easily be 7/10 or 9/10 — a big real-world difference. With 500 samples, 80% reliably means the model succeeds on 400 of 500 cases.

---

## 6. Running Your First Evaluation

This section walks through a complete evaluation from start to finish using the included sample dataset.

### Step 1 — Make sure Ollama is running

Before running any evaluation, confirm Ollama is active:

```bash
ollama list
```

You should see your downloaded models. If you get an error, start Ollama manually:

- **macOS/Linux:** Open Ollama from the Applications folder, or run `ollama serve` in a terminal
- **Windows:** Start the Ollama application from the Start menu

### Step 2 — Review your config

Open `config/example_eval.yaml` in a text editor and confirm it matches what you want to run. For this walkthrough, we'll use it as-is.

### Step 3 — Run the evaluation

From the `llm-eval-suite` directory:

```bash
python -m llm_eval --config config/example_eval.yaml
```

### What you'll see on screen

As the evaluation runs, you'll see progress updates followed by a summary table:

```
Loading dataset: datasets/sample_correctness.jsonl
Loaded 10 samples.

=== Model: qwen3:8b (provider: ollama) ===
Evaluating sample 1/10...
Evaluating sample 2/10...
Evaluating sample 3/10...
Evaluating sample 4/10...
Evaluating sample 5/10...
Evaluating sample 6/10...
Evaluating sample 7/10...
Evaluating sample 8/10...
Evaluating sample 9/10...
Evaluating sample 10/10...

  Evaluation Summary
  -------------------------------------------------------
  Metric                  Mean Score   Pass Rate   Samples
  -------------------------------------------------------
  correctness_exact_match     0.7000      70.0%        10
  latency                     0.8234      90.0%        10
  -------------------------------------------------------
  Total samples evaluated: 10
  Results saved to: results/correctness-eval_qwen3_8b_20260407_141523
```

**Reading this table:**
- **Mean Score** — The average score across all samples, from 0.0 (worst) to 1.0 (perfect)
- **Pass Rate** — What percentage of samples met or exceeded the threshold defined in your config
- **Samples** — Total number of dataset samples evaluated

### Step 4 — Find your results files

The tool creates a timestamped directory under `results/` every time you run an evaluation. In the example above, the results are in:

```
results/correctness-eval_qwen3_8b_20260407_141523/
├── results_detailed.json
└── results_summary.csv
```

The timestamp (`20260407_141523`) means April 7, 2026 at 14:15:23 — so you always know exactly when a run was performed.

### How to read `results_detailed.json`

This file contains the full trace for every sample. Open it in any text editor or JSON viewer. Here's one sample entry:

```json
[
  {
    "sample_idx": 0,
    "prompt": "What is 12 multiplied by 13?",
    "expected": "156",
    "model_response": "156",
    "latency_ms": 842.3,
    "evaluations": [
      {
        "metric_name": "correctness_exact_match",
        "score": 1.0,
        "passed": true,
        "details": {
          "mode": "exact_match",
          "expected": "156",
          "response": "156"
        }
      },
      {
        "metric_name": "latency",
        "score": 0.8316,
        "passed": true,
        "details": {
          "latency_ms": 842.3,
          "max_ms": 5000
        }
      }
    ]
  }
]
```

Each entry shows you:
- The exact prompt that was sent
- What the model responded
- How long it took (in milliseconds)
- A score and pass/fail result for each evaluator

This file is your **audit trail** — it captures every input and output, making evaluations reproducible and inspectable.

### How to read `results_summary.csv`

This file is designed for quick analysis in Excel or any spreadsheet program:

```csv
metric_name,mean_score,pass_rate,sample_count
correctness_exact_match,0.7000,0.7000,10
latency,0.8234,0.9000,10
```

**Columns:**
- `metric_name` — Which evaluator produced this row
- `mean_score` — Average score (0.0–1.0) across all samples
- `pass_rate` — Fraction of samples that passed (0.0–1.0; multiply by 100 for percentage)
- `sample_count` — Number of samples this metric was computed for

---

## 7. Understanding the Evaluators

This section explains each evaluator in depth: what it measures, when to use it, how to configure it, and how to interpret the results.

---

### Correctness

**What it measures:** Whether the model's answer matches the expected (ground-truth) answer.

This is the most fundamental evaluator — it tells you if the model is actually answering questions correctly. There are three modes, suited to different types of tasks.

---

#### Mode 1: Exact Match

**What it does:** Compares the model's response character-by-character to the expected answer. The comparison is case-insensitive and strips leading/trailing whitespace. A score of `1.0` means they match exactly; `0.0` means they don't.

**When to use it:**
- Mathematical calculations (`"156"`, `"12"`, `"0.75"`)
- Factual lookups with a single definitive answer (`"Paris"`, `"H2O"`, `"1945"`)
- Code outputs where the exact string matters
- Classification tasks where the answer must be one of a fixed set of labels

**When NOT to use it:**
- Summaries, explanations, or any open-ended prose — the model may be entirely correct but phrased differently
- Questions where multiple phrasings of the answer are acceptable

**Config:**
```yaml
evaluators:
  - name: "correctness"
    mode: "exact_match"
    threshold: 0.8
```

**Interpreting scores:**
- `1.0` — Exact match (passes if threshold ≤ 1.0)
- `0.0` — No match (fails)

The threshold in exact match mode effectively sets the required pass rate. Since every individual score is either 0 or 1, the mean score equals the fraction of samples answered correctly. A threshold of `0.8` means a sample passes if and only if it's a perfect match, and your overall pass rate will be the fraction of perfectly-matched samples.

---

#### Mode 2: Fuzzy Match

**What it does:** Measures text similarity between the response and the expected answer using sequence matching. Returns a score between `0.0` (completely different) and `1.0` (identical). This method counts how many characters are shared between the two strings, in order.

**When to use it:**
- Tasks where the answer might include slight rephrasing (`"The capital is Paris"` vs. `"Paris"`)
- Explanations or definitions where the core content matters more than exact wording
- Tasks where you'd grade a partial answer as partially correct

**Config:**
```yaml
evaluators:
  - name: "correctness"
    mode: "fuzzy_match"
    threshold: 0.7
```

**Interpreting scores:**
- `1.0` — Identical text
- `0.8–0.99` — Very similar, minor wording differences
- `0.5–0.79` — Partially similar, some correct content mixed with extra text
- `< 0.5` — Largely different

A threshold of `0.7` is a reasonable starting point. If your expected answers are brief (one word or short phrase) and you get a lot of false passes, tighten to `0.8`. If your expected answers are longer prose and legitimate paraphrasing keeps failing, loosen to `0.6`.

---

#### Mode 3: LLM-as-Judge

**What it does:** Uses a second LLM to grade the model's response on a scale of 1–5 (which is then normalized to 0.0–1.0). The judge is given the original question, the expected answer, and the model's response, and asked to rate the correctness.

**When to use it:**
- Open-ended questions where correctness is nuanced (`"Explain the tradeoffs between X and Y"`)
- Tasks where you want to assess quality of reasoning, not just factual accuracy
- Situations where exact or fuzzy match would produce too many false positives or false negatives

**Important caveats:**
- This mode requires an inference call to a judge model for every sample, making it slower and more expensive
- By default, the same model being evaluated acts as its own judge — this can be biased. Use a separate, trusted model as the judge when possible
- Results can vary slightly between runs unless the judge model is also set to `temperature: 0.0`

**Config (using the same model as judge):**
```yaml
evaluators:
  - name: "correctness"
    mode: "llm_judge"
    threshold: 0.75
```

**Config (using a separate judge model — advanced):**
```yaml
evaluators:
  - name: "correctness"
    mode: "llm_judge"
    threshold: 0.75
    judge_provider: "ollama"
    judge_model_config:
      name: "llama3.2:3b"
      provider: "ollama"
      params:
        temperature: 0.0
        max_tokens: 32
        base_url: "http://localhost:11434"
```

**Interpreting scores:**
- The 1–5 judge rating is normalized to `0.0–1.0` (a rating of 5 becomes 1.0, a rating of 1 becomes 0.0)
- A threshold of `0.75` corresponds roughly to a judge rating of 4 out of 5

---

### Latency

**What it measures:** How long the model takes to respond to each prompt.

Response time is measured from the moment the request is sent to when the full response is received, in milliseconds (ms). This evaluator tracks not just pass/fail, but also the statistical distribution of latency across all samples.

**When to use it:**
- Whenever the model is part of a real-time or interactive system
- When comparing models: a faster model that's slightly less accurate might be preferable for time-sensitive applications
- For establishing performance baselines before deployment

**Config:**
```yaml
evaluators:
  - name: "latency"
    max_ms: 5000
```

`max_ms` is your threshold — the maximum acceptable response time in milliseconds. Common values:

| Use Case | Suggested `max_ms` |
|----------|-------------------|
| Interactive chatbot | 2000–3000 |
| Analyst decision support | 5000–10000 |
| Batch processing (overnight) | 30000–60000 |
| Stress testing | 120000 (2 minutes) |

**Interpreting scores:**

The score for each sample is calculated as:

```
score = 1.0 - (actual_latency_ms / max_ms)
```

So a response that came in exactly at the threshold gets a score of `0.0`, and a response in half the allowed time gets `0.5`. A very fast response gets closer to `1.0`.

**In the summary table**, the pass rate tells you what fraction of responses came in under your `max_ms` threshold.

**In `results_detailed.json`**, each sample record includes the exact latency in milliseconds.

The suite also computes the following statistics over all samples:
- **mean** — Average response time
- **median** — The midpoint (half of responses were faster, half slower)
- **p95** — 95th percentile (95% of responses were faster than this)
- **p99** — 99th percentile (99% of responses were faster than this)

P95 and P99 are the most important for reliability assessments. If your p95 is 4800ms and your threshold is 5000ms, you know that under normal conditions 5% of users will be waiting near the limit.

---

### Robustness

**What it measures:** Whether the model still answers correctly when the question is slightly altered — through typos, different capitalization, or word substitutions.

A model that aces a clean test set but fails when a user misspells a word, or phrases the question differently, is not reliable for operational use. Robustness testing catches this brittleness.

**Why this matters:** Real users don't always write perfect, clean queries. Field operators under stress may make typos. Adversaries may intentionally perturb inputs to cause failures. Robustness testing gives you confidence (or evidence of lack thereof) about how the model handles imperfect input.

**How it works:** For each sample, the evaluator:
1. Records the model's score on the original (clean) prompt as a baseline
2. Creates three "perturbed" versions of the prompt
3. Sends each perturbed prompt to the model
4. Computes the mean score across all perturbed versions
5. Reports the ratio: `mean perturbed score / baseline score`

A robustness score of `1.0` means the model performs exactly as well on perturbed inputs as on clean inputs. A score of `0.7` means the model performs 30% worse when inputs are slightly garbled.

**The three perturbation types:**

| Type | What it does | Example |
|------|-------------|---------|
| `typo` | Randomly inserts, deletes, or swaps characters in 2–3 words | `"What is the captial of Frnce?"` |
| `case` | Randomly changes word capitalization | `"WHAT is the Capital OF france?"` |
| `rephrase` | Swaps some words for synonyms and rearranges adjacent word pairs | `"What is the capital of France?"` → `"What is the capital of France?"` (with synonym swaps where applicable) |

**Config:**
```yaml
evaluators:
  - name: "robustness"
    perturbations: ["typo", "case", "rephrase"]
    threshold: 0.8
```

You can run a subset of perturbations:
```yaml
evaluators:
  - name: "robustness"
    perturbations: ["typo"]   # only test typo robustness
    threshold: 0.75
```

**Interpreting scores:**
- `1.0` — Fully robust: perturbed performance equals baseline
- `0.8–0.99` — Good robustness: minor degradation from perturbations
- `0.6–0.79` — Moderate fragility: noticeably sensitive to input variations
- `< 0.6` — High fragility: significant performance drop on perturbed inputs

> **Note:** Robustness testing requires additional model inference calls (one per perturbation type per sample). A 100-sample dataset with 3 perturbation types will result in 400 total inference calls (100 baseline + 300 perturbed).

---

### Consistency

**What it measures:** Whether the model gives the same answer every time you ask the same question.

For mission-critical applications, you don't just need a model that's correct — you need a model whose behavior is predictable and repeatable. A model that gives the right answer 80% of the time and a wrong answer 20% of the time (even when asked identically) is not safe to rely on.

**Why consistency matters:**
- **Auditability:** Reviewers need to be able to reproduce the model's outputs
- **Reliability:** Operators need to trust that the system behaves the same way today as it did last week
- **Testing validity:** If results vary between runs, your test results aren't meaningful

**How to test consistency:** Run the same evaluation twice with `temperature: 0.0` (deterministic mode). If both runs produce identical outputs, the model is fully consistent. Any differences indicate non-determinism in the model or infrastructure.

**Temperature and consistency:**
- `temperature: 0.0` — Fully deterministic; the model always picks the single most likely next token
- `temperature: 0.1–0.3` — Very low variation; suitable for tasks where slight variation is acceptable
- `temperature: 0.7–1.0` — High variation; suitable for creative tasks, not for T&E evaluations

**How many runs to compare:** For a formal consistency assessment, run the same config at least 3–5 times and compare the `results_summary.csv` files. If mean scores vary by more than 1–2%, investigate whether `temperature: 0.0` is actually being respected by the model.

**Practical config for consistency testing:**
```yaml
run_name: "consistency-test-run-1"

models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0   # Critical for consistency
      max_tokens: 512
      base_url: "http://localhost:11434"

evaluators:
  - name: "correctness"
    mode: "exact_match"
    threshold: 0.8

dataset: "datasets/my_questions.jsonl"
output_dir: "results/"
```

Run this 3 times, changing `run_name` each time (`run-1`, `run-2`, `run-3`). Compare the `results_summary.csv` files from each run. Identical mean scores across all runs confirms deterministic behavior.

---

### Cost

**What it measures:** How many tokens the model consumes per request, which directly determines cloud API costs and informs hardware sizing for local deployments.

**What is a token?** Language models don't read text character-by-character — they break text into "tokens," which are roughly word fragments. The word "evaluation" is one token; "evaluations" might be two (`evaluat` + `ions`). As a rough rule of thumb, 1 token ≈ 0.75 English words, so 100 words ≈ 130–140 tokens.

**Why token tracking matters:**
- **Cloud models (e.g., OpenAI):** You pay per token. Knowing average tokens per request lets you forecast costs before scaling up.
- **Local models (Ollama):** You're not paying per token, but token throughput determines how many requests your hardware can handle per minute.
- **Air-gapped deployments:** Knowing token count helps size hardware (GPU memory, inference throughput).

**Where to find token data:** Token usage is captured in `results_detailed.json` for each sample. Look for `tokens_used` in the model response metadata:

```json
{
  "sample_idx": 0,
  "prompt": "What is 12 multiplied by 13?",
  "model_response": "156",
  "latency_ms": 842.3,
  ...
}
```

The Ollama adapter specifically captures:
- `eval_count` — Output tokens (the model's response)
- `prompt_eval_count` — Input tokens (your prompt)

**Cost estimation for cloud models:**

To estimate costs when using an OpenAI-compatible endpoint, use the token counts from a sample run against the provider's pricing:

```
Estimated cost = (total_input_tokens × input_price_per_1k) + (total_output_tokens × output_price_per_1k)
```

**Setting token budgets:** Use `max_tokens` in your model config to cap response length:

```yaml
models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 128    # Short answers only — saves tokens for factual Q&A
```

**Local vs. cloud cost comparison:**

| Deployment | Cost model | Key metric |
|------------|-----------|------------|
| Ollama (local) | Hardware + electricity | Tokens/second throughput |
| OpenAI API | $/million tokens | Cost per 1000 samples |
| Self-hosted cloud VM | VM cost + GPU | Cost amortized over usage |

For a rough local cost comparison: if a local GPU can process 50 tokens/second and your average response is 100 tokens, you can handle ~30 requests per minute. Compare that against your operational throughput requirements.

---

## 8. Comparing Multiple Models

One of the most powerful uses of this tool is head-to-head model comparison. You can evaluate multiple models in a single config file, and results are saved separately for each model.

### Setting up a multi-model comparison

Add multiple entries under `models`:

```yaml
run_name: "model-comparison-apr2026"

models:
  - name: "qwen3:8b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"

  - name: "llama3.2:3b"
    provider: "ollama"
    params:
      temperature: 0.0
      max_tokens: 512
      base_url: "http://localhost:11434"

evaluators:
  - name: "correctness"
    mode: "exact_match"
    threshold: 0.8
  - name: "latency"
    max_ms: 5000

dataset: "datasets/sample_correctness.jsonl"
output_dir: "results/"
```

Run it:

```bash
python -m llm_eval --config config/model-comparison-apr2026.yaml
```

The terminal will print a summary for each model sequentially:

```
=== Model: qwen3:8b (provider: ollama) ===
Evaluating sample 1/10...
...

  Evaluation Summary
  -------------------------------------------------------
  Metric                  Mean Score   Pass Rate   Samples
  -------------------------------------------------------
  correctness_exact_match     0.9000      90.0%        10
  latency                     0.8100      90.0%        10
  -------------------------------------------------------
  Total samples evaluated: 10
  Results saved to: results/model-comparison-apr2026_qwen3_8b_20260407_141523


=== Model: llama3.2:3b (provider: ollama) ===
Evaluating sample 1/10...
...

  Evaluation Summary
  -------------------------------------------------------
  Metric                  Mean Score   Pass Rate   Samples
  -------------------------------------------------------
  correctness_exact_match     0.7000      70.0%        10
  latency                     0.9300     100.0%        10
  -------------------------------------------------------
  Total samples evaluated: 10
  Results saved to: results/model-comparison-apr2026_llama3.2_3b_20260407_141602
```

### Reading the comparison table

Compile a comparison by opening both `results_summary.csv` files side by side:

| Metric | qwen3:8b | llama3.2:3b |
|--------|----------|-------------|
| correctness_exact_match | 90% pass | 70% pass |
| latency mean score | 0.81 | 0.93 |
| latency pass rate | 90% | 100% |

### Making a decision based on comparison data

There is rarely a single "best" model — the right choice depends on your requirements:

**Accuracy-first use case** (e.g., a knowledge retrieval system for technical reference):
- Choose **qwen3:8b**: 90% correctness pass rate vs. 70% for llama3.2:3b
- The 20-point accuracy advantage outweighs the slower speed

**Latency-first use case** (e.g., a real-time assistant with a 3-second response requirement):
- Choose **llama3.2:3b**: 100% latency pass rate vs. 90% for qwen3:8b
- Speed matters more if the accuracy gap is acceptable for the application

**Practical example — fast small model vs. slow large model:**

A common comparison is between a quantized 3B-parameter model and an 8B-parameter model:

- The **3B model** is faster (often 2–3x) and uses less memory, making it viable on CPU-only hardware or constrained edge devices
- The **8B model** is typically more accurate, better at reasoning, and handles more complex instructions

Use the eval suite to find the crossover point for your specific task:
1. Build a dataset representative of your actual operational queries
2. Run both models with `temperature: 0.0`
3. Check correctness pass rates by category — the 3B model may be adequate for simple queries even if it lags on complex ones

---

## 9. Adding Custom Components

### Adding a new model adapter

If you need to connect to a model that isn't Ollama or OpenAI-compatible, you can create a new adapter.

**Step 1 — Create a new file** in `llm_eval/models/`:

```python
# llm_eval/models/my_model.py

import time
import requests
from llm_eval.models.base import BaseModel, ModelResponse


class MyModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config["name"], config.get("params", {}))
        self.endpoint = self.params.get("endpoint", "http://my-model-api/generate")

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        payload = {"prompt": prompt, "max_tokens": self.params.get("max_tokens", 512)}

        start = time.perf_counter()
        resp = requests.post(self.endpoint, json=payload, timeout=120)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        return ModelResponse(
            text=data["text"],
            latency_ms=latency_ms,
            tokens_used=data.get("tokens_used"),
            metadata={}
        )
```

**Step 2 — Register it** in `llm_eval/runner.py`:

Find the `_get_model_class` function and add your new provider:

```python
def _get_model_class(provider: str):
    if provider == "ollama":
        from llm_eval.models.ollama_model import OllamaModel
        return OllamaModel
    elif provider == "openai":
        from llm_eval.models.openai_model import OpenAIModel
        return OpenAIModel
    elif provider == "my_provider":          # <-- Add this
        from llm_eval.models.my_model import MyModel
        return MyModel
    else:
        raise ValueError(f"Unknown model provider: '{provider}'")
```

**Step 3 — Use it** in your YAML config:

```yaml
models:
  - name: "my-custom-model"
    provider: "my_provider"
    params:
      endpoint: "http://192.168.1.50:8080/generate"
      temperature: 0.0
      max_tokens: 512
```

### Adding a new evaluator

If you need to measure something the built-in evaluators don't cover (for example, a domain-specific safety check, or a toxicity classifier), you can add a custom evaluator.

**Step 1 — Create a new file** in `llm_eval/evaluators/`:

```python
# llm_eval/evaluators/my_evaluator.py

from llm_eval.evaluators.base import BaseEvaluator, EvalResult


class MyEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.8)

    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        # Your scoring logic here
        # score must be a float between 0.0 and 1.0
        score = self._my_scoring_function(expected, response)

        return EvalResult(
            score=score,
            passed=score >= self.threshold,
            details={"expected": expected, "response": response},
            metric_name="my_metric"
        )

    def _my_scoring_function(self, expected: str, response: str) -> float:
        # Replace with your logic
        # Example: check if a required keyword appears in the response
        required_keyword = expected.lower()
        return 1.0 if required_keyword in response.lower() else 0.0
```

**Step 2 — Register it** in `llm_eval/runner.py`:

Find the `_get_evaluator_class` function and add your evaluator:

```python
def _get_evaluator_class(name: str):
    if name == "correctness":
        from llm_eval.evaluators.correctness import CorrectnessEvaluator
        return CorrectnessEvaluator
    elif name == "latency":
        from llm_eval.evaluators.latency import LatencyEvaluator
        return LatencyEvaluator
    elif name == "robustness":
        from llm_eval.evaluators.robustness import RobustnessEvaluator
        return RobustnessEvaluator
    elif name == "my_evaluator":           # <-- Add this
        from llm_eval.evaluators.my_evaluator import MyEvaluator
        return MyEvaluator
    else:
        raise ValueError(f"Unknown evaluator: '{name}'")
```

**Step 3 — Use it** in your YAML config:

```yaml
evaluators:
  - name: "my_evaluator"
    threshold: 0.8
```

### Adding new datasets

No code changes needed — just create a new `.jsonl` or `.csv` file following the format described in [Section 5](#5-creating-your-own-test-dataset), place it in the `datasets/` folder, and reference it in your config:

```yaml
dataset: "datasets/my_new_dataset.jsonl"
```

---

## 10. Tips for T&E Professionals

### Always set temperature to 0 for evaluations

```yaml
params:
  temperature: 0.0
```

Unless you are specifically testing how a model behaves with stochastic outputs, always use `temperature: 0.0`. This ensures:
- Identical inputs produce identical outputs (reproducible results)
- Re-running the same eval produces the same scores
- Your results can be independently verified

### Make your run names descriptive and include versions

Good run names create a self-documenting results directory:

```
correctness-qwen3-8b-v2.1-dataset-v4_20260407_141523/
latency-llama3.2-3b-baseline_20260407_152000/
robustness-mistral-7b-typo-only_20260408_090000/
```

Bad run names make it hard to find results later:

```
test_20260407_141523/
run1_20260407_152000/
```

### Document your evaluation methodology

Before running a formal evaluation, write down:
1. **Objective** — What capability are you assessing?
2. **Dataset** — How was it constructed? What domains does it cover? Who wrote the questions?
3. **Evaluator configuration** — Which evaluators, which modes, which thresholds?
4. **Model versions** — Exact model names (including version tags like `:8b`, `:7b-q4_K_M`)
5. **Hardware** — CPU/GPU/RAM of the machine used
6. **Date and operator** — When it was run and by whom

This documentation, combined with the timestamped results files, creates a complete audit trail.

### Archive results for audit trails

The `results/` directory is your evidence base. Never delete or overwrite it. Consider:
- Committing the `results/` directory to a Git repository for version-controlled history
- Archiving results to a shared network drive with folder-level write protection
- Including the full `results_detailed.json` in any formal evaluation report (it contains every input/output pair)

### Running evaluations in air-gapped environments

This tool is designed for fully offline use with Ollama:

1. **Download models** on a connected machine: `ollama pull qwen3:8b`
2. **Copy the model files** to the air-gapped machine (Ollama stores models in `~/.ollama/models/` on Linux/Mac, `C:\Users\<user>\.ollama\models\` on Windows)
3. **Install dependencies** from a local package mirror or pre-downloaded wheel files:
   ```bash
   pip install --no-index --find-links ./offline_packages requests pyyaml
   ```
4. **Run evaluations** exactly as described in this tutorial — no internet access needed

### Scaling up: more samples, more models, automated pipelines

**More samples:** Scale up your dataset by adding samples to your JSONL file. The tool processes them sequentially. A 500-sample dataset with 3 evaluators takes roughly 3× longer than a 500-sample dataset with 1 evaluator.

**More models:** Add models to the `models` list in your config. Each model runs through the full dataset sequentially.

**Automated pipelines:** Schedule evaluations on a regular cadence using cron (Linux/Mac) or Task Scheduler (Windows):

```bash
# Example cron job: run nightly at 2 AM
0 2 * * * cd /path/to/llm-eval-suite && python -m llm_eval --config config/nightly_eval.yaml
```

For CI/CD integration, add a step to your pipeline that runs the evaluation and checks the summary CSV for pass rate thresholds:

```bash
python -m llm_eval --config config/regression_eval.yaml
# Then check results_summary.csv — fail the pipeline if pass_rate < 0.9
```

### Keeping evaluations comparable across time

When running the same evaluation at different points in time (e.g., to track model improvement after fine-tuning), keep these constant:
- The exact dataset (same file, no modifications)
- The exact config (same evaluators, thresholds, parameters)
- The same hardware (latency results are hardware-dependent)
- `temperature: 0.0`

Change only the model version being evaluated.

---

## 11. Troubleshooting

### Error: "Could not connect to Ollama"

```
RuntimeError: Could not connect to Ollama at http://localhost:11434.
Make sure Ollama is running (`ollama serve`).
```

**Fix:** Ollama is not running. Start it:
- **macOS/Windows:** Launch the Ollama application from your Applications/Start menu
- **Linux:** Run `ollama serve` in a separate terminal window

Verify Ollama is running: `ollama list`

---

### Error: "Model not found" or HTTP 404

```
RuntimeError: Ollama API error 404: {"error":"model 'qwen3:8b' not found"}
```

**Fix:** The model hasn't been downloaded. Pull it first:

```bash
ollama pull qwen3:8b
```

Then verify it's available:

```bash
ollama list
```

The model name in your config must exactly match the name shown in `ollama list`.

---

### Error: "Config file not found"

```
Error: Config file not found: config/my_eval.yaml
```

**Fix:** Check two things:
1. You're running the command from the `llm-eval-suite` directory (use `pwd` on Mac/Linux or `cd` on Windows to check)
2. The file path in `--config` is correct and the file exists

---

### Error: "Failed to parse YAML config"

```
Error: Failed to parse YAML config 'config/my_eval.yaml': ...
```

**Fix:** Your YAML file has a formatting error. Common causes:
- **Incorrect indentation** — YAML uses spaces (not tabs) for indentation. Every nested level must be indented consistently (2 or 4 spaces per level)
- **Missing colon** — Every key must be followed by `:` and a space: `key: value`
- **Unquoted special characters** — If your `run_name` contains `:`  or `#`, wrap it in quotes: `run_name: "eval:v1"`

Use a free YAML validator online (such as [https://www.yamllint.com](https://www.yamllint.com)) to check your file before running.

---

### Error: "Dataset file not found"

```
FileNotFoundError: Dataset file not found: datasets/my_questions.jsonl
```

**Fix:** The path in your config's `dataset` field doesn't point to a real file. Check:
1. The file exists: `ls datasets/` (Mac/Linux) or `dir datasets\` (Windows)
2. The filename is spelled correctly (paths are case-sensitive on Linux)
3. You're running from the `llm-eval-suite` directory

---

### Error: "Invalid JSON on line N"

```
ValueError: Invalid JSON on line 3 of datasets/my_questions.jsonl: ...
```

**Fix:** Line 3 of your dataset file has a JSON formatting error. Common mistakes:
- Missing closing `}` on a line
- Using single quotes `'` instead of double quotes `"`
- A trailing comma after the last key-value pair: `{"prompt": "...", "expected_answer": "...",}` (remove the trailing comma)

Paste the problematic line into [https://jsonlint.com](https://jsonlint.com) to identify the exact error.

---

### Error: "Row N is missing required field(s)"

```
ValueError: Row 4 in 'datasets/my_questions.jsonl' is missing required field(s): ['expected_answer'].
Each row must have: ['expected_answer', 'prompt'].
```

**Fix:** One of your dataset rows is missing either `prompt` or `expected_answer`. Every row must have both. Check the file for typos in field names — `"expected_answer"` must be spelled exactly that way (case-sensitive).

---

### Error: "Unknown model provider"

```
ValueError: Unknown model provider: 'olama'. Supported: ollama, openai
```

**Fix:** Check the `provider` field in your config for typos. Currently supported values are exactly `"ollama"` and `"openai"` (lowercase).

---

### Error: "Unknown evaluator"

```
ValueError: Unknown evaluator: 'corectness'. Supported: correctness, latency, robustness
```

**Fix:** Check the `name` field under your evaluator for typos. Supported values are `"correctness"`, `"latency"`, and `"robustness"`.

---

### Error: "Request timed out after 120 seconds"

```
RuntimeError: Request to Ollama timed out after 120 seconds (model=qwen3:8b).
```

**Fix:** The model took longer than 2 minutes to respond. This usually happens when:
- The model is loading for the first time (subsequent requests will be faster)
- Your hardware is under heavy load
- `max_tokens` is set very high and the model is generating a very long response

Try running one sample manually in Ollama to confirm the model responds at all:

```bash
ollama run qwen3:8b "What is 2+2?"
```

If it responds, the issue is likely the first-load delay. Run the eval again — subsequent calls will be faster.

---

### Results look wrong: all scores are 0.0

If you're using `exact_match` and seeing 0.0 scores across the board, the model's responses likely contain more text than just the answer. For example:

- **Expected:** `"156"`
- **Model responded:** `"The answer is 156."`

These are not exact matches. Try one of these fixes:

1. **Switch to `fuzzy_match`** — it will recognize that `"156"` appears within the response
2. **Rephrase your prompts** to instruct the model to respond with only the answer: `"What is 12 multiplied by 13? Respond with only the number."`
3. **Use `llm_judge`** mode if you can't control the response format

---

### Getting more information about errors

Run the evaluation with `--verbose` to get full stack traces and per-sample debug output:

```bash
python -m llm_eval --config config/example_eval.yaml --verbose
```

This will print detailed information for each sample, including evaluator scores and any internal errors.

---

### Where to get help

- **Review the README:** `/llm-eval-suite/README.md` has a quick reference for CLI options and dataset format
- **Check existing configs:** The example config at `config/example_eval.yaml` is a working template
- **Examine the source:** The evaluator code in `llm_eval/evaluators/` is short and readable — if you want to understand exactly how a score is computed, reading the code is the fastest way

---

*Last updated: April 2026*
