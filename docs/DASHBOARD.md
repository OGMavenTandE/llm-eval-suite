# LLM Eval Dashboard

## Overview

`dashboard.html` is a single-file interactive viewer for LLM evaluation results produced by this suite. It provides visual reporting — KPI cards, charts, per-sample drill-downs, and side-by-side model comparisons — entirely within a self-contained HTML file.

It exists for T&E personnel working on air-gapped networks where installing web servers, Python packages, or cloud-based reporting tools is not an option. You double-click the file, load a JSON result, and have a fully interactive report. There are zero external dependencies: no CDN links, no npm packages, no server required.

---

## Features

### Tabs

| Tab | Description |
|-----|-------------|
| **Summary** | KPI cards (total samples, pass rate, mean score, metrics count) and a sortable metrics table with color-coded score bars. In comparison mode, KPIs are broken out per model. |
| **Charts** | An SVG horizontal bar chart of mean scores per metric, and an SVG radar/spider chart showing the overall score profile. Both charts update when you toggle dark/light mode. |
| **Samples** | Per-sample drill-down table. Click any row to expand it and see the full prompt, full model response, and per-evaluator scores with PASS/FAIL badges. Supports text search (by prompt) and pass/fail filtering. |
| **Comparison** | Side-by-side model comparison table (best scores highlighted per metric), plus a per-sample table showing each model's response and metric dots side by side. Only visible when a `comparison_detailed.json` file is loaded. |

### Additional capabilities

- **KPI cards** — at-a-glance metrics for every loaded model
- **Sortable metric table** — click any column header to sort ascending/descending
- **Color-coded score bars** — green ≥ 0.8, yellow ≥ 0.5, red < 0.5
- **SVG bar chart** — grouped bars for multi-model comparison; single bars for single-model runs
- **SVG radar/spider chart** — polygon overlay per model (requires ≥ 3 metrics)
- **Expandable sample rows** — click a row in the Samples tab to reveal the full prompt, response, and evaluation detail chips
- **Search and pass/fail filtering** — live search on prompt text; filter buttons for All / Passed / Failed
- **Side-by-side model comparison** — best-scoring model per sample highlighted; per-metric winner starred in expanded rows
- **Dark/light mode toggle** — sun/moon icon in the header; preference persists via `localStorage`

---

## Supported Data Formats

The dashboard reads JSON files produced by the eval suite's reporter. It auto-detects the format on load based on whether the first record has a `models` key.

### `results_detailed.json` — Single model

An array of sample records. Each record contains the raw prompt, expected answer, model response, latency, and an `evaluations` array.

```json
[
  {
    "sample_idx": 0,
    "prompt": "What is 12 multiplied by 13?",
    "expected": "156",
    "model_response": "156",
    "latency_ms": 2633.7,
    "evaluations": [
      {
        "metric_name": "correctness_fuzzy_match",
        "score": 1.0,
        "passed": true,
        "details": {
          "mode": "fuzzy_match",
          "expected": "156",
          "response": "156"
        }
      },
      {
        "metric_name": "latency",
        "score": 0.4733,
        "passed": true,
        "details": {
          "latency_ms": 2633.7,
          "max_ms": 5000
        }
      }
    ]
  }
]
```

### `comparison_detailed.json` — Multi-model

The same structure, but instead of a flat `evaluations` array at the top level, each record has a `models` object keyed by model name. Each model entry contains its own `response`, `latency_ms`, and `evaluations`.

```json
[
  {
    "sample_idx": 0,
    "prompt": "What is 12 multiplied by 13?",
    "expected": "156",
    "models": {
      "qwen3:8b": {
        "response": "156",
        "latency_ms": 2633.7,
        "evaluations": [
          {
            "metric_name": "correctness_fuzzy_match",
            "score": 1.0,
            "passed": true,
            "details": { "mode": "fuzzy_match", "expected": "156", "response": "156" }
          },
          {
            "metric_name": "latency",
            "score": 0.4733,
            "passed": true,
            "details": { "latency_ms": 2633.7, "max_ms": 5000 }
          }
        ]
      },
      "llama3:8b": {
        "response": "156",
        "latency_ms": 1841.2,
        "evaluations": [ ... ]
      }
    }
  }
]
```

**Auto-detection:** if `json[0].models` exists, the dashboard enters comparison mode and shows the Comparison tab. Otherwise it runs in single-model mode.

---

## How to Use

1. **Open the dashboard** — double-click `dashboard.html`, or navigate to it in your browser via a `file://` URL.
2. **Load a result file** — click the **Load JSON** button in the top-right header.
3. **Select a file** — choose a `results_detailed.json` or `comparison_detailed.json` from your eval run's output directory. The file name and detected mode (single / comparison) will appear next to the button.
4. **Navigate tabs** — use the **Summary**, **Charts**, **Samples**, and **Comparison** tabs to explore results.
5. **Drill into samples** — on the Samples tab, click any row to expand it and see the full prompt, full response, and per-evaluator detail.
6. **Filter and search** — use the search box to filter rows by prompt text; use the **All / Passed / Failed** buttons to narrow by outcome.
7. **Toggle theme** — click the sun/moon icon in the top-right to switch between dark and light mode. Your preference is saved automatically.

---

## Where to Find Result Files

The eval suite writes results to timestamped directories under `results/` in the project root:

```
results/
├── {run_name}_{timestamp}/
│   ├── results_detailed.json       ← load this for single-model view
│   └── results_summary.csv
│
└── {run_name}_comparison_{timestamp}/
    ├── comparison_detailed.json    ← load this for multi-model comparison view
    └── comparison_summary.csv
```

**Example paths:**

```
results/my-eval_20260407_150312/results_detailed.json
results/my-eval_comparison_20260407_150312/comparison_detailed.json
```

Both paths are printed to stdout at the end of every eval run.

---

## Air-Gapped Compatibility

The dashboard is designed specifically for restricted environments:

- **Single file** — the entire application is ~50 KB of HTML, CSS, and JavaScript in one file
- **Zero network calls** — no external requests of any kind; all assets are inlined
- **System fonts only** — uses the OS font stack (`-apple-system`, `Segoe UI`, `Roboto`, etc.); no web font downloads
- **Broad browser support** — tested in Chrome, Firefox, Edge, and Safari
- **`file://` compatible** — open directly from disk; no local server needed
- **No install** — no Python, Node.js, or any runtime required

---

## Customization

The color scheme is controlled by CSS custom properties at the top of `dashboard.html`. To modify the theme:

1. Open `dashboard.html` in a text editor.
2. Find the `[data-theme="dark"]` block (~line 22) to edit dark mode colors.
3. Find the `[data-theme="light"]` block (~line 54) to edit light mode colors.

Key variables:

| Variable | Purpose |
|----------|---------|
| `--accent` | Primary highlight color (tab underlines, buttons) |
| `--green` / `--yellow` / `--red` | Score threshold colors (≥0.8 / ≥0.5 / <0.5) |
| `--chart-1` … `--chart-5` | Per-model colors used in charts and legends |
| `--bg-body` / `--bg-surface` | Page and card background colors |
| `--text-primary` / `--text-secondary` | Main and muted text colors |

Changes take effect immediately on reload.
