# Edge demo: Q4 versus FP16

Two recipes for a Windows machine with an RTX 4070 Super and Ollama. One 3B model, FP16 against Q4_K_M. The short recipe is a two-minute smoke. The longer recipe is a quick standard subset and is the one to show for an accuracy delta.

The sample file `datasets/sample_factcheck_50.jsonl` is general knowledge. It is not the Department of War set. The standard preset does not use that file.

An 8B FP16 weight does not fit in 12 GB. Use the 3B pair below. Both fit on a 4070 Super.

## Before the meeting

Install the suite, then pull both tags:

```
python -m pip install -e ".[dev,api]"
ollama pull llama3.2:3b-instruct-fp16
ollama pull llama3.2:3b-instruct-q4_K_M
```

Optional third tag: `ollama pull llama3.2:3b-instruct-q8_0`.

`nvidia-smi` on PATH is enough for watts, VRAM, and the hardware stamp. `pip install pynvml` is optional. If the GPU is missing, or Ollama is on another machine, those counters show n/a and the reason. CPU, RAM, and OS are still stamped.

Confirm the tags with `ollama list`. If a name differs, use the name you have after the `=` in the command below.

## Two-minute smoke

`quant_quick` scores the first five fact-check questions, once each, on each quantization. Keep this for a wiring check. Five items are too few for a credible accuracy delta: a one-item swing is 20 points, and the report marks that as within ±1 item.

Dry run, from the repo root:

```
python -m llm_eval.edge quant-compare --dry-run --provider ollama --base-url http://127.0.0.1:11434 --baseline fp16=llama3.2:3b-instruct-fp16 --variant q4=llama3.2:3b-instruct-q4_K_M --preset quant_quick --dataset datasets/sample_factcheck_50.jsonl --tolerance 0.05 --out runs/quant-compare
```

This writes `runs/quant-compare/plan.json` and does not call the model. Drop `--dry-run` to run it. A larger `--per-suite` does not raise this preset above five questions.

## 30 to 45 minute standard subset

`standard_capability_quick` is the quick50 standard check: six suites, 50 items each, fixed seed. That is the fuller quick check and runs longer than a meeting slot if the model uses its thinking budget.

For the demo, cap each suite at 20 items. That is 120 items per quantization, 240 generations for the pair. On a 4070 Super with this 3B pair, expect about 30 to 45 minutes when answers stay short of the thinking cap. A model that fills the 512-token thinking budget on every item will land at the long end of that window, or past it.

```
python -m llm_eval.edge quant-compare --provider ollama --base-url http://127.0.0.1:11434 --baseline fp16=llama3.2:3b-instruct-fp16 --variant q4=llama3.2:3b-instruct-q4_K_M --preset standard_capability_quick --per-suite 20 --tolerance 0.05 --out runs/quant-compare
```

Omit `--per-suite` to score the full quick50 subset. Add a Q8 variant with another `--variant q8=llama3.2:3b-instruct-q8_0`.

Optional load probe, after the scored items:

```
--concurrency 4
```

That records p50 and p95 from four parallel calls. The scored answers stay sequential.

`--preset` accepts any suite preset the runner knows. `--per-suite N` caps standard benchmarks at N (it wins over quick50), and it caps fact-check, robustness, consistency, and garak prompts without raising a smaller preset cap. Department of War suites keep their item files.

## What you get

`runs/quant-compare/report.md`, `report.html`, and `report.json`.

- Per-category and per-benchmark accuracy delta against FP16. Each row shows the pass rate, the passed count, and n.
- A noise note when the passed counts differ by at most one item, or when the 95% binomial intervals overlap. The note does not change the verdict. A drop larger than `--tolerance` (0.05 here) still fails that row.
- Speed from the edge metrics: TTFT p50 and p95, output tokens per second, and latency p50 and p95.
- Power from the same run: average watts, peak watts, joules per output token, peak VRAM, and peak RAM. These numbers are whatever the 4070 Super reports during the run. They are not a fixed table.
- Verdict `pass`, `fail`, or `n/a`. Speed and power are reported. They do not change the verdict. A withheld score, or a counter that was not measured, is `n/a` with a reason.

Open the two runs in the app Compare view to see the same speed and power deltas next to the category bars.

## llama.cpp or vLLM

Use the connect type for that server. Do not use a cloud URL.

```
python -m llm_eval.edge quant-compare --provider llama_cpp --base-url http://127.0.0.1:8080/v1 --baseline fp16=model-fp16 --variant q4=model-q4 --preset standard_capability_quick --per-suite 20 --tolerance 0.05 --out runs/quant-compare
```

llama.cpp listens on `http://127.0.0.1:8080/v1`. vLLM listens on `http://127.0.0.1:8000/v1`. Pass `--provider vllm` and that base URL for an AWQ or other local vLLM model. The model name is whatever that server expects. Both connectors stream, so the edge report includes time to first token.
