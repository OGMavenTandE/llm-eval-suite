# Edge demo: Q4 versus FP16

Monday demo on a Windows machine with an RTX 4070 Super and Ollama. One model, two quantizations, the quick fact-check subset.

The sample file `datasets/sample_factcheck_50.jsonl` is general knowledge. It is not the Department of War set.

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

## Dry run

From the repo root, in PowerShell or Command Prompt:

```
python -m llm_eval.edge quant-compare --dry-run --provider ollama --base-url http://127.0.0.1:11434 --baseline fp16=llama3.2:3b-instruct-fp16 --variant q4=llama3.2:3b-instruct-q4_K_M --preset quant_quick --dataset datasets/sample_factcheck_50.jsonl --tolerance 0.05 --out runs/quant-compare
```

This writes `runs/quant-compare/plan.json` and does not call the model.

## Live compare

Drop `--dry-run`:

```
python -m llm_eval.edge quant-compare --provider ollama --base-url http://127.0.0.1:11434 --baseline fp16=llama3.2:3b-instruct-fp16 --variant q4=llama3.2:3b-instruct-q4_K_M --preset quant_quick --dataset datasets/sample_factcheck_50.jsonl --tolerance 0.05 --out runs/quant-compare
```

`quant_quick` scores the first five fact-check questions, once each. The same five run on FP16 and on Q4.

Add a Q8 variant with another `--variant q8=llama3.2:3b-instruct-q8_0`.

Optional load probe, after the scored questions:

```
--concurrency 4
```

That records p50 and p95 from four parallel calls. The scored answers stay sequential.

## What you get

`runs/quant-compare/report.md`, `report.html`, and `report.json`.

- Per-category and per-benchmark accuracy delta against FP16. A drop larger than `--tolerance` (0.05 here) fails that row.
- TTFT, output tokens/s, and latency p50/p95 deltas from the edge metrics.
- Average watts, peak watts, joules per output token, peak VRAM, and peak RAM.
- Verdict `pass`, `fail`, or `n/a`. Speed and power are reported. They do not change the verdict. A withheld score, or a counter that was not measured, is `n/a` with a reason.

Open the two runs in the app Compare view to see the same speed and power deltas next to the category bars.

## llama.cpp or AWQ

Point the same command at a local OpenAI-compatible server. Do not use a cloud URL.

```
python -m llm_eval.edge quant-compare --provider openai --base-url http://127.0.0.1:8080/v1 --baseline fp16=model-fp16 --variant awq=model-awq --preset quant_quick --dataset datasets/sample_factcheck_50.jsonl --tolerance 0.05 --out runs/quant-compare
```

llama.cpp's server is usually `http://127.0.0.1:8080/v1`. A local vLLM AWQ server uses the port you started. The model name is whatever that server expects.
