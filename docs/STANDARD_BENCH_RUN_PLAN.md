# Standard bench run plan

One RTX 4070 Super, 12 GB, nine stage-1 models, batch size 1. Do not treat this file as a measurement. The repository does not record stage-1 tokens per second. The README notes for this GPU are gpt2-medium accuracy counts. The rates below are planning assumptions.

Generation only. Code-test time in the sandbox is not included. Prompt tokens are charged at an assumed 1500 tok/s. Generated tokens, including the full 2048 thinking cap on thinking models, are charged at the decode rate below. Both rates are assumptions, not measurements. A prompt over its budget is not scored as a pass. OLMo 2 1B Instruct has a 4096 context. MMLU-Pro 5-shot prompts can exceed that budget. Those items increment `n_over_budget` and stay out of the accuracy denominator.

Run the short subset first. The full suites are the default when `--limits` is omitted. Use the full run after the short subset looks right.

## Assumptions

| Model key | Display name | Assumed tok/s | Thinking cap |
| --- | --- | --- | --- |
| olmo-2-1b | OLMo 2 1B Instruct | 80 | 0 |
| lfm-1.2b | LFM2.5-1.2B Instruct | 80 | 0 |
| granite-1b | Granite 4.0 H 1B | 80 | 0 |
| phi-4-mini-instruct | Phi-4-mini-instruct | 45 | 0 |
| gemma-4-e2b-8bit | Gemma 4 E2B-it | 40 | 0 |
| lfm-2.6b | LFM2.5-2.6B | 35 | 2048 |
| granite-4.2-3b | Granite 4.2 3B | 35 | 2048 |
| phi-4-mini-reasoning | Phi-4-mini-reasoning | 35 | 2048 |
| nemotron-3-nano-4b | NVIDIA Nemotron 3 Nano 4B | 35 | 2048 |

The 80 tok/s group is the 1.2B to 1.5B non-thinking models. Phi-4-mini-instruct is 45. Gemma 4 E2B-it at 8-bit is 40. The four thinking models, from 2.6B to 4B, are 35. Prefill for every model is 1500 tok/s. That prefill figure is an assumption, not a measured stage-1 rate.

## Item counts and tokens

Tokens per item are prompt plus answer. Thinking models add 2048 to the answer side.

| Suite | Items | Prompt | Answer | Tokens, no thinking | Tokens, thinking |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro | 12032 | 2800 | 350 | 3150 | 5198 |
| gpqa_diamond | 198 | 450 | 400 | 850 | 2898 |
| gsm8k | 1319 | 900 | 180 | 1080 | 3128 |
| math500 | 500 | 250 | 400 | 650 | 2698 |
| ifeval | 541 | 180 | 220 | 400 | 2448 |
| humaneval_plus | 164 | 280 | 180 | 460 | 2508 |
| mbpp_plus | 378 | 220 | 160 | 380 | 2428 |

Full size is 15132 items. These token budgets live in `standard_bench/plan.py`.

## Full run, hours

| Model | mmlu_pro | gpqa_diamond | gsm8k | math500 | ifeval | humaneval_plus | mbpp_plus | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLMo 2 1B Instruct | 20.86 | 0.29 | 1.04 | 0.72 | 0.43 | 0.11 | 0.23 | 23.68 |
| LFM2.5-1.2B Instruct | 20.86 | 0.29 | 1.04 | 0.72 | 0.43 | 0.11 | 0.23 | 23.68 |
| Granite 4.0 H 1B | 20.86 | 0.29 | 1.04 | 0.72 | 0.43 | 0.11 | 0.23 | 23.68 |
| Phi-4-mini-instruct | 32.23 | 0.51 | 1.69 | 1.26 | 0.75 | 0.19 | 0.39 | 37.01 |
| Gemma 4 E2B-it | 35.48 | 0.57 | 1.87 | 1.41 | 0.84 | 0.21 | 0.44 | 40.82 |
| LFM2.5-2.6B | 235.23 | 3.86 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 291.68 |
| Granite 4.2 3B | 235.23 | 3.86 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 291.68 |
| Phi-4-mini-reasoning | 235.23 | 3.86 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 291.68 |
| NVIDIA Nemotron 3 Nano 4B | 235.23 | 3.86 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 291.68 |

Nine models, full suites: 1315.59 hours. MMLU-Pro is still most of that. The short subset is the first run.

## Short subset

This is the first run. GPQA Diamond and HumanEval+ stay at full size because they are already small. The other caps are the first N items, or a seeded sample if you pass `--seed`.

| Suite | Limit |
| --- | --- |
| mmlu_pro | 200 |
| gpqa_diamond | 198 |
| gsm8k | 200 |
| math500 | 100 |
| ifeval | 100 |
| humaneval_plus | 164 |
| mbpp_plus | 50 |

| Model | Full hours | Short hours |
| --- | --- | --- |
| OLMo 2 1B Instruct | 23.68 | 1.16 |
| LFM2.5-1.2B Instruct | 23.68 | 1.16 |
| Granite 4.0 H 1B | 23.68 | 1.16 |
| Phi-4-mini-instruct | 37.01 | 1.93 |
| Gemma 4 E2B-it | 40.82 | 2.15 |
| LFM2.5-2.6B | 291.68 | 18.88 |
| Granite 4.2 3B | 291.68 | 18.88 |
| Phi-4-mini-reasoning | 291.68 | 18.88 |
| NVIDIA Nemotron 3 Nano 4B | 291.68 | 18.88 |

Nine models, full suites: 1315.59 hours. Nine models, short subset: 83.08 hours.

## Commands

Fetch once on a connected machine. GPQA needs `HF_TOKEN` in the environment. Do not copy the token to the closed network.

```bash
python -m standard_bench fetch --dest datasets/standard
```

The first run is the short subset. Add this to each command below:

```bash
--limits mmlu_pro=200,gpqa_diamond=198,gsm8k=200,math500=100,ifeval=100,humaneval_plus=164,mbpp_plus=50
```

`--model` is the display name in `dow_bench/models.json`. Thinking caps come from that file. Gemma is 8-bit. The others are fp16. Replace `$MODEL_DIR` with the local folder that already contains `config.json` and the weights. Omit `--limits` for the full suites.

```bash
python -m standard_bench run --data datasets/standard --model "OLMo 2 1B Instruct" --folder "$MODEL_DIR/OLMo-2-0425-1B-Instruct" --precision fp16 --output runs/olmo-public.csv --runs-dir runs/olmo-public
python -m standard_bench run --data datasets/standard --model "LFM2.5-1.2B Instruct" --folder "$MODEL_DIR/LFM2.5-1.2B-Instruct" --precision fp16 --output runs/lfm12-public.csv --runs-dir runs/lfm12-public
python -m standard_bench run --data datasets/standard --model "Granite 4.0 H 1B" --folder "$MODEL_DIR/granite-4.0-h-1b" --precision fp16 --output runs/granite1b-public.csv --runs-dir runs/granite1b-public
python -m standard_bench run --data datasets/standard --model "Phi-4-mini-instruct" --folder "$MODEL_DIR/Phi-4-mini-instruct" --precision fp16 --output runs/phi-instruct-public.csv --runs-dir runs/phi-instruct-public
python -m standard_bench run --data datasets/standard --model "Gemma 4 E2B-it" --folder "$MODEL_DIR/gemma-4-E2B-it" --precision 8bit --output runs/gemma-public.csv --runs-dir runs/gemma-public
python -m standard_bench run --data datasets/standard --model "LFM2.5-2.6B" --folder "$MODEL_DIR/LFM2.5-2.6B" --precision fp16 --output runs/lfm26-public.csv --runs-dir runs/lfm26-public
python -m standard_bench run --data datasets/standard --model "Granite 4.2 3B" --folder "$MODEL_DIR/granite-4.2-3b" --precision fp16 --output runs/granite3b-public.csv --runs-dir runs/granite3b-public
python -m standard_bench run --data datasets/standard --model "Phi-4-mini-reasoning" --folder "$MODEL_DIR/Phi-4-mini-reasoning" --precision fp16 --output runs/phi-reason-public.csv --runs-dir runs/phi-reason-public
python -m standard_bench run --data datasets/standard --model "NVIDIA Nemotron 3 Nano 4B" --folder "$MODEL_DIR/NVIDIA-Nemotron-3-Nano-4B-BF16" --precision fp16 --output runs/nemotron-public.csv --runs-dir runs/nemotron-public
```

A smoke that does not load a GPU:

```bash
python -m standard_bench smoke --runs-dir runs/standard-smoke --output runs/standard-smoke/public.csv
```
