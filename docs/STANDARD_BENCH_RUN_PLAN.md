# Standard bench run plan

One RTX 4070 Super, 12 GB, nine stage-1 models, batch size 1. Do not treat this file as a measurement. The repository does not record stage-1 tokens per second. The README notes for this GPU are gpt2-medium accuracy counts. The rates below are planning assumptions.

Generation only. Code-test time in the sandbox is not included. Prompt tokens are charged at an assumed 1500 tok/s. Generated tokens, including the full 2048 thinking cap on thinking models, are charged at the decode rate below. Both rates are assumptions, not measurements. A prompt over its budget is not scored as a pass. OLMo 2 1B Instruct has a 4096 context. MMLU-Pro 5-shot prompts can exceed that budget. Those items increment `n_over_budget` and stay out of the accuracy denominator.

The customer presets score six suites. GPQA Diamond is not in these hours. The full suites are the default when `--subset` is omitted. Use the short subset or the quick check before a full run.

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

Tokens per item are prompt plus answer. Thinking models add 2048 to the answer side on a full run. The quick check uses 512 instead, and only on thinking models.

| Suite | Items | Prompt | Answer | Tokens, no thinking | Tokens, thinking |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro | 12032 | 2800 | 350 | 3150 | 5198 |
| gsm8k | 1319 | 900 | 180 | 1080 | 3128 |
| math500 | 500 | 250 | 400 | 650 | 2698 |
| ifeval | 541 | 180 | 220 | 400 | 2448 |
| humaneval_plus | 164 | 280 | 180 | 460 | 2508 |
| mbpp_plus | 378 | 220 | 160 | 380 | 2428 |

Full size is 14934 items. These token budgets live in `standard_bench/plan.py`.

## Full run, hours

| Model | mmlu_pro | gsm8k | math500 | ifeval | humaneval_plus | mbpp_plus | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLMo 2 1B Instruct | 20.86 | 1.04 | 0.72 | 0.43 | 0.11 | 0.23 | 23.39 |
| LFM2.5-1.2B Instruct | 20.86 | 1.04 | 0.72 | 0.43 | 0.11 | 0.23 | 23.39 |
| Granite 4.0 H 1B | 20.86 | 1.04 | 0.72 | 0.43 | 0.11 | 0.23 | 23.39 |
| Phi-4-mini-instruct | 32.23 | 1.69 | 1.26 | 0.75 | 0.19 | 0.39 | 36.51 |
| Gemma 4 E2B-it | 35.48 | 1.87 | 1.41 | 0.84 | 0.21 | 0.44 | 40.26 |
| LFM2.5-2.6B | 235.23 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 287.81 |
| Granite 4.2 3B | 235.23 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 287.81 |
| Phi-4-mini-reasoning | 235.23 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 287.81 |
| NVIDIA Nemotron 3 Nano 4B | 235.23 | 23.54 | 9.74 | 9.76 | 2.91 | 6.64 | 287.81 |

Nine models, full suites: 1298.19 hours. MMLU-Pro is still most of that.

## Recommended first run

Do this in two stages. Stage 1 is the short subset on the five non-thinking models, with the usual answer caps and no thinking cap. Stage 2 is the quick check on the four reasoning models: 50 items per suite, the same items for every model, and a 512-token thinking cap. The answer caps stay the same. Non-thinking models can also run the quick check. The 512 cap does not bind for them.

Replace the 1500 tok/s prefill assumption with the speeds measured in stage 1 before stage 2 starts. The hours below still use 1500 tok/s.

### Stage 1, short subset

| Suite | Limit |
| --- | --- |
| mmlu_pro | 200 |
| gsm8k | 200 |
| math500 | 100 |
| ifeval | 100 |
| humaneval_plus | 164 |
| mbpp_plus | 50 |

| Model | Full hours | Short hours |
| --- | --- | --- |
| OLMo 2 1B Instruct | 23.39 | 0.87 |
| LFM2.5-1.2B Instruct | 23.39 | 0.87 |
| Granite 4.0 H 1B | 23.39 | 0.87 |
| Phi-4-mini-instruct | 36.51 | 1.42 |
| Gemma 4 E2B-it | 40.26 | 1.58 |
| LFM2.5-2.6B | 287.81 | 15.02 |
| Granite 4.2 3B | 287.81 | 15.02 |
| Phi-4-mini-reasoning | 287.81 | 15.02 |
| NVIDIA Nemotron 3 Nano 4B | 287.81 | 15.02 |

Nine models, full suites: 1298.19 hours. Nine models, short subset: 65.68 hours. Stage 1, the five non-thinking models on the short subset: 5.61 hours.

The same caps as `--limits mmlu_pro=200,gsm8k=200,math500=100,ifeval=100,humaneval_plus=164,mbpp_plus=50`.

### Stage 2, quick check

`standard_capability_quick` selects 50 items from each of the six suites with seed 20261011. Thinking models use a 512-token thinking cap. The table uses that cap for the four reasoning models and no thinking cap for the others.

| Model | Quick-check hours |
| --- | --- |
| OLMo 2 1B Instruct | 0.30 |
| LFM2.5-1.2B Instruct | 0.30 |
| Granite 4.0 H 1B | 0.30 |
| Phi-4-mini-instruct | 0.50 |
| Gemma 4 E2B-it | 0.56 |
| LFM2.5-2.6B | 1.85 |
| Granite 4.2 3B | 1.85 |
| Phi-4-mini-reasoning | 1.85 |
| NVIDIA Nemotron 3 Nano 4B | 1.85 |

Nine models, quick check: 9.38 hours. Stage 2, the four reasoning models: 7.41 hours. Stage 1 plus stage 2: 13.03 hours.

A full reasoning-model run at the 2048 cap is 287.81 hours. The quick check is the number to use first.

## Commands

Build the kit once on a connected machine. The command fetches the six public suites, packs the test kit code and those files, and prints the zip size. A kit built from the pinned revisions on 11 October 2026 is 5.05 MB. Copy the zip. Do not copy model weights in that zip. On the closed network, unzip it and run. The runner checks every SHA-256 before it scores.

```bash
python -m standard_bench bundle --output dist/standard-capability-kit.zip
```

If the six suites are already in a directory, pack that directory and skip the download:

```bash
python -m standard_bench bundle --data datasets/standard --output dist/standard-capability-kit.zip
```

Stage 1 is the short subset. `--model` is the display name in `dow_bench/models.json`. Thinking caps come from that file unless the quick preset or `--thinking-budget` sets another cap. Gemma is 8-bit. The others are fp16. Replace `$MODEL_DIR` with the local folder that already contains `config.json` and the weights. `$DATA` is `datasets/standard` inside the unpacked kit.

```bash
python -m standard_bench run --preset standard_capability --subset short --data "$DATA" --model "OLMo 2 1B Instruct" --folder "$MODEL_DIR/OLMo-2-0425-1B-Instruct" --precision fp16 --output runs/olmo-public.csv --runs-dir runs/olmo-public
python -m standard_bench run --preset standard_capability --subset short --data "$DATA" --model "LFM2.5-1.2B Instruct" --folder "$MODEL_DIR/LFM2.5-1.2B-Instruct" --precision fp16 --output runs/lfm12-public.csv --runs-dir runs/lfm12-public
python -m standard_bench run --preset standard_capability --subset short --data "$DATA" --model "Granite 4.0 H 1B" --folder "$MODEL_DIR/granite-4.0-h-1b" --precision fp16 --output runs/granite1b-public.csv --runs-dir runs/granite1b-public
python -m standard_bench run --preset standard_capability --subset short --data "$DATA" --model "Phi-4-mini-instruct" --folder "$MODEL_DIR/Phi-4-mini-instruct" --precision fp16 --output runs/phi-instruct-public.csv --runs-dir runs/phi-instruct-public
python -m standard_bench run --preset standard_capability --subset short --data "$DATA" --model "Gemma 4 E2B-it" --folder "$MODEL_DIR/gemma-4-E2B-it" --precision 8bit --output runs/gemma-public.csv --runs-dir runs/gemma-public
```

Stage 2 is the named quick check. Omit `--subset` and `--thinking-budget` to keep the preset's 50-item sample and 512-token cap. Pass either flag to override it.

```bash
python -m standard_bench run --preset standard_capability_quick --data "$DATA" --model "LFM2.5-2.6B" --folder "$MODEL_DIR/LFM2.5-2.6B" --precision fp16 --output runs/lfm26-public.csv --runs-dir runs/lfm26-public
python -m standard_bench run --preset standard_capability_quick --data "$DATA" --model "Granite 4.2 3B" --folder "$MODEL_DIR/granite-4.2-3b" --precision fp16 --output runs/granite3b-public.csv --runs-dir runs/granite3b-public
python -m standard_bench run --preset standard_capability_quick --data "$DATA" --model "Phi-4-mini-reasoning" --folder "$MODEL_DIR/Phi-4-mini-reasoning" --precision fp16 --output runs/phi-reason-public.csv --runs-dir runs/phi-reason-public
python -m standard_bench run --preset standard_capability_quick --data "$DATA" --model "NVIDIA Nemotron 3 Nano 4B" --folder "$MODEL_DIR/NVIDIA-Nemotron-3-Nano-4B-BF16" --precision fp16 --output runs/nemotron-public.csv --runs-dir runs/nemotron-public
```

Omit `--subset` on `standard_capability` for the full six suites. A smoke that does not load a GPU:

```bash
python -m standard_bench smoke --runs-dir runs/standard-smoke --output runs/standard-smoke/public.csv
```
