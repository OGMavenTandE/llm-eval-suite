# Air-gap install

This is the documented path for copying LLM Eval Suite onto a closed network (IL5, IL6, or JWICS) and running it there for a Department of War evaluation. It is not an accreditation. It does not mean the suite has been installed or run on a classified network. Build the wheelhouse on a connected machine, carry it across on approved media, and install with no index.

The pins match `llm_eval_suite/eval_env.py` and `run_app.bat`: torch 2.13.0, garak 0.17.0, transformers 5.18.0. The GPU wheel comes from the PyTorch CUDA 12.6 index.

## On the connected build machine

From the repo root, with Python 3.10 or newer:

```bash
python -m pip download -d wheelhouse torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip download -d wheelhouse garak==0.17.0 transformers==5.18.0 torch==2.13.0 --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip download -d wheelhouse .[api]
```

Download torch from the CUDA index first so the wheelhouse contains the CUDA build, not a CPU wheel from PyPI. Copy the repo and the `wheelhouse` folder. Do not copy `.venv`, `.venv-eval`, `runs/`, or `data/`.

`run_app.bat` calls pip and the PyTorch index on the machine where it runs. Use it only on the connected build machine. On the closed network, use the commands below instead.

## On the closed network

```bash
python -m pip install --no-index --find-links wheelhouse torch==2.13.0 garak==0.17.0 transformers==5.18.0
python -m pip install --no-index --find-links wheelhouse .[api]
pip install -e . --no-deps --no-index --no-build-isolation
```

The last command is the offline editable install. `--no-deps` keeps pip from looking up packages. `--no-index` blocks the package index. `--no-build-isolation` uses the setuptools already on the machine, which is what writes the console scripts, including `llm-eval-dow`. setuptools and wheel have to be installed before that command. If the Scripts directory is not on PATH, `python -m dow_bench` runs the same tools.

Set these before starting the app or a run:

```bash
set LLM_EVAL_OFFLINE=1
set LLM_EVAL_AIRGAP_BUILD=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
```

`LLM_EVAL_OFFLINE=1` is the runtime switch. `LLM_EVAL_AIRGAP_BUILD=1` hard-disables the Hugging Face hub branch even if a preset or a saved connection still names a hub repo. The same switch can be saved from the Connect screen, or set with `"offline": true` on a preset. A preset that sets it refuses weight downloads and blocks the OpenAI cloud API for that run. A model on this computer (Ollama, a local OpenAI-compatible server, or a Hugging Face folder) still runs.

The app serves its own fonts from `llm_eval_suite/static/fonts/`. It does not load a font CDN.

## Public capability kit

`standard_capability` scores MMLU-Pro, GSM8K, MATH-500, IFEval, HumanEval+, and MBPP+. `standard_capability_quick` scores the same six suites, 50 items each, with a fixed seed and a 512-token thinking cap on reasoning models. A stage run reads only local files. It does not call Hugging Face, the datasets server, or any other host. SWE-bench is not part of these presets. Model weights are not in the kit.

On a connected machine, with the repo installed:

```bash
python -m standard_bench bundle --output dist/standard-capability-kit.zip
```

That writes one zip: the test kit code, the six pinned suites, and `kit-manifest.json` with a SHA-256 for each file. The command prints the zip size. A kit built from the pinned revisions on 11 October 2026 is 5.09 MB. Copy that zip to the closed network with the wheelhouse. Do not copy `.venv` or `runs/`.

If the live Hub revision differs from the pin in `standard_bench/catalog.py`, the download inside `bundle` stops. `--allow-moved` is the only override, and it records the revision it actually downloaded.

On the closed network, set the offline variables above, unzip the kit, and check it:

```bash
python -m standard_bench verify-kit --root dist\standard-capability-kit.zip
python -m standard_bench verify-reference --data datasets\standard
python -m standard_bench run --preset standard_capability_quick --data datasets\standard --model "OLMo 2 1B Instruct" --folder path\to\model --output runs\public.csv
```

`verify-reference` runs the stored HumanEval+ and MBPP+ reference solutions. It does not call a model and it does not use the network. A failure means the kit or the scorer is not the pinned set. `--fixtures` runs the same check on the small packaged rows. On Windows the same command is the check. Python 3.12, offline, is the platform that reported MBPP+ 375/377 before the float and complex compare and the `sys.getsizeof` exclusion for task 596. A passing run prints `reference ok` on Linux and on Windows. Re-score a finished run from its saved responses, without loading a model, with `python -m standard_bench rescore --run-dir runs/<id> --data datasets/standard`. The scorecard and the CSV record the scorer version and the git commit.

Point `--data` at `datasets/standard` inside the unpacked kit, or set `LLM_EVAL_STANDARD_DATA`. The runner checks the dataset hashes before it scores, and it checks `kit-manifest.json` when that file sits above the data directory. A mismatch stops the suite. Nothing is downloaded to repair it. A quick run is stamped `quick-check` in the scorecard and the CSV, with the preset name, the subset, the per-suite counts, and the caps. Licenses are in [LICENSES.md](LICENSES.md). The suite notes are in [STANDARD_BENCH.md](STANDARD_BENCH.md). The 4070 Super plan is in [STANDARD_BENCH_RUN_PLAN.md](STANDARD_BENCH_RUN_PLAN.md).

## Model folders

Point every connection at a directory that already contains `config.json` and the weight files. A hub name such as `gpt2-medium` is refused while offline mode or the air-gap build is on.

The demo preset reads the base model from `models/gpt2-medium`, or from `LLM_EVAL_DEMO_BASE_FOLDER` if that variable is set. Put `config.json` and the weight files in that folder. The preset does not download them.

## Garak detector folders

garak 0.17.0 loads two classifiers from the Hugging Face hub when a misleading probe runs. Offline mode skips the detector and names the missing repo instead of downloading it. Copy these files into the folders below. Revisions are the Hub commit SHAs for those repos.

`garak-llm/refutation_detector_distilbert` at `906ac60cba379abc1ad1ed328acebeb17357adac`, used by `misleading.MustRefuteClaimModel`. Copy into `models/garak_detectors/refutation_detector_distilbert` (or `LLM_EVAL_GARAK_REFUTATION_DIR`):

- config.json
- model.safetensors
- special_tokens_map.json
- tokenizer.json
- tokenizer_config.json
- vocab.txt

`garak-llm/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli` at `75044664e962c6237d48ec4d72fa189fb8723fdc`, used by `misleading.MustContradictNLI` (a RoBERTa natural-language-inference model). Copy into `models/garak_detectors/roberta_nli` (or `LLM_EVAL_GARAK_NLI_DIR`):

- config.json
- merges.txt
- model.safetensors
- special_tokens_map.json
- tokenizer_config.json
- vocab.json

`LLM_EVAL_GARAK_DETECTORS_DIR` changes the parent of both folders. The stage-1 Department of War suites do not call these detectors. Government T&E does, because its garak list includes the misleading probes.

## Pinned offline wheels

Copy this wheel into `wheelhouse` on the connected machine and check the sha256 before moving the media. It is used only for 8-bit loading of models such as Gemma 4 E2B.

- bitsandbytes 0.50.2
- `bitsandbytes-0.50.2-py3-none-win_amd64.whl`
- sha256 `c697963c8fda3dcd0d7ebd9b5211ae4067feef7cd06e0350d4e816a434fe683d`
