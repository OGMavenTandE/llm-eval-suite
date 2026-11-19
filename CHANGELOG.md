# Changelog

## Unreleased

### Added

- Evaluation gates for small-model specialization (`python -m gates`, `llm-eval-gates`). One workload is walked down the Enterprise AI Adoption Curve with a GO, NO-GO, or INVALID decision at each gate: Gate 2 baseline measurement, Gate 3 right-sizing log audit, Gate 4 routing harness with stress cases, Gate 5 acceptance against the model it would replace (paired bootstrap non-inferiority and per-class recall bars), and Gate 6 drift (accuracy drop, label-mix PSI, version agreement). Each gate writes HTML, markdown, and JSON with the SHA-256 of every input. Exit codes are 0, 1, and 2 so a pipeline can stop on a gate.
- `gates.train_lora`: offline LoRA adapter training on a local base-model folder, trained on the `train` split only, with a training record and an optional merged folder.
- Synthetic message-triage reference case in `examples/message_triage/`: seeded generator, spec, audit rules, router cases, and a shifted monitoring month. No test message appears in training, and one phrasing per label is held out of training.
- `python -m gates demo` writes a simulated packet for layout checks. Every page carries a SIMULATED banner.

## 0.2.0 - 2026-10-14

Changes since v0.1.0 (`e61f8f7c`), from merged pull requests #38 through #47.

### Added

- Edge metrics on each run: streaming time to first token, output tokens per second, p50 and p95 latency, peak RAM and VRAM, GPU power, and a hardware stamp. A missing counter is n/a with a reason. (#38)
- vLLM and llama.cpp server connectors, with health checks, model listing, chat completions, a 512 token cap, and streaming time to first token. (#39)
- Quantization comparison preset and a delta report for accuracy, speed, and power against a chosen baseline. Each category and benchmark row includes the pass rate, the passed count, and n. A one-item swing is marked within +/- 1 item. A larger swing whose 95% binomial intervals overlap is marked as within that interval. (#40)
- Plugin API 0.2.0. Packages can register suites, scorers, reporters, connectors, installers, and panels. A plugin loads when its major API version matches the host. A run records the plugins that ran. Private wheels can be packed into the offline bundle. (#43)
- Offline bundle verification in CI. `scripts/verify_offline.sh` builds the bundle without a package index, installs it with `pip --no-index`, runs doctor, and runs a stub quant-compare. A socket connect fails the script. (#46)

### Changed

- Package author, maintainer, and owner are AI Eval Corporation. The offline bundle includes the certifi and tqdm MPL-2.0 texts and a Built with Llama notice. (#41)
- Default council judges are `phi4-mini` (MIT) and `granite4:micro` (Apache-2.0). A judge does not grade a model from its own family. (#42)
- Dataset fetch retries HTTP 429 and 5xx. At 20 items per suite, a failed quant row inside the 95% interval is labeled as sampling noise. The verdict stays fail. (#46)

### Fixed

- Quant-compare checks the standard benchmark directory before any model call, reads suite pass rates from `run.json`, isolates peak VRAM from leftover models, and exits nonzero when a run or the verdict fails. (#44)
- The demo-start test waits up to 30 seconds for each run to leave `running`, so Ubuntu CI does not fail while the quick demos are still going. (#45)
- OpenAI-compatible calls keep TLS verification and accept an optional CA bundle. Configured base URLs must be http or https. Link-local and metadata hosts are rejected. Localhost still connects. Evaluator, robustness, and connection-error tests cover those paths, and the summary and comparison share one score grouping. (#47)
