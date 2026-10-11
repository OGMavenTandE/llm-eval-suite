# Standard capability benchmarks

`standard_capability` is a native preset. It does not install lm-evaluation-harness, EvalPlus, or another eval framework. Prompting and scoring live in `standard_bench/`. A stage run is offline. It reads the six suites packed in the customer kit.

`standard_capability_quick` scores the same six suites, 50 items each, with seed 20261011. Reasoning models use a 512-token thinking cap. Non-reasoning models run the same 50 items, and that cap does not bind. `--subset` and `--thinking-budget` override the preset. A quick run records `protocol` `quick-check`, the subset, the item counts, and the caps. It is not a full-suite result.

Scores are deterministic. There is no LLM judge. Public items use `category` `public_benchmark` and `counts_toward_dow_score` false. `aieval_dow_bench_score` ignores these suites even if those flags are wrong. They do not enter the AIEval DoW Bench Score. GPQA Diamond is not in either customer preset. A customer run records it as not run.

Full sizes, with no `--limit` and no subset:

| Suite | Items |
| --- | --- |
| mmlu_pro | 12032 |
| gsm8k | 1319 |
| math500 | 500 |
| ifeval | 541 |
| humaneval_plus | 164 |
| mbpp_plus | 378 |

The total is 14934. HumanEval/32, MBPP+ task 255, and MBPP+ task 596 are in that count and are not scored. The quick check is 300 items, 50 from each suite. `--limit N` keeps the first N of each suite after an optional `--seed` shuffle. `--limits mmlu_pro=200,gsm8k=200` sets per-suite caps. `--subset short` is the screen in the run plan. `--subset quick50` is the 50-item sample. Over-budget prompts are not passes. Accuracy uses `n_scored`, the items that were generated.

Build the offline kit on a connected machine:

```bash
python -m standard_bench bundle --output dist/standard-capability-kit.zip
```

The command prints the zip size. A kit built from the pinned revisions on 11 October 2026 is 5.09 MB. The kit is the test kit code plus the six pinned suites, with a SHA-256 manifest. Model weights are not in the zip. On the closed network, unzip it and point `--data` at `datasets/standard`. The runner checks the hashes before it scores. Details are in [AIRGAP_INSTALL.md](AIRGAP_INSTALL.md) and [STANDARD_BENCH_RUN_PLAN.md](STANDARD_BENCH_RUN_PLAN.md).

## MMLU-Pro

Reference: TIGER-Lab/MMLU-Pro and `evaluate_from_local.py` in TIGER-AI-Lab/MMLU-Pro, 5-shot chain of thought, default ntrain 5. Paper: Wang et al., arXiv:2406.01574.

The prompt names the category, then five validation shots from that category, then the question. Options are `A. text` through `J. text`. Shot text replaces `A: Let's think step by step.` with `Answer: Let's think step by step.` The test item ends with `Answer: Let's think step by step.`

Extraction follows the official order. `answer is \(?([A-J])\)?` wins. Then `.*[aA]nswer:\s*([A-J])`. Then the last standalone `A` through `J`. Letters are case sensitive.

Deviation: a failed extraction is incorrect. The official local script draws a random letter. This runner does not. The official script also drops shots until the prompt fits 4096 tokens. This runner keeps 5 shots and records `prompt_over_budget` when the prompt exceeds the budget.

## GPQA Diamond, in-house only

GPQA Diamond is not part of `standard_capability`, `standard_capability_quick`, or the customer zip. Staff steps are in [GPQA_INHOUSE.md](GPQA_INHOUSE.md), which is not part of the customer kit. If the files are absent, the run continues and the scorecard says GPQA was not run.

Reference: Rein et al., arXiv:2311.12022. Diamond is the 198-row `gpqa_diamond.csv`. The prompt and the `Answer: $LETTER` line follow the OpenAI simple-evals multiple-choice template.

The four options are the correct answer plus three incorrect answers. `random.Random` is seeded with `20261010:` plus the record id, so a rerun assigns the same letters and the correct letter is not always A. Extraction uses the last `Answer: [A-D]`, otherwise the last standalone A through D.

Deviation: question text is not written to items, logs, or the CSV. The stored prompt and response are empty. The predicted letter and the expected letter remain.

## GSM8K

Reference: Cobbe et al., arXiv:2110.14168. The eight shots are the public gsm8k_cot training exemplars used by lm-evaluation-harness. They are not test rows. Gold is the number after `####`.

The reported score is exact match of the normalized final number. A number after `The answer is` wins. Otherwise the last number in the reply is used. Commas and currency marks are removed. `8.0` matches `8`. There is no calculator annotation and no majority vote.

## MATH-500

Reference: Hendrycks et al., arXiv:2103.03874, and the 500-problem split in HuggingFaceH4/MATH-500 from OpenAI prm800k. The prompt is the 0-shot Minerva form: the model finishes with `\boxed{}`.

The last `\boxed{}` wins, including nested braces. A missing box is a miss. Equivalence is sympy-free. The normalizer strips `\left` and `\right`, spacing, `\text{}`, and rewrites `\frac`, `\dfrac`, and `\tfrac` to a slash. Integers, decimals, and simple fractions match within 1e-6 relative to the larger magnitude. `1/2`, `0.5`, and `\frac{1}{2}` match. Symbolic rearrangement does not, so `x+1` and `1+x` do not match.

Deviation: this is not the original MATH 4-shot prompt, and it is not a sympy simplifier.

## IFEval

Reference: Zhou et al., arXiv:2311.07911, and the 25 instruction ids in google-research `instruction_following_eval`. The suite reports prompt-level and instruction-level accuracy, strict and loose.

Strict mode strips the response and requires every instruction to pass. Loose mode tries eight variants: the original, the original with asterisks removed, the first line dropped, the last line dropped, both lines dropped, and the asterisk-stripped copies of those line drops. An empty response fails. Prompt-level accuracy requires every instruction on that prompt. Instruction-level accuracy is the fraction of instructions that pass.

Deviation: sentence splits use the regex in `instructions_util`, not nltk punkt. Word counts use `\w+`. A hyphenated all-caps token counts as one capital word. `response_language` and the English case checks use a local script and stopword detector, not langdetect. An uncertain language fails the check. The official langdetect exception path counts the instruction as followed. Missing required arguments fail closed. Nothing is sampled.

`keywords:existence` is an unescaped search with no word boundary, so `cat` matches `category`. `keywords:forbidden_words` uses a word boundary and `re.escape`, so `cat` does not match `category`.

## HumanEval+ and MBPP+

Reference: Chen et al., arXiv:2107.03374, Austin et al., arXiv:2108.07732, and Liu et al., EvalPlus, arXiv:2305.01210. The metric is pass@1 with one greedy sample. Base tests and plus tests are separate. A plus pass requires the base tests and the plus tests.

HumanEval base tests come from openai/openai_humaneval at `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`. Plus tests come from evalplus/humanevalplus. The chat wrapper is `Complete the following Python function. Reply with the code only.` A fenced reply is unwrapped. If it defines the entry point, imports and helpers before that function stay, and helpers after it stay. Stop strings apply only after the entry function, and a following `def` or `class` is kept. Other stops still drop a trailing `print` or assert. A reply that does not define the entry point is the indented body. The official prompt is prepended, and every stop applies to that body, including a newline plus `class`, `def`, `#`, `if`, or `print`.

MBPP+ uses the EvalPlus task line, the statement, and the three base asserts. The same completion rule applies. Stop strings on a bare body are a newline plus a triple quote, `assert`, `class`, or `def`. The function header from the dataset is the prefix when the completion does not define the entry point. The upstream plus `test` field is module-level and has no `def check`. Fetch indents that body into `def check(candidate)` and renames `entry(` to `candidate(`, so the inputs, results, and assert loop are the plus test. The test function is renamed to `_standard_bench_check` when it runs, because MBPP+ task 56 defines a solution named `check`. The row's `test_imports` are prepended at module level, so a base assert that calls `math.isclose` can see `math`.

Deviation: the EvalPlus harness imports numpy. The plus-test assertion helper is replaced with a stdlib compare. Floats use `math.isclose` and complexes use `cmath.isclose`, including inside tuples, lists, and dicts, with abs_tol 1e-6 and rel_tol 1e-7. Integers, strings, bools, and other values stay exact. A set-comparison variant is kept when the test text contains `out = set(out)`, including when that line sits inside `check`, and sets stay exact. Helpers other than `is_floats` and `assertion` stay, including `ref_func` and `_poly`. Scoring strips `is_floats` and `assertion` before `check` and inside it, then rewrites the helper, so a kit fetched earlier still uses the tolerant compare. The run does not import numpy or evalplus. The pinned rows store the canonical solution: the HumanEval+ prompt plus `canonical_solution`, and the MBPP+ `code` field. `python -m standard_bench verify-reference --data datasets/standard` runs those solutions offline and requires every checked one to pass. `--fixtures` checks the packaged rows the same way.

## Code sandbox

Each test runs in `python -I` with a JSON payload on stdin. The parent enforces a wall-clock timeout. The default ceiling is 20 seconds, 20 seconds of CPU, and 1 GB of address space. Five seconds is not enough for MBPP+ task 599, which sums ranges past 1e8. The child blocks `socket.socket`, `create_connection`, and `getaddrinfo` before the program runs.

Three tasks are excluded from scoring. The rows stay in the pinned files. `verify-reference` does not require them to pass, and a run does not send them to the model. The suite metrics list them under `reference_exclusions`.

| Suite | Task | Reason |
| --- | --- | --- |
| HumanEval+ | HumanEval/32 | The plus test calls `_poly(*candidate(*inp), inp)`. `find_zero` returns a float, so the star raises TypeError. The reference solution passes the base tests. |
| MBPP+ | 255 | The plus test builds `combinations_with_replacement` for a length-77 sequence. The reference solution does the same and raises MemoryError at the 1 GB ceiling. The base tests pass. |
| MBPP+ | 596 | The plus results are frozen `sys.getsizeof` integers. The same object is 88 bytes on Linux and 96 bytes on Windows. The base tests call `sys.getsizeof` on both sides, so they match. |

A scan of the pinned HumanEval+ and MBPP+ rows for `sys.getsizeof`, `id(`, `hash(`, and memory-address expectations found no other task to exclude. HumanEval/39 calls `random` inside its reference primality test. Its expected values are fixed integers, so it stays scored.

On Linux the child also sets `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE`, and `RLIMIT_NOFILE`. It does not create a network namespace.

On Windows the child creates a job object when the process can: process and job memory caps, an active-process cap, and kill-on-close. If the job cannot be assigned, which happens when the process is already in a job, isolation falls back to the wall-clock timeout and the socket block. That is a process boundary, not a hypervisor and not a network namespace.

`verify-reference` is expected to pass every checked HumanEval+ and MBPP+ task on Linux and on Windows. The Windows report was Python 3.12, offline: MBPP+ was 375/377 before this scorer. Task 590 returns a tuple of a float pair and a complex. The last float digit and the tiny imaginary part differ across libm builds. That task stays scored. The plus assertion uses the tolerant compare above, so those differences pass and a wrong value still fails. The packaged MBPP+ fixture includes the real EvalPlus task 590 (Apache-2.0). `verify-reference --fixtures` runs that plus test, and the Windows CI job runs that command. Task 596 is the `sys.getsizeof` exclusion, on every platform, so the checked count is 376 and the pass count is 376. Re-score a finished run without loading a model:

```bash
python -m standard_bench rescore --run-dir runs/<id> --data datasets/standard
```

That writes `runs/<id>-rescored` and leaves the source run in place. `--in-place` copies `items.jsonl` and `run.json` to `.bak` files, then overwrites the run. The scorecard and the CSV record `scorer_version` and `scorer_commit`. `scorer_commit` is the git HEAD, or the package version plus the scorer version when `git rev-parse` cannot answer.

`describe_isolation()` returns the text for the current platform. The suite summary stores it on code-eval runs.
