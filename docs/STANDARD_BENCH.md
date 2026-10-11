# Standard capability benchmarks

`standard_capability` is a native preset. It does not install lm-evaluation-harness, EvalPlus, or another eval framework. Prompting and scoring live in `standard_bench/`. A stage run is offline. The only network path is `python -m standard_bench fetch` on a connected machine.

Scores are deterministic. There is no LLM judge. Public items use `category` `public_benchmark` and `counts_toward_dow_score` false. `aieval_dow_bench_score` ignores these suites even if those flags are wrong. They do not enter the AIEval DoW Bench Score.

Full sizes, with no `--limit`:

| Suite | Items |
| --- | --- |
| mmlu_pro | 12032 |
| gpqa_diamond | 198 |
| gsm8k | 1319 |
| math500 | 500 |
| ifeval | 541 |
| humaneval_plus | 164 |
| mbpp_plus | 378 |

The total is 15132. `--limit N` keeps the first N of each suite after an optional `--seed` shuffle. `--limits mmlu_pro=200,gsm8k=200` sets per-suite caps. Over-budget prompts are not passes. Accuracy uses `n_scored`, the items that were generated.

## MMLU-Pro

Reference: TIGER-Lab/MMLU-Pro and `evaluate_from_local.py` in TIGER-AI-Lab/MMLU-Pro, 5-shot chain of thought, default ntrain 5. Paper: Wang et al., arXiv:2406.01574.

The prompt names the category, then five validation shots from that category, then the question. Options are `A. text` through `J. text`. Shot text replaces `A: Let's think step by step.` with `Answer: Let's think step by step.` The test item ends with `Answer: Let's think step by step.`

Extraction follows the official order. `answer is \(?([A-J])\)?` wins. Then `.*[aA]nswer:\s*([A-J])`. Then the last standalone `A` through `J`. Letters are case sensitive.

Deviation: a failed extraction is incorrect. The official local script draws a random letter. This runner does not. The official script also drops shots until the prompt fits 4096 tokens. This runner keeps 5 shots and records `prompt_over_budget` when the prompt exceeds the budget.

## GPQA Diamond

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

HumanEval base tests come from openai/openai_humaneval at `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`. Plus tests come from evalplus/humanevalplus. The chat wrapper is `Complete the following Python function. Reply with the code only.` Stop strings after generation are a newline plus `class`, `def`, `#`, `if`, or `print`. If the completion contains `def <entry>`, the program starts there. Otherwise the official prompt is prepended.

MBPP+ uses the EvalPlus task line, the statement, and the three base asserts. Stop strings are a newline plus a triple quote, `assert`, `class`, or `def`. The function header from the dataset is the prefix when the completion does not define the entry point.

Deviation: the EvalPlus harness imports numpy. Fetch rewrites that helper to `math.isclose` with atol 1e-6 and rtol 1e-7. A set-comparison variant is kept when the original assertion does `out = set(out)`. The run does not import numpy or evalplus. Canonical solutions are not stored.

## Code sandbox

Each test runs in `python -I` with a JSON payload on stdin. The parent enforces a wall-clock timeout. The child blocks `socket.socket`, `create_connection`, and `getaddrinfo` before the program runs.

On Linux the child also sets `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE`, and `RLIMIT_NOFILE`. It does not create a network namespace.

On Windows the child creates a job object when the process can: process and job memory caps, an active-process cap, and kill-on-close. If the job cannot be assigned, which happens when the process is already in a job, isolation falls back to the wall-clock timeout and the socket block. That is a process boundary, not a hypervisor and not a network namespace.

`describe_isolation()` returns the text for the current platform. The suite summary stores it on code-eval runs.
