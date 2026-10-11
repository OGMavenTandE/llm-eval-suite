# Benchmark licenses

These rows are the datasets behind the capability presets. The customer kit vendors the six public suites and a SHA-256 manifest. Revisions are the Hub commits recorded on 2026-10-10.

GPQA Diamond is in-house only. It is not in the customer presets or the customer zip. Do not republish the questions in plain text. The authors ask that the questions not be republished. The Hub dataset is gated.

| Suite | Repository | Revision | License | Terms |
| --- | --- | --- | --- | --- |
| MMLU-Pro | TIGER-Lab/MMLU-Pro | b189ec765aa7ed75c8acfea42df31fdae71f97be | mit | Hugging Face dataset card license is MIT. |
| GPQA Diamond | Idavidrein/gpqa | 83022cefff930aea54f654c0b282e74b9eeda5c6 | cc-by-4.0 | Dataset card license is CC BY 4.0. Gated. Do not republish questions in plain text. |
| GSM8K | openai/gsm8k | 740312add88f781978c0658806c59bc2815b9866 | mit | Hugging Face dataset card license is MIT. Split is main/test. |
| MATH-500 | HuggingFaceH4/MATH-500 | 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be | mit | The card at this revision does not name a license. Upstream Hendrycks MATH is MIT. The 500 problems are the MATH split from OpenAI prm800k. |
| IFEval | google/IFEval | 966cd89545d6b6acfd7638bc708b98261ca58e84 | apache-2.0 | Hugging Face dataset card license is Apache-2.0. |
| HumanEval+ | evalplus/humanevalplus | d32357cf319e50e9c8d8dab5ea876c72b0fd321b | apache-2.0 | EvalPlus packaging is Apache-2.0. Base problems from openai/openai_humaneval at 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544 are MIT. |
| MBPP+ | evalplus/mbppplus | b2d74c91837c3f2a20c1299ae98133cbe7cfa077 | apache-2.0 | EvalPlus packaging is Apache-2.0. The underlying MBPP problems are CC BY 4.0. |

SWE-bench is out of scope and is not fetched.

The HumanEval+ and MBPP+ files in the kit include the reference solution for each task, under the same license as that suite. `python -m standard_bench verify-reference` runs those solutions on the pinned tests and requires every checked one to pass. Three tasks are excluded. HumanEval/32 is excluded because its plus test star-unpacks the float returned by `find_zero` (`_poly(*candidate(*inp), inp)`), which raises TypeError. The reference solution passes the base tests. MBPP+ task 255 is excluded because the plus test and the reference solution both build `combinations_with_replacement` for a length-77 sequence, which exceeds the 1 GB address space. The base tests pass. MBPP+ task 596 is excluded because the plus results are frozen `sys.getsizeof` integers. The same object is 88 bytes on Linux and 96 bytes on Windows. The base tests call `sys.getsizeof` on both sides, so they match. A scan of the pinned HumanEval+ and MBPP+ rows for `sys.getsizeof`, `id(`, `hash(`, and memory-address expectations found no other task to exclude. Those three rows stay in the files. Suite metrics record them in `reference_exclusions`.

JSONL readers split on newline only. `str.splitlines()` also splits on U+0085, and MMLU-Pro question 5215 contains that character.
