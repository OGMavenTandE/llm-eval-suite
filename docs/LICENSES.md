# Benchmark licenses

These rows are the datasets behind the `standard_capability` preset. The suite does not vendor the files. `python -m standard_bench fetch` downloads them on a connected machine and writes SHA-256 hashes. Revisions are the Hub commits recorded on 2026-10-10.

GPQA questions are not in this repository. Do not put them in fixtures, logs, or exports. The authors ask that the questions not be republished in plain text. The Hub dataset is gated. Fetch uses `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from the environment and does not store the token.

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
