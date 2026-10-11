"""Pinned public benchmarks. Revisions were read from the Hub on 2026-10-10.

The runner does not trust these names at score time. It trusts the SHA-256
manifest written by ``python -m standard_bench fetch``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    title: str
    repo: str
    revision: str
    license_id: str
    license_note: str
    full_size: int
    reference: str
    deviation: str
    gated: bool = False
    config: str = "default"
    split: str = "test"
    answer_cap: int = 1024
    prompt_budget: int = 4096
    # Direct file at the pinned revision, when the Hub stores jsonl or csv.
    direct_file: str = ""
    rows_config: str = ""
    rows_split: str = ""


SUITES: dict[str, SuiteSpec] = {
    "mmlu_pro": SuiteSpec(
        name="mmlu_pro",
        title="MMLU-Pro",
        repo="TIGER-Lab/MMLU-Pro",
        revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        license_id="mit",
        license_note="Hugging Face dataset card license is MIT.",
        full_size=12032,
        reference=(
            "TIGER-Lab/MMLU-Pro and the evaluator in "
            "TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py (5-shot CoT, default --ntrain 5). "
            "Paper: Wang et al., MMLU-Pro, arXiv:2406.01574."
        ),
        deviation=(
            "Failed answer extraction is scored incorrect. The official local script "
            "draws a random letter instead. This runner does not draw a letter. "
            "The official script also drops shots until the prompt fits 4096 tokens. "
            "This runner keeps 5 shots and records prompt_over_budget when the prompt "
            "exceeds the budget. It does not shorten the prompt."
        ),
        rows_config="default",
        rows_split="test",
        answer_cap=1024,
        prompt_budget=4096,
    ),
    "gpqa_diamond": SuiteSpec(
        name="gpqa_diamond",
        title="GPQA Diamond",
        repo="Idavidrein/gpqa",
        revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
        license_id="cc-by-4.0",
        license_note=(
            "Dataset card license is CC BY 4.0. The authors ask that questions not be "
            "republished in plain text, so the items stay out of this repository, "
            "fixtures, logs, and exports. The Hub dataset is gated."
        ),
        full_size=198,
        reference=(
            "Rein et al., GPQA, arXiv:2311.12022. Diamond is the 198-question subset "
            "in gpqa_diamond.csv. The prompt and letter extraction follow the "
            "multiple-choice template used by OpenAI simple-evals (Answer: $LETTER)."
        ),
        deviation=(
            "Options are shuffled with random.Random seeded by the record id and "
            "the fixed seed 20261010, so the correct letter is not always A and a "
            "rerun assigns the same letters. Question text is never written to "
            "items, logs, or the CSV."
        ),
        gated=True,
        direct_file="gpqa_diamond.csv",
        answer_cap=1024,
        prompt_budget=4096,
    ),
    "gsm8k": SuiteSpec(
        name="gsm8k",
        title="GSM8K",
        repo="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        license_id="mit",
        license_note="Hugging Face dataset card license is MIT. Split is main/test.",
        full_size=1319,
        reference=(
            "Cobbe et al., Training Verifiers to Solve Math Word Problems, "
            "arXiv:2110.14168. The 8-shot chain of thought is the gsm8k_cot exemplar "
            "set published with EleutherAI lm-evaluation-harness (the eight public "
            "training exemplars, not the test split)."
        ),
        deviation=(
            "The reported score is exact match of the normalized final number. "
            "A number after 'The answer is' wins. Otherwise the last number in the "
            "reply is used. Commas and currency marks are removed. 8.0 matches 8. "
            "There is no calculator annotation and no majority vote."
        ),
        config="main",
        rows_config="main",
        rows_split="test",
        answer_cap=512,
        prompt_budget=2048,
    ),
    "math500": SuiteSpec(
        name="math500",
        title="MATH-500",
        repo="HuggingFaceH4/MATH-500",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        license_id="mit",
        license_note=(
            "The Hugging Face card at this revision does not name a license. "
            "The problems are the 500-problem MATH split from OpenAI prm800k "
            "(Let's Verify Step by Step). Upstream Hendrycks MATH is MIT."
        ),
        full_size=500,
        reference=(
            "Hendrycks et al., Measuring Mathematical Problem Solving With the MATH "
            "Dataset, arXiv:2103.03874. The 500-problem split is the one in "
            "HuggingFaceH4/MATH-500, taken from OpenAI's prm800k MATH split. "
            "LightEval and Minerva prompt the model to finish with \\boxed{}."
        ),
        deviation=(
            "0-shot boxed prompt, not the original MATH 4-shot prompt. "
            "Equivalence is a sympy-free normalizer: latex cleanup, fraction "
            "expansion, and numeric comparison within 1e-6. It does not simplify "
            "symbolic expressions, so x+1 and 1+x do not match."
        ),
        direct_file="test.jsonl",
        answer_cap=1024,
        prompt_budget=2048,
    ),
    "ifeval": SuiteSpec(
        name="ifeval",
        title="IFEval",
        repo="google/IFEval",
        revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
        license_id="apache-2.0",
        license_note="Hugging Face dataset card license is Apache-2.0.",
        full_size=541,
        reference=(
            "Zhou et al., Instruction-Following Evaluation for Large Language Models, "
            "arXiv:2311.07911. Checkers follow google-research "
            "instruction_following_eval (25 instruction ids, strict and loose, "
            "prompt-level and instruction-level)."
        ),
        deviation=(
            "Sentence splits use the regex splitter from instructions_util, not nltk "
            "punkt, so the run does not download nltk data. Word counts use a \\w+ "
            "regex, which matches nltk's RegexpTokenizer. Capital-word counts treat "
            "a hyphenated all-caps word as one word. response_language and the "
            "English case checks use a local script and stopword detector instead "
            "of langdetect. An uncertain language fails the check. The official "
            "langdetect exception path counts the instruction as followed."
        ),
        direct_file="ifeval_input_data.jsonl",
        answer_cap=1024,
        prompt_budget=2048,
    ),
    "humaneval_plus": SuiteSpec(
        name="humaneval_plus",
        title="HumanEval+",
        repo="evalplus/humanevalplus",
        revision="d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        license_id="apache-2.0",
        license_note=(
            "EvalPlus packaging on the Hub is Apache-2.0. The underlying HumanEval "
            "problems (openai/openai_humaneval) are MIT."
        ),
        full_size=164,
        reference=(
            "Chen et al., Evaluating Large Language Models Trained on Code, "
            "arXiv:2107.03374, and Liu et al., Is Your Code Generated by ChatGPT "
            "Really Correct?, arXiv:2305.01210 (EvalPlus). Base tests come from "
            "openai/openai_humaneval at 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544. "
            "Plus tests come from evalplus/humanevalplus. Metric is pass@1 with "
            "greedy decoding."
        ),
        deviation=(
            "Chat models get a one-line completion instruction around the official "
            "HumanEval prompt. Stop strings are applied after generation. "
            "The EvalPlus plus harness imports numpy. The plus assertion is a stdlib "
            "compare: math.isclose and cmath.isclose for floats and complexes inside "
            "tuples, lists, and dicts, with abs_tol 1e-6 and rel_tol 1e-7. Other "
            "values stay exact. The run does not import numpy or evalplus."
        ),
        direct_file="test.jsonl",
        answer_cap=512,
        prompt_budget=2048,
    ),
    "mbpp_plus": SuiteSpec(
        name="mbpp_plus",
        title="MBPP+",
        repo="evalplus/mbppplus",
        revision="b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        license_id="apache-2.0",
        license_note=(
            "EvalPlus packaging on the Hub is Apache-2.0. The underlying MBPP "
            "problems are CC BY 4.0 (Google Research, Mostly Basic Python Problems)."
        ),
        full_size=378,
        reference=(
            "Austin et al., Program Synthesis with Large Language Models, "
            "arXiv:2108.07732 (MBPP), and Liu et al., EvalPlus, arXiv:2305.01210. "
            "This is the 378-task EvalPlus MBPP+ set. Base checks are the three "
            "asserts in test_list. Plus checks are the EvalPlus test body. "
            "Metric is pass@1 with greedy decoding."
        ),
        deviation=(
            "The prompt is the EvalPlus chat task: the natural-language statement, "
            "the three base asserts, and the header of the function those asserts "
            "call. The entry point is that function, not an earlier helper. "
            "Stop strings are applied after generation. "
            "The numpy assertion helper is rewritten to the same stdlib comparison "
            "used for HumanEval+. A custom assertion that does not call is_floats "
            "or allclose is kept, and assert exact_match is added when it is missing. "
            "Scoring rewrites it again so an older pin uses the current helper."
        ),
        rows_config="default",
        rows_split="test",
        answer_cap=512,
        prompt_budget=2048,
    ),
}


def suite_spec(name: str) -> SuiteSpec:
    key = (name or "").strip()
    if key not in SUITES:
        known = ", ".join(SUITES)
        raise KeyError(f"Unknown standard suite '{name}'. Known suites: {known}.")
    return SUITES[key]
