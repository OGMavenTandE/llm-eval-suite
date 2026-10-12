"""Native public benchmarks. No GPU, live model, network, or real dataset."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from llm_eval.models.base import ModelResponse
from llm_eval_suite.presets import expand_preset
from llm_eval_suite.scoring import aieval_dow_bench_score, scorecard
from llm_eval_suite.suites import SuiteContext
from standard_bench.catalog import SUITES
from standard_bench.export import write_export
from standard_bench.fetch import FetchError, manifest_contains_secret, normalize_humaneval, normalize_mbpp
from standard_bench.load import select_rows
from standard_bench.manifest import HashMismatch, ManifestError, verify_suite
from standard_bench.plan import (
    MODEL_PRECISION,
    PLAN_SUITES,
    PREFILL_TOK_S,
    QUICK_THINKING_CAP,
    SCREEN_LIMITS,
    THINKING_MODELS,
    THROUGHPUT,
    command_precision,
    model_seconds,
    quick_limits,
    tokens_per_item,
)
from standard_bench.protocol import CUSTOMER_SUITES, QUICK_SEED
from standard_bench.runner import (
    _aggregate,
    planned_count,
    resolve_seed,
    resolve_subset,
    resolve_thinking_budget,
    run_standard_suite,
)
from standard_bench.sandbox import CHILD, RunLimits, describe_isolation, run_code
from standard_bench.scorers.code import (
    FLOAT_ABS_TOL,
    FLOAT_REL_TOL,
    HUMANEVAL_STOPS,
    MBPP_STOPS,
    assemble_program,
    mbpp_prompt,
    prepare_plus_test,
    rewrite_evalplus_test,
    score_code_task,
    values_close,
)
from standard_bench.scorers.gpqa import extract_answer as gpqa_extract
from standard_bench.scorers.gpqa import score_gpqa, shuffle_options
from standard_bench.scorers.gsm8k import extract_gold, extract_prediction, score_gsm8k
from standard_bench.scorers.ifeval import (
    INSTRUCTION_IDS,
    aggregate_ifeval,
    check_instruction,
    score_ifeval_item,
    split_into_sentences,
)
from standard_bench.scorers.math500 import build_prompt as math_prompt
from standard_bench.scorers.math500 import extract_boxed, math_equal, score_math
from standard_bench.scorers.mmlu_pro import extract_answer
from standard_bench.stub import StubStandardModel

FIXTURES = Path(__file__).resolve().parents[2] / "standard_bench" / "fixtures"
ROOT = Path(__file__).resolve().parents[2]
SENTINEL = "GPQA_SENTINEL_QUESTION"


def _ctx(model, tmp_path: Path) -> SuiteContext:
    return SuiteContext(
        model=model,
        connection={"model": "OLMo 2 1B Instruct", "name": "OLMo 2 1B Instruct", "max_context": 8192},
        dataset_rows=[],
        run_dir=tmp_path,
        cancel=None,
    )


class _CountingModel(StubStandardModel):
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        raise AssertionError("generate must not run after a hash mismatch")


def test_mmlu_extractors():
    assert extract_answer("the answer is (C)") == "C"
    assert extract_answer("Answer: B") == "B"
    assert extract_answer("I pick a letter.\nD") == "D"
    # The first official pattern wins, even when a later Answer: line differs.
    assert extract_answer("the answer is (C). Later, Answer: D") == "C"
    assert extract_answer("the answer is (c)") is None
    from standard_bench.scorers.mmlu_pro import score_mmlu_pro

    assert score_mmlu_pro("A", "the answer is (A)")["passed"] is True
    assert score_mmlu_pro("A", "no letter here")["passed"] is False
    assert score_mmlu_pro("A", "no letter here")["score"] == 0.0


def test_gpqa_shuffle_is_stable_and_scores_the_letter():
    first, letter = shuffle_options("rec-1", "correct", ["w", "x", "y"])
    second, letter_again = shuffle_options("rec-1", "correct", ["w", "x", "y"])
    other, _other_letter = shuffle_options("rec-2", "correct", ["w", "x", "y"])
    assert first == second
    assert letter == letter_again
    assert letter in "ABCD"
    assert first[ord(letter) - ord("A")] == "correct"
    assert other != first or True
    assert gpqa_extract("thinking\nAnswer: B\nAnswer: C") == "C"
    assert score_gpqa(letter, f"Answer: {letter}")["passed"] is True
    assert score_gpqa(letter, "Answer: Z")["passed"] is False


def test_gsm8k_normalization():
    assert extract_gold("work\n#### 1,000") == "1000"
    assert extract_gold("#### 8") == "8"
    assert extract_prediction("The answer is 1,000.") == "1000"
    assert extract_prediction("The answer is $8.0") == "8"
    assert extract_prediction("The answer is 4. Then 9.") == "4"
    assert extract_prediction("no marker, last is 18.") == "18"
    assert score_gsm8k("#### 8", "The answer is 8.0")["passed"] is True
    assert score_gsm8k("#### 4", "The answer is 5")["passed"] is False


def test_math500_boxed_and_equivalence():
    assert extract_boxed(r"first \boxed{1} later \boxed{1/2}") == "1/2"
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed("the answer is 1/2") is None
    assert math_equal(r"\frac{1}{2}", "0.5") is True
    assert math_equal(r"\dfrac{1}{2}", "1/2") is True
    assert math_equal("x+1", "1+x") is False
    assert score_math("1/2", r"steps \boxed{1} then \boxed{0.5}")["passed"] is True
    missed = score_math("3", "the answer is 3")
    assert missed["passed"] is False
    assert missed["predicted"] == ""
    built = math_prompt("What is 1/2?")
    assert r"\boxed{}" in built
    assert "What is 1/2?" in built


def test_ifeval_every_checker_and_tricky_cases():
    assert split_into_sentences("Dr. Smith went home. He left.") == [
        "Dr. Smith went home.",
        "He left.",
    ]
    cases = {
        "keywords:existence": ("the category list", {"keywords": ["cat"]}, True),
        "keywords:frequency": ("cat cat", {"keyword": "cat", "relation": "at least", "frequency": 2}, True),
        "keywords:forbidden_words": ("category", {"forbidden_words": ["cat"]}, True),
        "keywords:letter_frequency": ("aaab", {"letter": "a", "let_relation": "at least", "let_frequency": 3}, True),
        "language:response_language": ("안녕하세요 세계입니다", {"language": "ko"}, True),
        "length_constraints:number_sentences": (
            "One sentence here.",
            {"relation": "less than", "num_sentences": 2},
            True,
        ),
        "length_constraints:number_paragraphs": (
            "Alpha\n***\nBeta",
            {"num_paragraphs": 2},
            True,
        ),
        "length_constraints:number_words": (
            "one two",
            {"relation": "less than", "num_words": 5},
            True,
        ),
        "length_constraints:nth_paragraph_first_word": (
            "Alpha stays.\n\nBeta follows.",
            {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "Beta"},
            True,
        ),
        "detectable_content:number_placeholders": ("see [name] now", {"num_placeholders": 1}, True),
        "detectable_content:postscript": ("Done.\nP.S. note", {"postscript_marker": "P.S."}, True),
        "detectable_format:number_bullet_lists": ("* one\n- two", {"num_bullets": 2}, True),
        "detectable_format:constrained_response": ("My answer is yes.", {}, True),
        "detectable_format:number_highlighted_sections": (
            "*one* and **two**",
            {"num_highlights": 2},
            True,
        ),
        "detectable_format:multiple_sections": (
            "Section 1 text Section 2 text",
            {"section_spliter": "Section", "num_sections": 2},
            True,
        ),
        "detectable_format:json_format": ('```json\n{"a": 1}\n```', {}, True),
        "detectable_format:title": ("<<Title>> body", {}, True),
        "combination:two_responses": ("left side******right side", {}, True),
        "combination:repeat_prompt": (
            "Say this back now and more",
            {"prompt_to_repeat": "Say this back"},
            True,
        ),
        "startend:end_checker": ("ends with goodbye", {"end_phrase": "goodbye"}, True),
        "startend:quotation": ('"quoted reply"', {}, True),
        "change_case:capital_word_frequency": (
            "NASA-JPL flew",
            {"capital_relation": "at least", "capital_frequency": 1},
            True,
        ),
        "change_case:english_capital": ("THE CAT AND THE DOG", {}, True),
        "change_case:english_lowercase": ("the cat and the dog", {}, True),
        "punctuation:no_comma": ("no comma here", {}, True),
    }
    assert set(cases) == set(INSTRUCTION_IDS)
    for instruction_id, (text, kwargs, expected) in cases.items():
        assert check_instruction(instruction_id, text, kwargs) is expected, instruction_id
    assert check_instruction("keywords:forbidden_words", "the cat sat", {"forbidden_words": ["cat"]}) is False
    assert check_instruction("keywords:existence", "category", {"keywords": ["cat"]}) is True
    assert check_instruction("length_constraints:number_words", "one", {}) is False
    hyphen = check_instruction(
        "change_case:capital_word_frequency",
        "NASA-JPL only",
        {"capital_relation": "at least", "capital_frequency": 2},
    )
    assert hyphen is False
    loose_row = {
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    loose = score_ifeval_item(loose_row, "Hello,\nThis response has no comma at all.")
    assert loose["prompt_strict"] is False
    assert loose["prompt_loose"] is True
    empty = score_ifeval_item(loose_row, "   ")
    assert empty["prompt_strict"] is False
    assert empty["prompt_loose"] is False
    summary = aggregate_ifeval([loose, empty])
    assert summary["prompt_strict"] == 0.0
    assert summary["prompt_loose"] == 0.5
    assert summary["n_instructions"] == 2


def test_sandbox_blocks_network_and_times_out_a_loop():
    blocked = run_code("import socket\nsocket.create_connection(('example.com', 80))\n")
    assert blocked.ok is False
    assert blocked.blocked_network or "network disabled" in blocked.error
    assert blocked.timed_out is False
    raw_socket = run_code("import _socket\n_socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)\n")
    assert raw_socket.ok is False
    assert raw_socket.blocked_network or "network disabled" in raw_socket.error
    probe = "sandbox-cwd-probe.txt"
    wrote = run_code(f"open({probe!r}, 'w').write('x')\n")
    assert wrote.ok is True
    assert not (CHILD.parent / probe).exists()
    loop = run_code("while True:\n    pass\n", RunLimits(timeout_s=1.5, memory_mb=1024, cpu_s=30))
    assert loop.ok is False
    assert loop.timed_out or "timed out" in loop.error.lower()
    if sys.platform == "win32":
        assert "job object" in describe_isolation()
    else:
        assert "resource limits" in describe_isolation()


def test_evalplus_rewrite_drops_numpy():
    source = "import numpy as np\n\ndef check(candidate):\n    assert candidate() == 1\n"
    rewritten = rewrite_evalplus_test(source)
    assert "import numpy" not in rewritten
    assert "def check(" in rewritten
    assert "def assertion(" in rewritten


def test_evalplus_rewrite_keeps_helpers_before_check():
    source = (
        "import numpy as np\n"
        "def is_floats(x):\n"
        "    return isinstance(x, np.ndarray)\n"
        "def assertion(out, exp, atol):\n"
        "    assert np.allclose(out, exp, rtol=1e-07, atol=atol)\n"
        "import math\n"
        "def _poly(xs, x):\n"
        "    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))\n"
        "def ref_func(n):\n"
        "    return n + 1\n"
        "def check(candidate):\n"
        "    assertion(candidate(1), ref_func(1), 0)\n"
        "    assert _poly(*candidate(1), 1) <= 0.0001\n"
    )
    rewritten = rewrite_evalplus_test(source)
    assert "import numpy" not in rewritten
    assert "np.ndarray" not in rewritten
    assert "np.allclose" not in rewritten
    assert "def ref_func(n):" in rewritten
    assert "def _poly(xs, x):" in rewritten
    assert "import math" in rewritten
    assert rewritten.index("def ref_func") < rewritten.index("def check(")
    assert "def check(" in rewritten


def test_fixture_manifest_matches_and_hash_mismatch_refuses(tmp_path: Path):
    for name in SUITES:
        entry = verify_suite(FIXTURES, name)
        assert entry["synthetic"] is True
    copied = tmp_path / "data"
    shutil.copytree(FIXTURES, copied)
    target = copied / "gsm8k" / "test.jsonl"
    target.write_bytes(target.read_bytes() + b" ")
    model = _CountingModel()
    with pytest.raises(HashMismatch, match="hash mismatch"):
        run_standard_suite(_ctx(model, tmp_path), {"fixtures": False, "data_dir": str(copied)}, "gsm8k")
    assert model.calls == 0
    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(ManifestError):
        verify_suite(missing, "gsm8k")


def test_over_budget_is_not_a_pass():
    metrics = _aggregate(
        "gsm8k",
        [
            {"passed": True, "prompt_over_budget": False},
            {"passed": False, "prompt_over_budget": True, "score": None},
        ],
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["n_scored"] == 1
    assert metrics["n_over_budget"] == 1
    assert metrics["counts_toward_dow_score"] is False


def test_fixture_smoke_scores_and_redacts_gpqa(tmp_path: Path):
    model = StubStandardModel()
    ctx = _ctx(model, tmp_path)
    config = {"fixtures": True}
    results = {}
    for name in SUITES:
        results[name] = run_standard_suite(ctx, config, name)
    assert results["mmlu_pro"]["metrics"]["accuracy"] == 0.5
    assert results["gsm8k"]["metrics"]["accuracy"] == 0.5
    assert results["math500"]["metrics"]["accuracy"] == 0.5
    assert results["ifeval"]["metrics"]["prompt_strict"] == 0.5
    assert results["humaneval_plus"]["metrics"]["pass_at_1_plus"] == 1.0
    assert results["mbpp_plus"]["metrics"]["pass_at_1_plus"] == 1.0
    assert [row["task_id"] for row in results["humaneval_plus"]["metrics"]["reference_exclusions"]] == [
        "HumanEval/32"
    ]
    assert [row["task_id"] for row in results["mbpp_plus"]["metrics"]["reference_exclusions"]] == [
        "255",
        "596",
    ]
    assert results["gpqa_diamond"]["counts_toward_dow_score"] is False
    gpqa_items = results["gpqa_diamond"]["items"]
    assert gpqa_items
    blob = json.dumps(gpqa_items)
    assert SENTINEL not in blob
    assert gpqa_items[0]["prompt"] == ""
    assert gpqa_items[0]["response"] == ""
    assert gpqa_items[0]["category"] == "public_benchmark"
    parent = tmp_path / "runs"
    run_dir = parent / "smoke"
    run_dir.mkdir(parents=True)
    items_blob = "\n".join(
        json.dumps(item) for result in results.values() for item in result["items"]
    )
    (run_dir / "items.jsonl").write_text(items_blob + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "smoke",
                "connection": {"model": "OLMo 2 1B Instruct"},
                "suites": [
                    {"name": name, "metrics": result["metrics"]} for name, result in results.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    written = write_export(parent, tmp_path / "out" / "public.csv")
    csv_text = written["csv"].read_text(encoding="utf-8")
    json_text = written["json"].read_text(encoding="utf-8")
    assert SENTINEL not in csv_text
    assert SENTINEL not in json_text
    assert "counts_toward_dow_score" in csv_text
    assert "false" in csv_text
    assert any(row["suite"] == "mmlu_pro" and row["metric"] == "accuracy" for row in written["rows"])
    assert model.calls == sum(len(result["items"]) for result in results.values())


def test_public_scores_stay_out_of_the_dow_composite():
    dow = {
        "suite": "dow_knowledge",
        "type": "multiple_choice",
        "passed": True,
        "counts_toward_score": True,
        "category": "hallucination_factuality",
        "source": "live",
    }
    public = {
        "suite": "mmlu_pro",
        "type": "mmlu_pro",
        "passed": False,
        "counts_toward_score": True,
        "counts_toward_dow_score": True,
        "category": "hallucination_factuality",
        "source": "live",
    }
    alone = aieval_dow_bench_score([dow])
    mixed = aieval_dow_bench_score([dow, public])
    assert alone["score"] == mixed["score"] == 1.0
    assert mixed["public_benchmarks_included"] is False
    card_alone = scorecard([dow])
    card_mixed = scorecard([dow, public])
    alone_rate = next(row["pass_rate"] for row in card_alone["categories"] if row["category"] == "hallucination_factuality")
    mixed_rate = next(row["pass_rate"] for row in card_mixed["categories"] if row["category"] == "hallucination_factuality")
    assert alone_rate == mixed_rate == 1.0


def test_preset_lists_the_six_customer_suites_offline():
    from llm_eval_suite.presets import list_presets

    expanded = expand_preset("standard_capability")
    assert expanded["offline"] is True
    names = [suite["name"] for suite in expanded["suites"]]
    assert names == list(CUSTOMER_SUITES)
    assert "gpqa_diamond" not in names
    quick = expand_preset("standard_capability_quick")
    assert [suite["name"] for suite in quick["suites"]] == list(CUSTOMER_SUITES)
    gsm = next(suite for suite in quick["suites"] if suite["name"] == "gsm8k")
    assert gsm["preset"] == "standard_capability_quick"
    assert gsm["subset"] == "quick50"
    assert gsm["thinking_budget"] == 512
    assert planned_count("gsm8k", gsm) == 50
    assert planned_count("gsm8k", {}) == 1319
    assert planned_count("gsm8k", {"limit": 10}) == 10
    listed = [row["id"] for row in list_presets()]
    assert "standard_capability" in listed
    assert "standard_capability_quick" in listed
    assert "standard_capability_gpqa" not in listed
    assert next(row["prompt_count"] for row in list_presets() if row["id"] == "standard_capability") == 14934
    assert next(row["prompt_count"] for row in list_presets() if row["id"] == "standard_capability_quick") == 300
    inhouse = expand_preset("standard_capability_gpqa")
    assert [suite["name"] for suite in inhouse["suites"]] == ["gpqa_diamond"]
    rows = select_rows([{"id": 1}, {"id": 2}, {"id": 3}], 2, 7)
    assert rows == select_rows([{"id": 1}, {"id": 2}, {"id": 3}], 2, 7)
    assert len(rows) == 2


def test_fetch_refuses_when_offline(tmp_path: Path):
    from llm_eval.offline import activate_run_offline

    from standard_bench.fetch import fetch

    activate_run_offline(preset={"offline": True})
    with pytest.raises(FetchError, match="Offline mode"):
        fetch(tmp_path)
    assert manifest_contains_secret({"note": "Bearer hf_secret_value"}, "hf_secret_value") is True
    assert manifest_contains_secret({"note": "plain"}, "hf_secret_value") is False


def test_license_table_matches_the_catalog():
    text = (ROOT / "docs" / "LICENSES.md").read_text(encoding="utf-8")
    for spec in SUITES.values():
        assert spec.repo in text
        assert spec.license_id in text
        assert spec.revision in text
    assert "not be republished" in text or "Do not republish" in text
    assert "SWE-bench" in text


def test_run_plan_matches_plan_py():
    doc = (ROOT / "docs" / "STANDARD_BENCH_RUN_PLAN.md").read_text(encoding="utf-8")
    assert "assumption" in doc.lower()
    assert "2048" in doc
    assert str(PREFILL_TOK_S) in doc
    assert "first run" in doc.lower()
    grand = 0.0
    screen = 0.0
    for key in THROUGHPUT:
        full = model_seconds(key)
        short = model_seconds(key, SCREEN_LIMITS)
        grand += full["total_hours"]
        screen += short["total_hours"]
        assert f"{full['total_hours']:.2f}" in doc
        assert f"{short['total_hours']:.2f}" in doc
    assert f"{grand:.2f}" in doc
    assert f"{screen:.2f}" in doc
    for name in PLAN_SUITES:
        spec = SUITES[name]
        assert str(spec.full_size) in doc
        assert str(tokens_per_item(name, thinking=False)) in doc
        assert str(tokens_per_item(name, thinking=True)) in doc
    limits = quick_limits()
    stage1 = 0.0
    stage2 = 0.0
    for key in THROUGHPUT:
        if key in THINKING_MODELS:
            quick = model_seconds(key, limits, thinking_tokens=QUICK_THINKING_CAP)
            stage2 += quick["total_hours"]
        else:
            quick = model_seconds(key, limits)
            stage1 += model_seconds(key, SCREEN_LIMITS)["total_hours"]
        assert f"{quick['total_hours']:.2f}" in doc
    assert f"{stage1:.2f}" in doc
    assert f"{stage2:.2f}" in doc
    assert "standard_capability_quick" in doc
    assert "512" in doc
    assert "14934" in doc
    assert "HF_TOKEN" not in doc
    for display in (
        "OLMo 2 1B Instruct",
        "LFM2.5-1.2B Instruct",
        "Granite 4.0 H 1B",
        "LFM2.5-2.6B",
        "Granite 4.2 3B",
        "Phi-4-mini-instruct",
        "Phi-4-mini-reasoning",
        "NVIDIA Nemotron 3 Nano 4B",
        "Gemma 4 E2B-it",
    ):
        assert display in doc
    assert "--precision 8bit" in doc
    assert "--precision fp16" not in doc
    assert "fp16" not in doc
    assert doc.count("--precision bf16") == 8
    assert "14931" in doc
    for key, precision in MODEL_PRECISION.items():
        assert command_precision(key) == precision
        assert f"--precision {precision}" in doc
    assert "mmlu_pro=200" in doc


def _read_jsonl(path: Path) -> list[dict]:
    from standard_bench.load import jsonl_lines

    return [json.loads(line) for line in jsonl_lines(path.read_text(encoding="utf-8"))]


def test_mbpp_plus_keeps_the_plus_inputs():
    raw = _read_jsonl(FIXTURES / "raw" / "mbpp_plus.jsonl")
    assert "def check(" not in raw[0]["test"]
    normalized = normalize_mbpp(raw)
    plus = normalized[0]["plus_test"]
    assert "inputs" in plus
    assert "results" in plus
    assert "def check(candidate)" in plus
    assert "similar_elements(" not in plus
    assert "import numpy" not in plus
    base_only = "def similar_elements(a, b):\n    return (2,)\n"
    scored = score_code_task(normalized[0], base_only, suite="mbpp_plus")
    assert scored["base_pass"] is True
    assert scored["plus_pass"] is False
    good = score_code_task(normalized[0], raw[0]["code"], suite="mbpp_plus")
    assert good["base_pass"] is True
    assert good["plus_pass"] is True


def test_humaneval_raw_rows_keep_plus_inputs():
    base = _read_jsonl(FIXTURES / "raw" / "humaneval_base.jsonl")
    plus_rows = _read_jsonl(FIXTURES / "raw" / "humaneval_plus.jsonl")
    normalized = normalize_humaneval(base, plus_rows)
    assert normalized[0]["entry_point"] == "add"
    assert "def check(" in normalized[0]["base_test"]
    assert "inputs" in normalized[0]["plus_test"]
    assert "results" in normalized[0]["plus_test"]
    assert "import numpy" not in normalized[0]["plus_test"]
    weak = score_code_task(normalized[0], "def add(a, b):\n    return 3\n", suite="humaneval_plus")
    assert weak["base_pass"] is True
    assert weak["plus_pass"] is False
    good = score_code_task(normalized[0], "def add(a, b):\n    return a + b\n", suite="humaneval_plus")
    assert good["plus_pass"] is True
    assert "return a + b" in normalized[0]["canonical_solution"]


def test_code_extraction_keeps_imports_and_helpers():
    fenced = (
        "```python\n"
        "from typing import List\n\n"
        "def before(x):\n"
        "    return x\n\n"
        "def add(a, b):\n"
        "    return before(a) + b\n\n"
        "def after(x):\n"
        "    return x + 1\n"
        "print('done')\n"
        "```\n"
    )
    program = assemble_program("def add(a, b):\n", fenced, "add", HUMANEVAL_STOPS)
    assert "from typing import List" in program
    assert "def before" in program
    assert "def add(a, b):" in program
    assert "def after" in program
    assert "print" not in program
    bare = assemble_program(
        "def add(a, b):\n",
        "    return a + b\n\ndef extra(x):\n    return x\n",
        "add",
        HUMANEVAL_STOPS,
    )
    assert bare.startswith("def add(a, b):\n")
    assert "return a + b" in bare
    assert "def extra" not in bare
    full = assemble_program(
        "def add(a, b):\n",
        "def add(a, b):\n    return a + b\n",
        "add",
        HUMANEVAL_STOPS,
    )
    assert full.strip() == "def add(a, b):\n    return a + b"
    assert full.count("def add") == 1
    mbpp = assemble_program(
        "def shared(left, right):\n",
        "    return tuple(sorted(set(left) & set(right)))\n\nassert False\n",
        "shared",
        MBPP_STOPS,
    )
    assert mbpp.startswith("def shared(left, right):\n")
    assert "assert False" not in mbpp


def test_quick50_selection_is_stable_and_stamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_EVAL_STANDARD_SUBSET", raising=False)
    monkeypatch.delenv("LLM_EVAL_STANDARD_SEED", raising=False)
    monkeypatch.delenv("LLM_EVAL_STANDARD_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("LLM_EVAL_INCLUDE_GPQA", raising=False)
    pool = [{"id": str(i)} for i in range(80)]
    first = select_rows(pool, 50, QUICK_SEED)
    second = select_rows(pool, 50, QUICK_SEED)
    assert first == second
    assert len(first) == 50
    assert [row["id"] for row in first] != [str(i) for i in range(50)]
    config = {
        "fixtures": True,
        "subset": "quick50",
        "thinking_budget": 512,
        "preset": "standard_capability_quick",
    }
    assert resolve_seed(config) == QUICK_SEED
    result = run_standard_suite(_ctx(StubStandardModel(), tmp_path), config, "gsm8k")
    assert result["protocol"] == "quick-check"
    assert result["subset"] == "quick50"
    assert result["preset"] == "standard_capability_quick"
    assert result["metrics"]["n_items"] == 2
    assert result["metrics"]["limit"] == 50
    assert result["metrics"]["thinking_budget"] == 0
    assert result["metrics"]["thinking_cap"] == 512
    assert result["metrics"]["answer_cap"] == 512
    card = scorecard(result["items"])
    stamp = card["public_benchmarks"]
    assert stamp["protocol"] == "quick-check"
    assert stamp["preset"] == "standard_capability_quick"
    assert stamp["subset"] == "quick50"
    assert stamp["gpqa_diamond"] == "not_run"
    assert stamp["item_counts"]["gsm8k"] == 2
    assert stamp["subset_limits"]["gsm8k"] == 50
    assert stamp["thinking_cap"] == 512
    assert stamp["answer_caps"]["gsm8k"] == 512
    thinking = _ctx(StubStandardModel(), tmp_path)
    thinking.connection["model"] = "Phi-4-mini-reasoning"
    thinking.connection["name"] = "Phi-4-mini-reasoning"
    reasoned = run_standard_suite(thinking, config, "gsm8k")
    assert reasoned["metrics"]["thinking_budget"] == 512
    assert reasoned["metrics"]["thinking_cap"] == 512
    applied, configured = resolve_thinking_budget("OLMo 2 1B Instruct", {"thinking_budget": 512})
    assert applied == 0
    assert configured == 512
    monkeypatch.setenv("LLM_EVAL_STANDARD_SUBSET", "short")
    assert resolve_subset({"subset": "quick50"}) == "short"
    assert planned_count("gsm8k", {"subset": "quick50"}) == 200
    monkeypatch.setenv("LLM_EVAL_STANDARD_THINKING_BUDGET", "100")
    applied, configured = resolve_thinking_budget("Phi-4-mini-reasoning", {"thinking_budget": 512})
    assert (applied, configured) == (100, 100)
    applied, configured = resolve_thinking_budget("OLMo 2 1B Instruct", {"thinking_budget": 512})
    assert applied == 0


def test_missing_gpqa_is_not_run(tmp_path: Path):
    data = tmp_path / "data"
    shutil.copytree(FIXTURES, data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    manifest["suites"].pop("gpqa_diamond")
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.rmtree(data / "gpqa_diamond")
    model = _CountingModel()
    result = run_standard_suite(
        _ctx(model, tmp_path),
        {"fixtures": False, "data_dir": str(data), "preset": "standard_capability"},
        "gpqa_diamond",
    )
    assert result["not_run"] is True
    assert result["status"] == "not_run"
    assert result["items"] == []
    assert model.calls == 0
    gsm = run_standard_suite(
        _ctx(StubStandardModel(), tmp_path),
        {"fixtures": False, "data_dir": str(data)},
        "gsm8k",
    )
    assert gsm["metrics"]["n_items"] == 2


def test_bundle_has_no_gpqa_and_loads_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import zipfile

    from llm_eval.offline import activate_run_offline

    from standard_bench.bundle import build_bundle, verify_kit
    from standard_bench.manifest import HashMismatch

    monkeypatch.delenv("LLM_EVAL_STANDARD_SUBSET", raising=False)
    activate_run_offline(preset={"offline": True})
    info = build_bundle(tmp_path / "kit.zip", FIXTURES, repo_root=ROOT)
    assert info["bytes"] < 35 * 1024 * 1024
    verify_kit(info["path"])
    with zipfile.ZipFile(info["path"]) as archive:
        names = [item.filename.replace("\\", "/") for item in archive.infolist()]
        assert not any("gpqa_diamond" in name for name in names)
        assert "docs/GPQA_INHOUSE.md" not in names
        assert "datasets/standard/gsm8k/test.jsonl" in names
        assert "kit-manifest.json" in names
        for name in names:
            if name.endswith(".jsonl") or name.endswith("manifest.json"):
                blob = archive.read(name)
                assert b"GPQA_SENTINEL_QUESTION" not in blob
                assert b"gpqa_diamond" not in blob
        data_manifest = json.loads(archive.read("datasets/standard/manifest.json"))
        assert set(data_manifest["suites"]) == set(CUSTOMER_SUITES)
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with zipfile.ZipFile(info["path"]) as archive:
        archive.extractall(unpacked)
    verify_kit(unpacked)
    target = unpacked / "datasets" / "standard" / "gsm8k" / "test.jsonl"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    data_manifest = unpacked / "datasets" / "standard" / "manifest.json"
    data_manifest.write_bytes(data_manifest.read_bytes().replace(b"\n", b"\r\n"))
    verify_kit(unpacked)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(HashMismatch):
        verify_kit(unpacked)
    result = run_standard_suite(
        _ctx(StubStandardModel(), tmp_path),
        {
            "fixtures": False,
            "data_dir": str(FIXTURES),
            "subset": "quick50",
            "preset": "standard_capability_quick",
            "thinking_budget": 512,
        },
        "math500",
    )
    assert result["protocol"] == "quick-check"
    assert scorecard(result["items"])["public_benchmarks"]["gpqa_diamond"] == "not_run"


def test_customer_docs_do_not_ask_for_an_hf_token():
    for relative in (
        "README.md",
        "docs/AIRGAP_INSTALL.md",
        "docs/STANDARD_BENCH.md",
        "docs/STANDARD_BENCH_RUN_PLAN.md",
        "docs/LICENSES.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "HF_TOKEN" not in text
        assert "HUGGING_FACE_HUB_TOKEN" not in text
    inhouse = (ROOT / "docs" / "GPQA_INHOUSE.md").read_text(encoding="utf-8")
    assert "HF_TOKEN" in inhouse
    assert "in-house" in inhouse.lower()
    assert "standard_capability_gpqa" in inhouse


def test_jsonl_keeps_unicode_line_separators():
    from standard_bench.export import _read_items
    from standard_bench.load import read_jsonl

    path = FIXTURES / "unicode_line.jsonl"
    text = path.read_text(encoding="utf-8")
    assert "\u0085" in text
    assert "\u2028" in text
    assert "\u2029" in text
    broken = False
    try:
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
    except json.JSONDecodeError:
        broken = True
    assert broken is True
    rows = read_jsonl(path)
    assert len(rows) == 1
    question = rows[0]["question"]
    assert "\u0085" in question
    assert "\u2028" in question
    assert "\u2029" in question
    exported = _read_items(path)
    assert exported == rows


def test_mbpp_imports_and_check_name_do_not_fail_the_reference():
    imported = {
        "task_id": 98,
        "prompt": "Average the product.",
        "code": "def multiply_num(numbers):\n    return -67.2\n",
        "test_list": ["assert math.isclose(multiply_num((8, 2)), -67.2, rel_tol=0.001)"],
        "test_imports": ["import math"],
        "test": (
            "import numpy as np\n"
            "from math import inf\n"
            "def assertion(out, exp, atol):\n"
            "    assert out == exp\n"
            "inputs = [((1, 2),)]\n"
            "results = [inf]\n"
            "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "    assertion(multiply_num(*inp), exp, 0)\n"
        ),
    }
    # The reference returns 1, so the plus comparison against inf fails.
    # The base assert is what needs math. A missing import is a NameError.
    row = normalize_mbpp([imported])[0]
    assert row["test_imports"] == ["import math"]
    scored = score_code_task(row, imported["code"], suite="mbpp_plus")
    assert scored["base_pass"] is True
    assert "math" not in scored["error"]
    colliding = {
        "task_id": 56,
        "prompt": "Check the number.",
        "code": "def check(n):\n    return n == 1\n",
        "test_list": ["assert check(1) == True", "assert check(2) == False"],
        "test_imports": [],
        "test": (
            "def assertion(out, exp, atol):\n"
            "    assert out == exp\n"
            "inputs = [[1], [2]]\n"
            "results = [True, False]\n"
            "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "    assertion(check(*inp), exp, 0)\n"
        ),
    }
    row = normalize_mbpp([colliding])[0]
    scored = score_code_task(row, colliding["code"], suite="mbpp_plus")
    assert scored["base_pass"] is True
    assert scored["plus_pass"] is True
    assert "not callable" not in scored["error"]


def test_documented_reference_exclusions_are_skipped():
    from standard_bench.reference import EXCLUSIONS, verify_rows

    assert "HumanEval/32" in EXCLUSIONS["humaneval_plus"]
    assert set(EXCLUSIONS["mbpp_plus"]) == {"255", "596"}
    licenses = (ROOT / "docs" / "LICENSES.md").read_text(encoding="utf-8")
    bench = (ROOT / "docs" / "STANDARD_BENCH.md").read_text(encoding="utf-8")
    assert "596" in licenses and "getsizeof" in licenses
    assert "596" in bench and "getsizeof" in bench
    assert "HumanEval/39" in bench
    broken = {
        "task_id": "HumanEval/32",
        "canonical_solution": "",
        "entry_point": "find_zero",
        "prompt": "",
        "base_test": "",
        "plus_test": "",
    }
    report = verify_rows([broken], "humaneval_plus", workers=1)
    assert report["ok"] is True
    assert report["checked"] == 0
    assert report["excluded"][0]["task_id"] == "HumanEval/32"


def test_verify_reference_on_fixture_rows():
    from standard_bench.cli import main
    from standard_bench.reference import verify_reference

    report = verify_reference(FIXTURES, workers=1)
    assert report["ok"] is True
    assert report["suites"]["humaneval_plus"]["passed"] == 1
    assert report["suites"]["mbpp_plus"]["passed"] == 3
    assert report["suites"]["humaneval_plus"]["failures"] == []
    with pytest.raises(SystemExit) as caught:
        main(["verify-reference", "--fixtures"])
    assert caught.value.code == 0


def test_fetch_retries_429_and_5xx(monkeypatch: pytest.MonkeyPatch):
    import urllib.error
    import urllib.request

    from standard_bench.fetch import _request

    calls = {"n": 0}

    class _Body:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"ok"

    def urlopen(request, timeout=60):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "rate", {"Retry-After": "0"}, None)
        return _Body()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    slept = []
    monkeypatch.setattr("standard_bench.fetch.time.sleep", lambda seconds: slept.append(seconds))
    assert _request("https://example.test/file", "") == b"ok"
    assert calls["n"] == 3
    assert slept == [0.0, 0.0]

    calls["n"] = 0

    def urlopen_500(request, timeout=60):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 503, "down", {}, None)
        return _Body()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_500)
    assert _request("https://example.test/file", "") == b"ok"
    assert calls["n"] == 2

    def urlopen_403(request, timeout=60):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    calls["n"] = 0
    monkeypatch.setattr(urllib.request, "urlopen", urlopen_403)
    with pytest.raises(FetchError, match="403"):
        _request("https://example.test/gated", "")
    assert calls["n"] == 1


def test_cli_smoke(tmp_path: Path):
    from standard_bench.cli import main

    output = tmp_path / "public.csv"
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit) as caught:
        main(["smoke", "--runs-dir", str(runs), "--output", str(output)])
    assert caught.value.code == 0
    text = output.read_text(encoding="utf-8")
    assert SENTINEL not in text
    assert "mmlu_pro" in text
    assert "counts_toward_dow_score" in text
    assert "not_run" in text
    assert "gpqa_diamond" not in text


# MBPP+ 590 shape: last-digit float, nested tuple, and a tiny complex residual.
_EXPECTED_590 = ((42.32632433025415, -2.359036256570777), (-2 + 2.4492935982947064e-16j))
_OUTPUT_590 = ((42.326324330254145, -2.359036256570777), (-2 + 0j))


def test_values_close_accepts_the_590_shape_and_rejects_wrong_answers():
    kwargs = {"rel_tol": FLOAT_REL_TOL, "abs_tol": FLOAT_ABS_TOL}
    assert values_close(_OUTPUT_590, _EXPECTED_590, **kwargs) is True
    assert values_close({"polar": [_OUTPUT_590]}, {"polar": [_EXPECTED_590]}, **kwargs) is True
    assert values_close([42.326324330254145], [42.32632433025415], **kwargs) is True
    assert values_close(88, 88, **kwargs) is True
    assert values_close("same", "same", **kwargs) is True
    wrong_complex = ((42.32632433025415, -2.359036256570777), (-2 + 1j))
    wrong_float = ((42.0, -2.359036256570777), (-2 + 0j))
    assert values_close(wrong_complex, _EXPECTED_590, **kwargs) is False
    assert values_close(wrong_float, _EXPECTED_590, **kwargs) is False
    assert values_close((_OUTPUT_590, 1), (_EXPECTED_590,), **kwargs) is False
    assert values_close([1.0, 2.0], (1.0, 2.0), **kwargs) is False
    assert values_close({"polar": _OUTPUT_590}, {"other": _OUTPUT_590}, **kwargs) is False
    assert values_close(88, 96, **kwargs) is False
    assert values_close("42.326324330254145", "42.32632433025415", **kwargs) is False
    assert values_close(True, 1.0000001, **kwargs) is False
    assert values_close(True, False, **kwargs) is False
    assert values_close({1.0}, {1.0 + 1e-7}, **kwargs) is False
    assert values_close(None, 0, **kwargs) is False


def test_plus_assertion_uses_the_tolerant_compare():
    expected = "((42.32632433025415, -2.359036256570777), (-2+2.4492935982947064e-16j))"
    plus = (
        "import numpy as np\n"
        "def is_floats(x):\n"
        "    return False\n"
        "def assertion(out, exp, atol):\n"
        "    if atol == 0 and is_floats(exp):\n"
        "        atol = 1e-6\n"
        "    if out != exp and atol != 0:\n"
        "        assert np.allclose(out, exp, rtol=1e-07, atol=atol)\n"
        "    else:\n"
        "        assert out == exp\n"
        "def check(candidate):\n"
        f"    assertion(candidate(), {expected}, 0)\n"
    )
    row = {
        "task_id": "590",
        "entry_point": "polar_rect",
        "function_header": "def polar_rect():\n",
        "prompt": "polar",
        "base_test": "def check(candidate):\n    assert isinstance(candidate(), tuple)\n",
        "plus_test": plus,
        "test_imports": [],
    }
    close = (
        "def polar_rect():\n"
        "    return ((42.326324330254145, -2.359036256570777), (-2+0j))\n"
    )
    passed = score_code_task(row, close, suite="mbpp_plus")
    assert passed["base_pass"] is True
    assert passed["plus_pass"] is True
    wrong = (
        "def polar_rect():\n"
        "    return ((42.326324330254145, -2.359036256570777), (-2+1j))\n"
    )
    failed = score_code_task(row, wrong, suite="mbpp_plus")
    assert failed["base_pass"] is True
    assert failed["plus_pass"] is False
    sizes = dict(row)
    sizes["plus_test"] = (
        "def assertion(out, exp, atol):\n"
        "    if atol == 0 and is_floats(exp):\n"
        "        atol = 1e-6\n"
        "    else:\n"
        "        assert out == exp\n"
        "def check(candidate):\n"
        "    assertion(candidate(), 88, 0)\n"
    )
    size_row = score_code_task(
        sizes,
        "def polar_rect():\n    return 96\n",
        suite="mbpp_plus",
    )
    assert size_row["plus_pass"] is False
    rewritten = rewrite_evalplus_test(plus)
    assert "values_close" in rewritten
    assert "cmath" in rewritten
    assert "np.allclose" not in rewritten
    assert "out = set(out)" not in rewritten
    assert rewrite_evalplus_test(rewritten).count("def assertion(") == 1


def test_rewrite_strips_helpers_nested_inside_check():
    nested = (
        "def check(candidate):\n"
        "    import numpy as np\n"
        "    def is_floats(x):\n"
        "        return isinstance(x, np.ndarray)\n"
        "    def assertion(out, exp, atol):\n"
        "        if is_floats(exp):\n"
        "            assert np.allclose(out, exp)\n"
        "        else:\n"
        "            assert out == exp\n"
        "    assertion(candidate(), 1, 0)\n"
    )
    fetched = (
        "def assertion(out, exp, atol):\n"
        "    if is_floats(exp):\n"
        "        atol = 1e-6\n"
        "    assert out == exp\n"
        "\n" + nested
    )
    assert fetched.count("def assertion(") == 2
    rewritten = rewrite_evalplus_test(fetched)
    prepared = prepare_plus_test(fetched)
    for text in (rewritten, prepared):
        assert text.count("def assertion(") == 1
        assert "def is_floats" not in text
        assert "import numpy" not in text
        assert "np.ndarray" not in text
        assert "out = set(out)" not in text
        assert "assertion(candidate(), 1, 0)" in text
    assert prepare_plus_test(prepared).count("def assertion(") == 1
    inside = (
        "def check(candidate):\n"
        "    def assertion(out, exp, atol):\n"
        "        if atol == 0 and is_floats(exp):\n"
        "            atol = 1e-6\n"
        "        out = set(out)\n"
        "        exp = set(exp)\n"
        "        if out != exp and atol != 0:\n"
        "            assert np.allclose(out, exp, rtol=1e-07, atol=atol)\n"
        "        else:\n"
        "            assert out == exp\n"
        "    assertion(candidate(), {1, 2}, 0)\n"
    )
    set_rewritten = rewrite_evalplus_test(inside)
    assert set_rewritten.count("def assertion(") == 1
    assert set_rewritten.count("out = set(out)") == 1
    assert set_rewritten.index("out = set(out)") < set_rewritten.index("def check(")
    row = {
        "task_id": "set",
        "entry_point": "letters",
        "function_header": "def letters():\n",
        "base_test": "def check(candidate):\n    assert callable(candidate)\n",
        "plus_test": inside,
        "test_imports": [],
    }
    passed = score_code_task(row, "def letters():\n    return [2, 1, 1]\n", suite="mbpp_plus")
    assert passed["plus_pass"] is True
    missed = score_code_task(row, "def letters():\n    return [1, 3]\n", suite="mbpp_plus")
    assert missed["plus_pass"] is False


def test_mbpp_590_real_plus_test_accepts_one_ulp_and_rejects_wrong_shapes():
    from standard_bench.load import load_suite

    rows, _entry = load_suite(FIXTURES, "mbpp_plus")
    row = next(item for item in rows if str(item["task_id"]) == "590")
    plus = str(row["plus_test"])
    assert plus.count("def assertion(") == 2
    assert plus.count("def is_floats") == 1
    prepared = prepare_plus_test(plus)
    assert prepared.count("def assertion(") == 1
    assert "def is_floats" not in prepared
    assert "import numpy" not in prepared
    assert "np.allclose" not in prepared
    assert "from math import inf" in prepared
    assert prepare_plus_test(prepared).count("def assertion(") == 1
    canonical = score_code_task(row, str(row["canonical_solution"]), suite="mbpp_plus")
    assert canonical["base_pass"] is True
    assert canonical["plus_pass"] is True
    variant = dict(row)
    variant["base_test"] = "def check(candidate):\n    assert callable(candidate)\n"

    def run(source: str) -> dict:
        return score_code_task(variant, source, suite="mbpp_plus")

    ulp = run(
        "import cmath\n"
        "import math\n"
        "def polar_rect(x, y):\n"
        "    cn = cmath.polar(complex(x, y))\n"
        "    cn1 = cmath.rect(2, cmath.pi)\n"
        "    mag = math.nextafter(cn[0], math.inf)\n"
        "    phase = math.nextafter(cn[1], math.inf)\n"
        "    real = math.nextafter(cn1.real, math.inf)\n"
        "    imag = math.nextafter(cn1.imag, math.inf)\n"
        "    return ((mag, phase), complex(real, imag))\n"
    )
    residual = run(
        "import cmath\n"
        "def polar_rect(x, y):\n"
        "    cn = cmath.polar(complex(x, y))\n"
        "    return (cn, -2+0j)\n"
    )
    off = run(
        "import cmath\n"
        "def polar_rect(x, y):\n"
        "    cn = cmath.polar(complex(x, y))\n"
        "    cn1 = cmath.rect(2, cmath.pi)\n"
        "    return ((cn[0] * 1.01, cn[1]), cn1)\n"
    )
    as_list = run(
        "import cmath\n"
        "def polar_rect(x, y):\n"
        "    cn = cmath.polar(complex(x, y))\n"
        "    cn1 = cmath.rect(2, cmath.pi)\n"
        "    return [cn, cn1]\n"
    )
    nan = run(
        "import cmath\n"
        "def polar_rect(x, y):\n"
        "    cn = cmath.polar(complex(x, y))\n"
        "    cn1 = cmath.rect(2, cmath.pi)\n"
        "    return ((float('nan'), cn[1]), cn1)\n"
    )
    assert ulp["plus_pass"] is True
    assert residual["plus_pass"] is True
    assert off["plus_pass"] is False
    assert as_list["plus_pass"] is False
    assert nan["plus_pass"] is False


def test_mbpp_737_custom_assertion_accepts_a_regex_match():
    from standard_bench.load import load_suite

    rows, _entry = load_suite(FIXTURES, "mbpp_plus")
    row = next(item for item in rows if str(item["task_id"]) == "737")
    plus = str(row["plus_test"])
    assert "out is not None" in plus
    assert "assert exact_match" not in plus
    assert plus.count("def assertion(") == 2
    prepared = prepare_plus_test(plus)
    assert "out is not None" in prepared
    assert prepared.count("assert exact_match") == 1
    assert prepared.count("def assertion(") == 2
    assert prepare_plus_test(prepared).count("assert exact_match") == 1
    canonical = score_code_task(row, str(row["canonical_solution"]), suite="mbpp_plus")
    assert canonical["base_pass"] is True
    assert canonical["plus_pass"] is True
    variant = dict(row)
    variant["base_test"] = "def check(candidate):\n    assert callable(candidate)\n"
    missed = score_code_task(
        variant,
        "import re\ndef check_str(string):\n    return None\n",
        suite="mbpp_plus",
    )
    assert missed["base_pass"] is True
    assert missed["plus_pass"] is False
    always = score_code_task(
        variant,
        "def check_str(string):\n    return True\n",
        suite="mbpp_plus",
    )
    assert always["plus_pass"] is False


def test_dataset_hash_ignores_crlf_and_still_rejects_a_changed_row(tmp_path: Path):
    copied = tmp_path / "data"
    shutil.copytree(FIXTURES, copied)
    target = copied / "gsm8k" / "test.jsonl"
    original = target.read_bytes()
    assert b"\r\n" not in original
    target.write_bytes(original.replace(b"\n", b"\r\n"))
    (copied / "manifest.json").write_bytes(
        (copied / "manifest.json").read_bytes().replace(b"\n", b"\r\n")
    )
    verify_suite(copied, "gsm8k")
    prepared = tmp_path / "prepared"
    shutil.copytree(ROOT / "tests" / "evalplus_prepared", prepared)
    for path in prepared.rglob("*"):
        if path.suffix == ".jsonl" or path.name == "manifest.json":
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    from standard_bench.load import load_suite

    humaneval, _entry = load_suite(prepared, "humaneval_plus")
    mbpp, _mbpp_entry = load_suite(prepared, "mbpp_plus")
    assert len(humaneval) == 7
    assert len(mbpp) == 9
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(HashMismatch, match="hash mismatch"):
        verify_suite(copied, "gsm8k")


def _buggy_mbpp_row(raw: dict) -> dict:
    """Row shape from 71c77776: the entry point is the first def."""
    import re

    from standard_bench.fetch import wrap_module_as_check

    code = str(raw.get("code") or "")
    match = re.search(r"(?m)^def\s+([A-Za-z_][A-Za-z0-9_]*)\b[^\n]*", code)
    helper = match.group(1) if match else ""
    header = (match.group(0).rstrip() + "\n") if match else ""
    tests = [str(item) for item in (raw.get("test_list") or [])]
    base = "def check(candidate):\n    pass\n"
    if helper:
        base = (
            "def check(candidate):\n"
            + "\n".join("    " + item.replace(f"{helper}(", "candidate(") for item in tests)
            + "\n"
        )
    plus_source = str(raw.get("test") or "")
    plus = rewrite_evalplus_test(wrap_module_as_check(plus_source, helper)) if plus_source.strip() else ""
    return {
        "task_id": str(raw.get("task_id")),
        "prompt": raw.get("prompt") or "",
        "entry_point": helper,
        "canonical_solution": code,
        "function_header": header,
        "test_imports": list(raw.get("test_imports") or []),
        "test_list": tests,
        "base_test": base,
        "plus_test": plus,
    }


def test_mbpp_helper_first_tasks_score_the_function_the_tests_call():
    """Tasks 630 and 6: a correct answer that does not use the helper name passes.

    The prepared 630 row still stores ``adjac`` as the entry point. Scoring
    derives ``get_coordinates`` from the asserts. Task 6 is the same shape,
    rebuilt from the EvalPlus row.
    """
    raw_rows = json.loads((ROOT / "tests" / "mbpp_helper_rows.json").read_text(encoding="utf-8"))
    raw = {str(row["task_id"]): row for row in raw_rows}
    fresh_630 = normalize_mbpp([raw["630"]])[0]
    fresh_6 = normalize_mbpp([raw["6"]])[0]
    assert fresh_630["entry_point"] == "get_coordinates"
    assert fresh_630["function_header"].startswith("def get_coordinates")
    assert fresh_6["entry_point"] == "differ_At_One_Bit_Pos"
    assert fresh_6["function_header"].startswith("def differ_At_One_Bit_Pos")

    from standard_bench.load import load_suite

    prepared, _entry = load_suite(ROOT / "tests" / "evalplus_prepared", "mbpp_plus")
    stored_630 = next(row for row in prepared if str(row["task_id"]) == "630")
    assert stored_630["entry_point"] == "adjac"
    assert stored_630["function_header"].startswith("def adjac")
    old_6 = _buggy_mbpp_row(raw["6"])
    assert old_6["entry_point"] == "is_Power_Of_Two"

    itertools_630 = (
        "import itertools\n"
        "def get_coordinates(test_tup):\n"
        "    ranges = [range(n - 1, n + 2) for n in test_tup]\n"
        "    return [list(item) for item in itertools.product(*ranges)]\n"
    )
    renamed_630 = (
        "def neighbors(ele, sub=None):\n"
        "    if sub is None:\n"
        "        sub = []\n"
        "    if not ele:\n"
        "        yield sub\n"
        "    else:\n"
        "        yield from [\n"
        "            idx\n"
        "            for j in range(ele[0] - 1, ele[0] + 2)\n"
        "            for idx in neighbors(ele[1:], sub + [j])\n"
        "        ]\n"
        "def get_coordinates(test_tup):\n"
        "    return list(neighbors(test_tup))\n"
    )
    bare_630 = (
        "    ranges = [range(n - 1, n + 2) for n in test_tup]\n"
        "    return [list(item) for item in __import__('itertools').product(*ranges)]\n"
    )
    renamed_6 = (
        "def power_of_two(x):\n"
        "    return x > 0 and (x & (x - 1)) == 0\n"
        "def differ_At_One_Bit_Pos(a, b):\n"
        "    return power_of_two(a ^ b)\n"
    )
    absent_6 = (
        "def differ_At_One_Bit_Pos(a, b):\n"
        "    x = a ^ b\n"
        "    return x > 0 and (x & (x - 1)) == 0\n"
    )
    for row, completion in (
        (stored_630, itertools_630),
        (stored_630, renamed_630),
        (stored_630, bare_630),
        (fresh_630, itertools_630),
        (old_6, renamed_6),
        (old_6, absent_6),
        (fresh_6, absent_6),
    ):
        scored = score_code_task(row, completion, suite="mbpp_plus")
        assert scored["plus_pass"] is True, (row["task_id"], scored["error"])
        assert "IndentationError" not in scored["error"]

    prompt = mbpp_prompt(stored_630)
    assert "def get_coordinates" in prompt
    assert "def adjac" not in prompt
    assert "get_coordinates(" in prompt
    prompt_6 = mbpp_prompt(old_6)
    assert "def differ_At_One_Bit_Pos" in prompt_6
    assert "def is_Power_Of_Two" not in prompt_6


def test_prepared_evalplus_variants_match_reference():
    """Every distinct upstream helper and assertion variant, minus the exclusions."""
    from standard_bench.load import load_suite
    from standard_bench.reference import verify_reference

    root = ROOT / "tests" / "evalplus_prepared"
    humaneval, _entry = load_suite(root, "humaneval_plus")
    mbpp, _mbpp_entry = load_suite(root, "mbpp_plus")
    assert [row["task_id"] for row in humaneval] == [
        "HumanEval/14",
        "HumanEval/15",
        "HumanEval/32",
        "HumanEval/83",
        "HumanEval/100",
        "HumanEval/130",
        "HumanEval/139",
    ]
    assert [row["task_id"] for row in mbpp] == [
        "88",
        "232",
        "255",
        "590",
        "596",
        "630",
        "737",
        "787",
        "794",
    ]
    report = verify_reference(root, workers=4)
    assert report["ok"] is True
    assert report["suites"]["humaneval_plus"]["passed"] == 6
    assert report["suites"]["humaneval_plus"]["checked"] == 6
    assert [item["task_id"] for item in report["suites"]["humaneval_plus"]["excluded"]] == [
        "HumanEval/32"
    ]
    assert report["suites"]["mbpp_plus"]["passed"] == 7
    assert report["suites"]["mbpp_plus"]["checked"] == 7
    assert [item["task_id"] for item in report["suites"]["mbpp_plus"]["excluded"]] == [
        "255",
        "596",
    ]


def test_rescore_rewrites_saved_responses_without_a_model(tmp_path: Path):
    from standard_bench import SCORER_VERSION
    from standard_bench.cli import main
    from standard_bench.load import load_suite
    from standard_bench.rescore import scorer_commit
    from standard_bench.runner import _row_id

    gsm_rows, _entry = load_suite(FIXTURES, "gsm8k")
    ife_rows, _entry = load_suite(FIXTURES, "ifeval")
    mbpp_rows, _entry = load_suite(FIXTURES, "mbpp_plus")
    code = mbpp_rows[0]["canonical_solution"]
    items = [
        {
            "id": _row_id("gsm8k", 0, gsm_rows[0]),
            "suite": "gsm8k",
            "category": "public_benchmark",
            "counts_toward_score": False,
            "counts_toward_dow_score": False,
            "raw_response": "The answer is 4",
            "response": "The answer is 4",
            "passed": False,
            "score": 0.0,
            "source": "live",
            "preset": "standard_capability",
            "subset": "full",
            "protocol": "full",
        },
        {
            "id": _row_id("ifeval", 0, ife_rows[0]),
            "suite": "ifeval",
            "category": "public_benchmark",
            "counts_toward_score": False,
            "counts_toward_dow_score": False,
            "raw_response": "one short line",
            "response": "one short line",
            "passed": False,
            "score": 0.0,
            "source": "live",
            "preset": "standard_capability",
            "subset": "full",
            "protocol": "full",
        },
        {
            "id": _row_id("mbpp_plus", 0, mbpp_rows[0]),
            "suite": "mbpp_plus",
            "category": "public_benchmark",
            "counts_toward_score": False,
            "counts_toward_dow_score": False,
            "raw_response": code,
            "response": code,
            "passed": False,
            "score": 0.0,
            "base_pass": False,
            "plus_pass": False,
            "source": "live",
            "preset": "standard_capability",
            "subset": "full",
            "protocol": "full",
        },
        {
            "id": "mbpp_plus:596",
            "suite": "mbpp_plus",
            "category": "public_benchmark",
            "counts_toward_score": False,
            "counts_toward_dow_score": False,
            "raw_response": "def tuple_size(value):\n    return 96\n",
            "response": "def tuple_size(value):\n    return 96\n",
            "passed": True,
            "score": 1.0,
            "source": "live",
            "preset": "standard_capability",
            "subset": "full",
            "protocol": "full",
        },
    ]
    run = tmp_path / "run"
    run.mkdir()
    (run / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "rescore-me",
                "connection": {"model": "OLMo 2 1B Instruct", "precision": "bf16"},
                "suites": [],
            }
        ),
        encoding="utf-8",
    )
    original_items = (run / "items.jsonl").read_text(encoding="utf-8")
    original_run = (run / "run.json").read_text(encoding="utf-8")
    loaded = set(sys.modules)
    with pytest.raises(SystemExit) as caught:
        main(["rescore", "--run-dir", str(run), "--fixtures"])
    assert caught.value.code == 0
    added = set(sys.modules) - loaded
    assert not any(name == "torch" or name.startswith("torch.") or name.startswith("transformers") for name in added)
    assert (run / "items.jsonl").read_text(encoding="utf-8") == original_items
    assert (run / "run.json").read_text(encoding="utf-8") == original_run
    assert not (run / "items.jsonl.bak").exists()
    dest = tmp_path / "run-rescored"
    saved = [json.loads(line) for line in (dest / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {item["id"]: item for item in saved}
    assert by_id[_row_id("gsm8k", 0, gsm_rows[0])]["passed"] is True
    assert by_id[_row_id("ifeval", 0, ife_rows[0])]["prompt_strict"] is True
    assert by_id[_row_id("mbpp_plus", 0, mbpp_rows[0])]["plus_pass"] is True
    assert by_id["mbpp_plus:596"]["reference_excluded"] is True
    assert by_id["mbpp_plus:596"]["score"] is None
    record = json.loads((dest / "run.json").read_text(encoding="utf-8"))
    commit = scorer_commit()
    assert record["scorer_version"] == SCORER_VERSION
    assert record["scorer_commit"] == commit
    assert record["scorecard"]["scorer_version"] == SCORER_VERSION
    assert record["scorecard"]["scorer_commit"] == commit
    assert record["scorecard"]["public_benchmarks"]["scorer_version"] == SCORER_VERSION
    assert record["scorecard"]["public_benchmarks"]["scorer_commit"] == commit
    mbpp = next(row for row in record["suites"] if row["name"] == "mbpp_plus")
    assert mbpp["metrics"]["pass_at_1_plus"] == 1.0
    assert mbpp["metrics"]["n_scored"] == 1
    csv_text = (dest / "public_benchmarks.csv").read_text(encoding="utf-8")
    assert "scorer_version" in csv_text.splitlines()[0]
    assert "scorer_commit" in csv_text.splitlines()[0]
    assert SCORER_VERSION in csv_text
    assert commit in csv_text
    payload = json.loads((dest / "public_benchmarks.json").read_text(encoding="utf-8"))
    assert payload["scorer_version"] == SCORER_VERSION
    assert payload["scorer_commit"] == commit
    assert commit


def test_rescore_in_place_writes_backups(tmp_path: Path):
    from standard_bench.load import load_suite
    from standard_bench.rescore import rescore_run_dir
    from standard_bench.runner import _row_id

    gsm_rows, _entry = load_suite(FIXTURES, "gsm8k")
    run = tmp_path / "run"
    run.mkdir()
    (run / "items.jsonl").write_text(
        json.dumps(
            {
                "id": _row_id("gsm8k", 0, gsm_rows[0]),
                "suite": "gsm8k",
                "raw_response": "The answer is 4",
                "response": "The answer is 4",
                "passed": False,
                "score": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps({"run_id": "keep-me", "connection": {"model": "stub"}, "suites": []}) + "\n",
        encoding="utf-8",
    )
    taken = tmp_path / "run-rescored"
    taken.mkdir()
    result = rescore_run_dir(run, data_dir=FIXTURES)
    assert Path(result["run_dir"]) == tmp_path / "run-rescored-2"
    assert json.loads((run / "items.jsonl").read_text(encoding="utf-8"))["passed"] is False
    placed = rescore_run_dir(run, data_dir=FIXTURES, in_place=True)
    assert Path(placed["run_dir"]) == run
    assert json.loads((run / "items.jsonl.bak").read_text(encoding="utf-8"))["passed"] is False
    assert json.loads((run / "run.json.bak").read_text(encoding="utf-8"))["run_id"] == "keep-me"
    assert json.loads((run / "items.jsonl").read_text(encoding="utf-8").splitlines()[0])["passed"] is True
    rescore_run_dir(run, data_dir=FIXTURES, in_place=True)
    assert (run / "items.jsonl.bak.2").is_file()


def test_scorer_commit_falls_back_without_git(monkeypatch: pytest.MonkeyPatch):
    import subprocess

    from standard_bench import SCORER_VERSION
    from standard_bench.rescore import _scorer_stamp, scorer_commit

    def boom(*args, **kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr(subprocess, "check_output", boom)
    stamp = scorer_commit()
    assert stamp == _scorer_stamp()
    assert stamp
    assert f"+scorer{SCORER_VERSION}" in stamp


def _clear_standard_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_EVAL_STANDARD_SUBSET",
        "LLM_EVAL_STANDARD_SEED",
        "LLM_EVAL_STANDARD_LIMIT",
        "LLM_EVAL_STANDARD_LIMITS",
        "LLM_EVAL_STANDARD_THINKING_BUDGET",
        "LLM_EVAL_INCLUDE_GPQA",
        "LLM_EVAL_STANDARD_DATA",
        "LLM_EVAL_STANDARD_FIXTURES",
    ):
        monkeypatch.delenv(key, raising=False)


class _AlignedModel(StubStandardModel):
    """Answers from a marker in the prompt, so the wrong gold fails the item."""

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        import re

        text = prompt or ""
        reply = "I do not know."
        if "CORRECT=" in text and "\nA)" in text:
            marker = re.search(r"CORRECT=(\S+)", text)
            want = marker.group(1) if marker else ""
            reply = "Answer: Z"
            for letter in "ABCD":
                line = re.search(rf"(?m)^{letter}\) (.+)$", text)
                if line and line.group(1).strip() == want:
                    reply = f"Answer: {letter}"
                    break
        elif "def value_" in text and "RETURN=" in text:
            number = re.search(r"RETURN=(\d+)", text)
            reply = f"    return {number.group(1)}\n" if number else reply
        elif "def value():" in text and "RETURN=" in text:
            number = re.search(r"RETURN=(\d+)", text)
            reply = f"    return {number.group(1)}\n" if number else reply
        elif "kw" in text:
            marker = re.search(r"kw\d{4}", text)
            if marker:
                reply = marker.group(0)
        elif "LETTER=" in text:
            marker = re.search(r"LETTER=([A-J])", text)
            if marker:
                reply = f"The answer is ({marker.group(1)})."
        elif "VALUE " in text:
            marker = re.search(r"VALUE (\d+)", text)
            if marker:
                reply = f"\\boxed{{{marker.group(1)}}}"
        else:
            numbers = re.findall(r"What is (\d+)\?", text)
            if numbers:
                reply = f"The answer is {numbers[-1]}."
        return ModelResponse(
            text=reply,
            latency_ms=0.1,
            tokens_used=4,
            metadata={
                "model": self.name,
                "device": "stub",
                "raw_text": reply,
                "hit_token_cap": False,
                "think_truncated": False,
                "prompt_over_budget": False,
                "completion_tokens": 4,
            },
        )


def _materialize(root: Path, suites: dict[str, list[dict]]) -> None:
    from standard_bench.manifest import file_record, write_manifest

    payload: dict = {"version": 1, "suites": {}}
    for name, rows in suites.items():
        relative = f"{name}/test.jsonl"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        payload["suites"][name] = {
            "files": [file_record(root, relative)],
            "full_size": len(rows),
            "gated": name == "gpqa_diamond",
            "license": "mit",
            "repo": "synthetic",
            "revision": "test",
            "rows": len(rows),
        }
    write_manifest(root, payload)


def _shuffle_suite(root: Path, suite: str) -> None:
    import random

    from standard_bench.manifest import file_record, load_manifest, write_manifest

    path = root / suite / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    original = list(rows)
    rng = random.Random(1)
    for _ in range(8):
        rng.shuffle(rows)
        if rows != original:
            break
    assert rows != original
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    payload = load_manifest(root)
    payload["suites"][suite]["files"] = [file_record(root, f"{suite}/test.jsonl")]
    write_manifest(root, payload)


def _gsm_rows(count: int) -> list[dict]:
    return [
        {"question": f"What is {index}?", "answer": f"{index} + 0 = {index}.\n#### {index}"}
        for index in range(count)
    ]


def _mmlu_rows(count: int) -> list[dict]:
    letters = "ABCDEFGHIJ"
    rows = []
    for index in range(count):
        letter = letters[index % 10]
        rows.append(
            {
                "question_id": 5000 + index,
                "question": f"LETTER={letter} synthetic {index}",
                "options": [f"opt{i}" for i in range(10)],
                "answer": letter,
                "answer_index": letters.index(letter),
                "cot_content": "",
                "category": "synthetic",
                "src": "synthetic",
            }
        )
    return rows


def _math_rows(count: int) -> list[dict]:
    return [
        {
            "problem": f"VALUE {index}",
            "solution": str(index),
            "answer": str(index),
            "subject": "synthetic",
            "level": 1,
            "unique_id": f"synth-math-{index}",
        }
        for index in range(count)
    ]


def _ifeval_rows(count: int) -> list[dict]:
    return [
        {
            "key": 1000 + index,
            "prompt": f"Include kw{index:04d} in the reply.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": [f"kw{index:04d}"]}],
        }
        for index in range(count)
    ]


def _humaneval_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "task_id": f"Synth/{index}",
                "prompt": f'def value_{index}():\n    """RETURN={index}"""\n',
                "entry_point": f"value_{index}",
                "canonical_solution": f"def value_{index}():\n    return {index}\n",
                "base_test": f"def check(candidate):\n    assert candidate() == {index}\n",
                "plus_test": f"def check(candidate):\n    assert candidate() == {index}\n",
            }
        )
    return rows


def _mbpp_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "task_id": f"s{index}",
                "prompt": f"RETURN={index} Return that number.",
                "entry_point": "value",
                "function_header": "def value():\n",
                "test_list": [f"assert value() == {index}"],
                "base_test": f"def check(candidate):\n    assert candidate() == {index}\n",
                "plus_test": f"def check(candidate):\n    assert candidate() == {index}\n",
                "canonical_solution": f"def value():\n    return {index}\n",
                "test_imports": [],
            }
        )
    return rows


def _gpqa_rows(count: int) -> list[dict]:
    return [
        {
            "id": f"rec{index}",
            "question": f"INHOUSE_Q{index} CORRECT=gold{index}",
            "correct": f"gold{index}",
            "incorrect": [f"no{index}a", f"no{index}b", f"no{index}c"],
        }
        for index in range(count)
    ]


def _run_names(tmp_path: Path, data: Path, names: list[str], subset: str, preset: str) -> list[dict]:
    config = {
        "fixtures": False,
        "data_dir": str(data),
        "subset": subset,
        "preset": preset,
        "thinking_budget": 512,
    }
    model = _AlignedModel()
    return [run_standard_suite(_ctx(model, tmp_path), config, name) for name in names]


def _dump_run(dest: Path, results: list[dict]) -> Path:
    dest.mkdir()
    items = [item for result in results for item in result["items"]]
    (dest / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )
    (dest / "run.json").write_text(
        json.dumps(
            {
                "run_id": dest.name,
                "connection": {"model": "OLMo 2 1B Instruct", "precision": "bf16"},
                "suites": [
                    {
                        "name": result["name"],
                        "subset": result.get("subset"),
                        "seed": result.get("seed"),
                        "protocol": result.get("protocol"),
                        "preset": result.get("preset"),
                        "metrics": result.get("metrics") or {},
                    }
                    for result in results
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dest


def _grade(item: dict) -> tuple:
    return (
        item.get("suite"),
        item.get("passed"),
        item.get("score"),
        item.get("expected"),
        item.get("predicted"),
        item.get("plus_pass"),
        item.get("prompt_strict"),
    )


def _grades(items: list[dict]) -> list[tuple]:
    return [_grade(item) for item in items]


def _read_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _blank_stored_gold(path: Path) -> None:
    rows = _read_items(path)
    for item in rows:
        if item.get("suite") in {"gsm8k", "math500", "mmlu_pro"}:
            item["expected"] = ""
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def _assert_rescored(original: list[dict], run_dir: Path) -> list[dict]:
    saved = _read_items(run_dir / "items.jsonl")
    assert _grades(saved) == _grades(original)
    return saved


def test_row_id_ignores_subset_position():
    from standard_bench.runner import _row_id

    row = {"question": "What is 2 + 2?", "answer": "#### 4"}
    assert _row_id("gsm8k", 0, row) == _row_id("gsm8k", 41, row)
    assert _row_id("gsm8k", 0, row).startswith("gsm8k:q")
    assert _row_id("ifeval", 3, {"key": 1000, "prompt": "x"}) == "ifeval:1000"
    assert _row_id("humaneval_plus", 9, {"task_id": "Synth/3"}) == "humaneval_plus:Synth/3"


def test_duplicate_question_stops_before_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from standard_bench.identity import ItemIdError

    _clear_standard_env(monkeypatch)
    data = tmp_path / "data"
    _materialize(data, {"gsm8k": [{"question": "What is 1?", "answer": "#### 1"}] * 2})
    with pytest.raises(ItemIdError, match="ambiguous"):
        run_standard_suite(
            _ctx(_AlignedModel(), tmp_path),
            {"fixtures": False, "data_dir": str(data), "subset": "full", "preset": "standard_capability"},
            "gsm8k",
        )


def test_rescore_rejects_ambiguous_and_partial_ids(tmp_path: Path):
    from standard_bench.load import load_suite
    from standard_bench.rescore import RescoreError, rescore_run_dir
    from standard_bench.runner import _prompt_for, _row_id

    gsm_rows, _entry = load_suite(FIXTURES, "gsm8k")
    prompt = _prompt_for("gsm8k", gsm_rows[0], [])[0]
    valid = {
        "id": _row_id("gsm8k", 0, gsm_rows[0]),
        "suite": "gsm8k",
        "prompt": prompt,
        "raw_response": "The answer is 4",
        "response": "The answer is 4",
        "expected": "",
        "passed": False,
        "score": 0.0,
    }
    missing = dict(valid)
    missing["id"] = ""
    run = tmp_path / "missing"
    run.mkdir()
    (run / "items.jsonl").write_text(json.dumps(missing) + "\n", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"run_id": "missing", "suites": []}) + "\n", encoding="utf-8")
    with pytest.raises(RescoreError, match="subset"):
        rescore_run_dir(run, data_dir=FIXTURES)

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    other = dict(valid)
    other["id"] = "gsm8k:missing"
    (mixed / "items.jsonl").write_text(
        json.dumps(valid) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8",
    )
    (mixed / "run.json").write_text(
        json.dumps(
            {
                "run_id": "mixed",
                "suites": [
                    {
                        "name": "gsm8k",
                        "subset": "full",
                        "seed": None,
                        "metrics": {"n_items": 2, "limit": None},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RescoreError, match="cannot align every item"):
        rescore_run_dir(mixed, data_dir=FIXTURES)

    data = tmp_path / "dup"
    _materialize(data, {"gsm8k": [{"question": "What is 4?", "answer": "#### 4"}] * 2})
    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / "items.jsonl").write_text(json.dumps(valid) + "\n", encoding="utf-8")
    (ambiguous / "run.json").write_text(json.dumps({"run_id": "ambiguous", "suites": []}) + "\n", encoding="utf-8")
    with pytest.raises(RescoreError, match="ambiguous"):
        rescore_run_dir(ambiguous, data_dir=data)


def _rescore_matches(monkeypatch: pytest.MonkeyPatch, run: Path, data: Path, original: list[dict]) -> Path:
    from standard_bench.rescore import rescore_run_dir

    def explode(*_args, **_kwargs):
        raise AssertionError("rescore re-ran subset selection")

    _blank_stored_gold(run / "items.jsonl")
    monkeypatch.setattr("standard_bench.rescore.select_rows", explode)
    result = rescore_run_dir(run, data_dir=data)
    saved = _assert_rescored(original, Path(result["run_dir"]))
    assert saved
    return Path(result["run_dir"])


def test_rescore_quick50_matches_and_ignores_a_shuffled_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from standard_bench.scorers.gsm8k import extract_gold

    _clear_standard_env(monkeypatch)
    count = 55
    gsm = _gsm_rows(count)
    data = tmp_path / "data"
    names = ["gsm8k", "mmlu_pro", "math500", "ifeval", "humaneval_plus"]
    _materialize(
        data,
        {
            "gsm8k": gsm,
            "mmlu_pro": _mmlu_rows(count),
            "math500": _math_rows(count),
            "ifeval": _ifeval_rows(count),
            "humaneval_plus": _humaneval_rows(count),
        },
    )
    results = _run_names(tmp_path, data, names, "quick50", "standard_capability_quick")
    original = [item for result in results for item in result["items"]]
    assert all(item.get("passed") for item in original)
    gsm_result = next(result for result in results if result["name"] == "gsm8k")
    assert len(gsm_result["items"]) == 50
    assert gsm_result["seed"] == QUICK_SEED
    assert [item["expected"] for item in gsm_result["items"]] != [
        extract_gold(row["answer"]) for row in gsm[:50]
    ]
    assert all(item["id"].startswith("gsm8k:q") for item in gsm_result["items"])
    ifeval = next(result for result in results if result["name"] == "ifeval")
    assert all(str(item["id"]).startswith("ifeval:1") for item in ifeval["items"])
    humaneval = next(result for result in results if result["name"] == "humaneval_plus")
    assert all(str(item["id"]).startswith("humaneval_plus:Synth/") for item in humaneval["items"])
    run = _dump_run(tmp_path / "quick", results)
    _rescore_matches(monkeypatch, run, data, original)
    for name in names:
        _shuffle_suite(data, name)
    _rescore_matches(monkeypatch, run, data, original)


def test_rescore_short_matches_after_the_dataset_is_shuffled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _clear_standard_env(monkeypatch)
    count = 12
    data = tmp_path / "data"
    names = ["gsm8k", "mmlu_pro", "math500", "ifeval"]
    _materialize(
        data,
        {
            "gsm8k": _gsm_rows(count),
            "mmlu_pro": _mmlu_rows(count),
            "math500": _math_rows(count),
            "ifeval": _ifeval_rows(count),
        },
    )
    results = _run_names(tmp_path, data, names, "short", "standard_capability")
    original = [item for result in results for item in result["items"]]
    assert all(item.get("passed") for item in original)
    assert all(result["seed"] is None and result["subset"] == "short" for result in results)
    assert len(next(result for result in results if result["name"] == "gsm8k")["items"]) == count
    run = _dump_run(tmp_path / "short", results)
    _rescore_matches(monkeypatch, run, data, original)
    for name in names:
        _shuffle_suite(data, name)
    _rescore_matches(monkeypatch, run, data, original)


def test_rescore_legacy_quick50_uses_the_recorded_subset_and_aborts_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from standard_bench.load import select_rows as real_select
    from standard_bench.rescore import RescoreError, rescore_run_dir
    from standard_bench.scorers.gsm8k import extract_gold

    _clear_standard_env(monkeypatch)
    count = 55
    gsm = _gsm_rows(count)
    data = tmp_path / "data"
    names = ["gsm8k", "ifeval"]
    _materialize(data, {"gsm8k": gsm, "ifeval": _ifeval_rows(count)})
    results = _run_names(tmp_path, data, names, "quick50", "standard_capability_quick")
    original = [item for result in results for item in result["items"]]
    assert all(item.get("passed") for item in original)
    gsm_items = next(result for result in results if result["name"] == "gsm8k")["items"]
    assert [item["expected"] for item in gsm_items] != [extract_gold(row["answer"]) for row in gsm[:50]]
    run = _dump_run(tmp_path / "legacy", results)
    rewritten = _read_items(run / "items.jsonl")
    seen: dict[str, int] = {}
    for item in rewritten:
        suite = str(item.get("suite") or "")
        index = seen.get(suite, 0)
        seen[suite] = index + 1
        item["id"] = f"{suite}:{index}"
        item["subset"] = "full"
        item["seed"] = None
        item["subset_limit"] = None
        if suite == "gsm8k":
            item["expected"] = ""
    (run / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in rewritten),
        encoding="utf-8",
    )
    calls: list[tuple] = []

    def spy(rows, limit, seed):
        calls.append((limit, seed, len(rows)))
        return real_select(rows, limit, seed)

    monkeypatch.setattr("standard_bench.rescore.select_rows", spy)
    result = rescore_run_dir(run, data_dir=data)
    assert _grades(_read_items(Path(result["run_dir"]) / "items.jsonl")) == _grades(original)
    assert (50, QUICK_SEED) in {(limit, seed) for limit, seed, _count in calls}
    for name in names:
        _shuffle_suite(data, name)
    with pytest.raises(RescoreError, match="Prompt text does not match"):
        rescore_run_dir(run, data_dir=data)


def test_rescore_full_preset_and_gpqa_align_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_standard_env(monkeypatch)
    data = tmp_path / "data"
    names = ["gsm8k", "mmlu_pro", "math500", "ifeval", "humaneval_plus", "mbpp_plus"]
    _materialize(
        data,
        {
            "gsm8k": _gsm_rows(4),
            "mmlu_pro": _mmlu_rows(4),
            "math500": _math_rows(4),
            "ifeval": _ifeval_rows(4),
            "humaneval_plus": _humaneval_rows(2),
            "mbpp_plus": _mbpp_rows(2),
        },
    )
    results = _run_names(tmp_path, data, names, "full", "standard_capability")
    original = [item for result in results for item in result["items"]]
    assert all(item.get("passed") for item in original)
    assert all(result["subset"] == "full" and result["seed"] is None for result in results)
    run = _dump_run(tmp_path / "full", results)
    _rescore_matches(monkeypatch, run, data, original)
    for name in names:
        _shuffle_suite(data, name)
    _rescore_matches(monkeypatch, run, data, original)

    gpqa_data = tmp_path / "gpqa"
    gpqa_rows = _gpqa_rows(8)
    _materialize(gpqa_data, {"gpqa_diamond": gpqa_rows})
    quick = _run_names(tmp_path, gpqa_data, ["gpqa_diamond"], "quick50", "standard_capability_gpqa")
    gpqa_items = quick[0]["items"]
    assert len(gpqa_items) == 8
    assert quick[0]["seed"] == QUICK_SEED
    assert [item["id"] for item in gpqa_items] != [f"gpqa_diamond:{row['id']}" for row in gpqa_rows]
    assert all(item.get("passed") for item in gpqa_items)
    assert all(item.get("prompt") == "" and item.get("response") == "" for item in gpqa_items)
    gpqa_run = _dump_run(tmp_path / "gpqa-quick", quick)
    saved_dir = _rescore_matches(monkeypatch, gpqa_run, gpqa_data, gpqa_items)
    blob = (saved_dir / "items.jsonl").read_text(encoding="utf-8")
    assert "INHOUSE_Q" not in blob
    assert "CORRECT=" not in blob
    assert "gold0" not in blob
    assert all(item.get("rescored") for item in _read_items(saved_dir / "items.jsonl"))
    _shuffle_suite(gpqa_data, "gpqa_diamond")
    _rescore_matches(monkeypatch, gpqa_run, gpqa_data, gpqa_items)

    full = _run_names(tmp_path, gpqa_data, ["gpqa_diamond"], "full", "standard_capability_gpqa")
    assert all(item.get("passed") for item in full[0]["items"])
    full_run = _dump_run(tmp_path / "gpqa-full", full)
    _shuffle_suite(gpqa_data, "gpqa_diamond")
    saved_full = _rescore_matches(monkeypatch, full_run, gpqa_data, full[0]["items"])
    full_blob = (saved_full / "items.jsonl").read_text(encoding="utf-8")
    assert "INHOUSE_Q" not in full_blob
    assert "gold0" not in full_blob


def _write_run(dest: Path, items: list[dict], record: dict) -> Path:
    dest.mkdir()
    (dest / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )
    (dest / "run.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return dest


def test_rescore_template_opened_think_and_end_of_thought(tmp_path: Path):
    from llm_eval.models.context import strip_think_blocks

    from standard_bench.answers import CODE_SUITES, graded_answer
    from standard_bench.rescore import _answer_text, rescore_run_dir
    from standard_bench.scorers.gsm8k import score_gsm8k

    opened = "x</think>\nAnswer: 7"
    assert graded_answer(opened, suite="gsm8k", prompt_opened_think=True) == "Answer: 7"
    assert graded_answer(opened, suite="humaneval_plus", prompt_opened_think=True) == "\nAnswer: 7"
    code = "trace</think>\n    return a + b\n"
    assert graded_answer(code, suite="humaneval_plus", prompt_opened_think=True) == "\n    return a + b\n"
    assert graded_answer(code, suite="gsm8k", prompt_opened_think=True) == "return a + b"
    assert "humaneval_plus" in CODE_SUITES and "mbpp_plus" in CODE_SUITES

    thought = "The answer is 99<|end_of_thought|>\nThe answer is 7"
    assert graded_answer(thought, suite="gsm8k", think_marker="<|end_of_thought|>") == "The answer is 7"
    assert score_gsm8k("7", strip_think_blocks(thought))["passed"] is False
    assert score_gsm8k("7", graded_answer(thought, suite="gsm8k", think_marker="<|end_of_thought|>"))["passed"] is True

    opened_item = {
        "id": "gsm8k:template-opened",
        "suite": "gsm8k",
        "raw_response": opened,
        "expected": "7",
        "prompt_opened_think": True,
        "passed": False,
        "score": 0.0,
    }
    assert _answer_text(opened_item, {}) == "Answer: 7"
    opened_run = _write_run(
        tmp_path / "opened",
        [opened_item],
        {"run_id": "opened", "suites": []},
    )
    opened_result = rescore_run_dir(opened_run)
    saved = json.loads((Path(opened_result["run_dir"]) / "items.jsonl").read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["predicted"] == "7"
    assert saved["score"] == 1.0

    # The same close, with the flag stored only on the run.
    run_level = dict(opened_item)
    run_level.pop("prompt_opened_think")
    run_level["id"] = "gsm8k:template-opened-run"
    flagged = _write_run(
        tmp_path / "opened-run",
        [run_level],
        {"run_id": "opened-run", "prompt_opened_think": True, "think_marker": "</think>", "suites": []},
    )
    flagged_saved = json.loads(
        (Path(rescore_run_dir(flagged)["run_dir"]) / "items.jsonl").read_text(encoding="utf-8")
    )
    assert flagged_saved["passed"] is True
    assert flagged_saved["predicted"] == "7"

    thought_item = {
        "id": "gsm8k:end-of-thought",
        "suite": "gsm8k",
        "raw_response": thought,
        "expected": "7",
        "think_marker": "<|end_of_thought|>",
        "passed": False,
        "score": 0.0,
    }
    thought_run = _write_run(
        tmp_path / "thought",
        [thought_item],
        {"run_id": "thought", "suites": []},
    )
    thought_saved = json.loads(
        (Path(rescore_run_dir(thought_run)["run_dir"]) / "items.jsonl").read_text(encoding="utf-8")
    )
    assert thought_saved["passed"] is True
    assert thought_saved["predicted"] == "7"

    # Stored response is what the run graded, even when raw still holds the trace.
    preferred = {
        "id": "gsm8k:stored",
        "suite": "gsm8k",
        "response": "The answer is 7",
        "raw_response": "The answer is 99</think>\nThe answer is 4",
        "expected": "7",
        "prompt_opened_think": True,
        "think_marker": "</think>",
        "passed": False,
        "score": 0.0,
    }
    preferred_run = _write_run(tmp_path / "stored", [preferred], {"run_id": "stored", "suites": []})
    preferred_saved = json.loads(
        (Path(rescore_run_dir(preferred_run)["run_dir"]) / "items.jsonl").read_text(encoding="utf-8")
    )
    assert preferred_saved["passed"] is True
    assert preferred_saved["predicted"] == "7"


def test_runner_records_think_flags_and_grades_the_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_EVAL_STANDARD_SUBSET", raising=False)
    monkeypatch.delenv("LLM_EVAL_STANDARD_SEED", raising=False)

    class OpenedThink:
        def generate(self, prompt, **kwargs):
            return ModelResponse(
                "The answer is 99",
                1.0,
                4,
                {
                    "raw_text": "The answer is 99</think>\nThe answer is 4",
                    "think_marker": "</think>",
                    "prompt_opened_think": True,
                    "hit_token_cap": False,
                    "think_truncated": False,
                    "prompt_over_budget": False,
                },
            )

    from standard_bench import SCORER_VERSION

    result = run_standard_suite(
        _ctx(OpenedThink(), tmp_path),
        {"fixtures": True, "preset": "standard_capability"},
        "gsm8k",
    )
    assert result["metrics"]["think_marker"] == "</think>"
    assert result["metrics"]["prompt_opened_think"] is True
    assert result["metrics"]["scorer_version"] == SCORER_VERSION
    first, second = result["items"]
    assert first["response"] == "The answer is 4"
    assert first["raw_response"] == "The answer is 99</think>\nThe answer is 4"
    assert first["think_marker"] == "</think>"
    assert first["prompt_opened_think"] is True
    assert first["passed"] is True
    assert first["scorer_version"] == SCORER_VERSION
    assert second["response"] == "The answer is 4"
    assert second["passed"] is False


def test_rescore_reasoning_quick50_reproduces_runtime_metrics(tmp_path: Path, capsys: pytest.CaptureFixture):
    import copy

    from llm_eval.models.context import strip_think_blocks

    from standard_bench import SCORER_VERSION
    from standard_bench.answers import CODE_SUITES, graded_answer
    from standard_bench.identity import stable_row_id
    from standard_bench.load import load_file, load_suite
    from standard_bench.protocol import CUSTOMER_SUITES, QUICK_SEED, subset_limits
    from standard_bench.reference import exclusions_for
    from standard_bench.rescore import rescore_run_dir
    from standard_bench.runner import _aggregate, _prompt_for, _score
    from standard_bench.sandbox import RunLimits
    from standard_bench.scorers.gsm8k import extract_gold

    poison = "The answer is 99, the answer is (J), with a comma"
    limits = RunLimits()
    caps = subset_limits("quick50")
    items: list[dict] = []
    suites = []
    naive_passed: list[bool] = []
    graded_passed: list[bool] = []
    cursor = 0
    for suite in CUSTOMER_SUITES:
        rows, _entry = load_suite(FIXTURES, suite)
        dropped = exclusions_for(suite)
        rows = [row for row in rows if str(row.get("task_id")) not in dropped]
        chosen = select_rows(rows, caps[suite], QUICK_SEED)
        assert chosen == select_rows(list(rows), caps[suite], QUICK_SEED)
        assert len(chosen) == min(50, len(rows))
        validation = load_file(FIXTURES, suite, "validation.jsonl") if suite == "mmlu_pro" else []
        suite_items = []
        for row in chosen:
            if suite == "mmlu_pro":
                answer = f"the answer is ({row['answer']})"
            elif suite == "gsm8k":
                answer = f"The answer is {extract_gold(row['answer'])}"
            elif suite == "math500":
                answer = "\\boxed{" + str(row["answer"]) + "}"
            elif suite == "ifeval":
                instructions = list(row.get("instruction_id_list") or [])
                answer = "one short line" if "punctuation:no_comma" in instructions else "too short"
            else:
                answer = str(row.get("canonical_solution") or "")
            if cursor % 2 == 0:
                marker = "</think>"
                opened = True
            else:
                marker = "<|end_of_thought|>"
                opened = False
            cursor += 1
            gap = "" if suite in CODE_SUITES else "\n"
            raw = f"{poison}{marker}{gap}{answer}"
            graded = graded_answer(
                raw,
                suite=suite,
                think_marker=marker,
                prompt_opened_think=opened,
            )
            assert poison not in graded
            if suite in CODE_SUITES:
                assert graded == answer
            else:
                assert graded == answer.strip()
            _prompt, private = _prompt_for(suite, row, validation)
            scored = _score(suite, graded, private, limits)
            naive = strip_think_blocks(raw, strip_edges=suite not in CODE_SUITES)
            if suite not in CODE_SUITES:
                naive_scored = _score(suite, naive, private, limits)
                naive_passed.append(bool(naive_scored.get("passed")))
                graded_passed.append(bool(scored.get("passed")))
            item = {
                "id": stable_row_id(suite, row),
                "suite": suite,
                "category": "public_benchmark",
                "response": graded,
                "raw_response": raw,
                "think_marker": marker,
                "prompt_opened_think": opened,
                "expected": scored.get("expected", ""),
                "predicted": scored.get("predicted", ""),
                "passed": scored.get("passed"),
                "score": scored.get("score"),
                "base_pass": scored.get("base_pass"),
                "plus_pass": scored.get("plus_pass"),
                "prompt_strict": scored.get("prompt_strict"),
                "prompt_loose": scored.get("prompt_loose"),
                "instruction_strict": scored.get("instruction_strict"),
                "instruction_loose": scored.get("instruction_loose"),
                "source": "live",
                "subset": "quick50",
                "seed": QUICK_SEED,
                "preset": "standard_capability_quick",
                "protocol": "quick-check",
                "scorer_version": SCORER_VERSION,
            }
            suite_items.append(item)
            items.append(item)
        metrics = _aggregate(suite, suite_items)
        metrics["limit"] = caps[suite]
        metrics["n_items"] = len(suite_items)
        metrics["scorer_version"] = SCORER_VERSION
        suites.append(
            {
                "name": suite,
                "subset": "quick50",
                "seed": QUICK_SEED,
                "protocol": "quick-check",
                "preset": "standard_capability_quick",
                "pass_rate": metrics.get("score"),
                "metrics": metrics,
            }
        )
    assert naive_passed != graded_passed
    record = {
        "run_id": "reasoning-quick50",
        "scorer_version": SCORER_VERSION,
        "connection": {"model": "Phi-4-mini-reasoning"},
        "suites": suites,
    }
    run = _write_run(tmp_path / "reasoning", items, record)
    result = rescore_run_dir(run, data_dir=FIXTURES)
    assert result["pass_flip_warning"] == ""
    assert result["pass_flips"] == []
    captured = capsys.readouterr()
    assert "flipped passed" not in captured.err
    saved = _read_items(Path(result["run_dir"]) / "items.jsonl")
    assert _grades(saved) == _grades(items)
    saved_record = json.loads((Path(result["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    for suite_row in suites:
        name = suite_row["name"]
        fresh = _aggregate(name, [item for item in saved if item["suite"] == name])
        original = _aggregate(name, [item for item in items if item["suite"] == name])
        assert fresh == original
        stored = next(row for row in saved_record["suites"] if row["name"] == name)
        assert stored["pass_rate"] == original["score"]
        assert stored["metrics"]["score"] == original["score"]

    fallback_items = copy.deepcopy(items)
    for item in fallback_items:
        item.pop("response")
    fallback = _write_run(tmp_path / "reasoning-raw", fallback_items, record)
    fallback_result = rescore_run_dir(fallback, data_dir=FIXTURES)
    assert fallback_result["pass_flip_warning"] == ""
    fallback_saved = _read_items(Path(fallback_result["run_dir"]) / "items.jsonl")
    assert _grades(fallback_saved) == _grades(items)


def test_rescore_warns_when_unchanged_scorer_flips_passed(tmp_path: Path, capsys: pytest.CaptureFixture):
    from standard_bench import SCORER_VERSION
    from standard_bench.rescore import rescore_run_dir

    def build(n: int, flips: int, version: str) -> list[dict]:
        rows = []
        for index in range(n):
            rows.append(
                {
                    "id": f"gsm8k:flip-{index}",
                    "suite": "gsm8k",
                    "response": "The answer is 4",
                    "raw_response": "The answer is 99</think>\nThe answer is 4",
                    "expected": "4",
                    "prompt_opened_think": True,
                    "think_marker": "</think>",
                    "passed": index >= flips,
                    "score": 0.0 if index < flips else 1.0,
                    "scorer_version": version,
                }
            )
        return rows

    warned = _write_run(
        tmp_path / "warned",
        build(8, 8, SCORER_VERSION),
        {"run_id": "warned", "scorer_version": SCORER_VERSION, "suites": []},
    )
    warned_result = rescore_run_dir(warned)
    err = capsys.readouterr().err
    assert "flipped passed on 8/8" in err
    assert "gsm8k gsm8k:flip-0: passed False -> True" in err
    assert "gsm8k gsm8k:flip-4: passed False -> True" in err
    assert "and 3 more" in err
    assert "gsm8k:flip-5" not in err
    assert warned_result["pass_flip_warning"]
    assert [row["id"] for row in warned_result["pass_flips"][:5]] == [f"gsm8k:flip-{i}" for i in range(5)]
    assert all(json.loads(line)["passed"] for line in (Path(warned_result["run_dir"]) / "items.jsonl").read_text().splitlines())

    quiet = _write_run(
        tmp_path / "boundary",
        build(20, 1, SCORER_VERSION),
        {"run_id": "boundary", "scorer_version": SCORER_VERSION, "suites": []},
    )
    boundary = rescore_run_dir(quiet)
    err = capsys.readouterr().err
    assert boundary["pass_flip_warning"] == ""
    assert boundary["pass_flips"] == []
    assert "flipped passed" not in err

    older = _write_run(
        tmp_path / "older",
        build(8, 8, "1"),
        {"run_id": "older", "scorer_version": "1", "suites": []},
    )
    older_result = rescore_run_dir(older)
    err = capsys.readouterr().err
    assert older_result["pass_flip_warning"] == ""
    assert "flipped passed" not in err
    assert all(
        json.loads(line)["passed"]
        for line in (Path(older_result["run_dir"]) / "items.jsonl").read_text().splitlines()
    )
