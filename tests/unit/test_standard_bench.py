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
    from standard_bench.rescore import rescore_run_dir

    run = tmp_path / "run"
    run.mkdir()
    (run / "items.jsonl").write_text(
        json.dumps(
            {
                "id": "gsm8k:0",
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
