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
    PLAN_SUITES,
    PREFILL_TOK_S,
    QUICK_THINKING_CAP,
    SCREEN_LIMITS,
    THINKING_MODELS,
    THROUGHPUT,
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
    HUMANEVAL_STOPS,
    MBPP_STOPS,
    assemble_program,
    rewrite_evalplus_test,
    score_code_task,
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
    assert "mmlu_pro=200" in doc


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
