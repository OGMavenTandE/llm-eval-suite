"""Default council licenses and the same-family rule. No model server."""

from __future__ import annotations

from llm_eval_suite.connections import (
    COVER_JUDGE_MODEL,
    DEFAULT_JUDGES,
    cover_judge,
    model_family,
    same_family,
)
from llm_eval_suite.council import run_council


def test_default_panel_is_phi_and_granite():
    models = [row["model"] for row in DEFAULT_JUDGES]
    assert models == ["phi4-mini", "granite4:micro"]
    assert DEFAULT_JUDGES[0]["license"] == "MIT"
    assert DEFAULT_JUDGES[1]["license"] == "Apache-2.0"
    assert "qwen" not in " ".join(models)


def test_family_names_cover_ollama_tags_and_repo_ids():
    assert model_family("phi4-mini") == "phi"
    assert model_family("microsoft/Phi-4-mini-instruct") == "phi"
    assert model_family("granite4:micro") == "granite"
    assert model_family("Granite 4.2 3B") == "granite"
    assert model_family("llama3.2:3b") == "llama"
    assert model_family("qwen2.5:3b-instruct") == ""
    assert same_family({"model": "phi4-mini"}, {"model": "Phi-4-mini-instruct"})
    assert not same_family({"model": "phi4-mini"}, {"model": "granite4:micro"})
    assert cover_judge({"model": "phi4-mini"})["model"] == COVER_JUDGE_MODEL
    assert cover_judge({"model": "granite4:micro"})["model"] == COVER_JUDGE_MODEL
    assert cover_judge({"model": "llama3.2:3b"}) is None
    assert cover_judge({"model": "qwen2.5:3b-instruct"}) is None


def test_phi_candidate_is_graded_by_granite_and_llama():
    seen = []

    def generate_for(judge):
        seen.append(judge["model"])

        def generate(prompt: str) -> str:
            if prompt.startswith("TASK: rank"):
                return "RANKING: Review A > Review B"
            return "The failure count is 0."

        return generate

    run = {
        "run_id": "family",
        "status": "completed",
        "validity": "ok",
        "connection": {"model": "phi4-mini"},
        "scorecard": {"categories": []},
    }
    council = run_council(
        run,
        [],
        [
            {"model": "phi4-mini", "type": "ollama", "base_url": "http://127.0.0.1:11434"},
            {"model": "granite4:micro", "type": "ollama", "base_url": "http://127.0.0.1:11434"},
        ],
        under_test=run["connection"],
        chairman_name="phi4-mini",
        generate_for=generate_for,
    )
    assert "phi4-mini" in council["excluded_judges"]
    assert seen == ["granite4:micro", "llama3.2:3b"]
    assert council["chairman"] == "granite4:micro"
    assert "phi4-mini" not in seen


def test_granite_candidate_drops_granite_and_keeps_llama():
    seen = []

    def generate_for(judge):
        seen.append(judge["model"])

        def generate(_prompt: str) -> str:
            return "The failure count is 0."

        return generate

    run = {
        "run_id": "granite",
        "status": "completed",
        "validity": "ok",
        "connection": {"model": "granite4:micro"},
        "scorecard": {"categories": []},
    }
    run_council(
        run,
        [],
        [{"model": "granite4:micro", "type": "ollama", "base_url": "http://127.0.0.1:11434"}],
        under_test=run["connection"],
        generate_for=generate_for,
    )
    assert seen == ["llama3.2:3b"]


def test_llama_candidate_does_not_add_another_llama_judge():
    seen = []

    def generate_for(judge):
        seen.append(judge["model"])

        def generate(_prompt: str) -> str:
            return "The failure count is 0."

        return generate

    run = {
        "run_id": "llama",
        "status": "completed",
        "validity": "ok",
        "connection": {"model": "llama3.2:3b"},
        "scorecard": {"categories": []},
    }
    run_council(
        run,
        [],
        [
            {"model": "phi4-mini", "type": "ollama", "base_url": "http://127.0.0.1:11434"},
            {"model": "llama3.2:3b", "type": "ollama", "base_url": "http://127.0.0.1:11434"},
        ],
        under_test=run["connection"],
        generate_for=generate_for,
    )
    assert seen == ["phi4-mini"]
