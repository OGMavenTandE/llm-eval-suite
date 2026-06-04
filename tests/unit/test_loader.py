import json
from pathlib import Path

import pytest

from llm_eval.datasets.loader import REQUIRED_FIELDS, load_dataset


def test_load_valid_jsonl_dataset(sample_dataset_path: Path):
    rows = load_dataset(str(sample_dataset_path))
    assert len(rows) == 2
    assert rows[0]["prompt"] == "What is 2+2?"
    assert rows[0]["expected_answer"] == "4"


def test_missing_required_field_raises(invalid_dataset_path: Path):
    with pytest.raises(ValueError, match="missing required field"):
        load_dataset(str(invalid_dataset_path))


def test_empty_prompt_raises(tmp_path: Path):
    path = tmp_path / "empty_prompt.jsonl"
    path.write_text(json.dumps({"prompt": "  ", "expected_answer": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'prompt' must be a non-empty string"):
        load_dataset(str(path))


def test_empty_dataset_raises(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset is empty"):
        load_dataset(str(path))


def test_unsupported_format_raises(tmp_path: Path):
    path = tmp_path / "data.txt"
    path.write_text("not a dataset", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(str(path))


def test_required_fields_constant():
    assert REQUIRED_FIELDS == {"prompt", "expected_answer"}
