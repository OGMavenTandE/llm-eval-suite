import json
from datetime import datetime
from pathlib import Path

from llm_eval.storage.index import RunIndex


def test_index_file_created_on_first_register(tmp_path: Path):
    output_dir = tmp_path / "results"
    index = RunIndex(str(output_dir))

    assert not index.index_path.exists()

    index.register_run(
        run_id="first-run",
        run_name="alpha",
        status="validated",
        dry_run=True,
    )

    assert index.index_path.exists()
    assert index.index_path.name == RunIndex.INDEX_FILENAME


def test_register_list_and_order(tmp_path: Path):
    output_dir = tmp_path / "results"
    index = RunIndex(str(output_dir))

    index.register_run(
        run_id="older",
        run_name="older-run",
        status="validated",
        dry_run=True,
        started_at=datetime(2026, 1, 1, 10, 0, 0),
    )
    index.register_run(
        run_id="newer",
        run_name="newer-run",
        status="completed",
        dry_run=False,
        started_at=datetime(2026, 1, 2, 10, 0, 0),
    )

    runs = index.list_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] == "newer"
    assert runs[1]["run_id"] == "older"


def test_get_run_by_id(tmp_path: Path):
    output_dir = tmp_path / "results"
    index = RunIndex(str(output_dir))

    index.register_run(
        run_id="run-42",
        run_name="lookup-test",
        status="completed",
        dry_run=False,
        output_dir=str(output_dir),
        audit_path=str(output_dir / "audit" / "run-42.json"),
        config_hash="abc123",
        dataset_path="datasets/sample.jsonl",
        model_names=["model-a"],
        evaluator_names=["correctness"],
    )

    entry = index.get_run("run-42")
    assert entry is not None
    assert entry["run_name"] == "lookup-test"
    assert entry["audit_path"] == str(output_dir / "audit" / "run-42.json")
    assert entry["config_hash"] == "abc123"
    assert entry["model_names"] == ["model-a"]
    assert index.get_run("missing-id") is None


def test_register_updates_existing_run(tmp_path: Path):
    output_dir = tmp_path / "results"
    index = RunIndex(str(output_dir))

    index.register_run(run_id="run-1", run_name="alpha", status="validated", dry_run=True)
    index.register_run(
        run_id="run-1",
        run_name="alpha",
        status="completed",
        dry_run=False,
        error_message=None,
    )

    runs = index.list_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"

    with index.index_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1


def test_uses_temp_directory_isolated_per_output_dir(tmp_path: Path):
    index_a = RunIndex(str(tmp_path / "results-a"))
    index_b = RunIndex(str(tmp_path / "results-b"))

    index_a.register_run(run_id="a1", run_name="a", status="validated", dry_run=True)
    index_b.register_run(run_id="b1", run_name="b", status="validated", dry_run=True)

    assert len(index_a.list_runs()) == 1
    assert len(index_b.list_runs()) == 1
    assert index_a.get_run("b1") is None
    assert index_b.get_run("a1") is None
