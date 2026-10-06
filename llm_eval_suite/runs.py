"""Run lifecycle: progress, cancel, resume, and hashed files on disk."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_eval.datasets.loader import load_dataset
from llm_eval_suite.compare import compare_runs
from llm_eval_suite.council import run_council
from llm_eval_suite.presets import expand_preset
from llm_eval_suite.report_html import render_report
from llm_eval_suite.scoring import failing_items, scorecard
from llm_eval_suite.suites import RUNNERS, SuiteContext


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retry_io(action, attempts: int = 10):
    """Retry file operations that Windows denies while another thread has the file open."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return action()
        except (PermissionError, OSError) as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    return None


def _public_connection(connection: dict) -> dict:
    return {
        "id": connection.get("id"),
        "name": connection.get("name"),
        "type": connection.get("type"),
        "base_url": connection.get("base_url"),
        "model": connection.get("model"),
        "mode": connection.get("mode"),
        "max_context": connection.get("max_context"),
        "folder": connection.get("folder"),
        "cloud": bool(connection.get("cloud")),
    }


class RunManager:
    def __init__(self, runs_dir: str | Path, *, model_factory=None):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.model_factory = model_factory
        self._cancel: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        connection: dict,
        preset_id: str,
        dataset_path: str,
        resume_run_id: str | None = None,
        background: bool = True,
    ) -> dict:
        if resume_run_id:
            run_id = resume_run_id
            run_dir = self.runs_dir / run_id
            if not (run_dir / "run.json").is_file():
                raise FileNotFoundError(f"No run to resume: {resume_run_id}")
        else:
            run_id = uuid.uuid4().hex[:12]
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        expanded = expand_preset(preset_id)
        rows = load_dataset(dataset_path)
        dataset_hash = sha256_file(Path(dataset_path))
        completed = self._completed_ids(run_dir)
        existing_items = self._read_items(run_dir)
        record = self._read_run(run_dir) or {}
        record.update(
            {
                "run_id": run_id,
                "status": "running",
                "preset": preset_id,
                "preset_snapshot": expanded,
                "connection": _public_connection(connection),
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": dataset_hash,
                    "row_count": len(rows),
                },
                "created_at": record.get("created_at") or _now(),
                "completed_at": None,
                "progress": {"suites": []},
                "suites": [],
                "scorecard": {},
                "items_completed": len(completed),
                "error": None,
                "audit": record.get("audit") or {"events": []},
            }
        )
        self._write_run(run_dir, record)
        cancel = threading.Event()
        with self._lock:
            self._cancel[run_id] = cancel

        def _job():
            self._execute(
                run_id=run_id,
                run_dir=run_dir,
                connection=connection,
                expanded=expanded,
                rows=rows,
                cancel=cancel,
                completed=completed,
                existing_items=existing_items,
            )

        if background:
            thread = threading.Thread(target=_job, daemon=True)
            with self._lock:
                self._threads[run_id] = thread
            thread.start()
        else:
            _job()
        return self.get(run_id)

    def cancel(self, run_id: str) -> dict:
        event = self._cancel.get(run_id)
        if event is not None:
            event.set()
        run_dir = self.runs_dir / run_id
        record = self._read_run(run_dir)
        if record is None:
            raise FileNotFoundError(run_id)
        if record.get("status") == "running":
            record["status"] = "cancel_requested"
            self._write_run(run_dir, record)
        return record

    def get(self, run_id: str) -> dict:
        record = self._read_run(self.runs_dir / run_id)
        if record is None:
            raise FileNotFoundError(run_id)
        record["failures"] = failing_items(self._read_items(self.runs_dir / run_id))[:100]
        return record

    def items(self, run_id: str) -> list[dict]:
        return self._read_items(self.runs_dir / run_id)

    def list_runs(self) -> list[dict]:
        rows = []
        if not self.runs_dir.is_dir():
            return rows
        for path in sorted(self.runs_dir.iterdir(), reverse=True):
            record = self._read_run(path)
            if record is None:
                continue
            rows.append(
                {
                    "run_id": record.get("run_id"),
                    "status": record.get("status"),
                    "preset": record.get("preset"),
                    "model": (record.get("connection") or {}).get("model"),
                    "created_at": record.get("created_at"),
                    "overall_pass_percent": (record.get("scorecard") or {}).get("overall_pass_percent"),
                }
            )
        return rows

    def compare(self, left_id: str, right_id: str) -> dict:
        left = self._read_run(self.runs_dir / left_id)
        right = self._read_run(self.runs_dir / right_id)
        if left is None or right is None:
            raise FileNotFoundError("Both runs must exist.")
        return compare_runs(left, right, self.items(left_id), self.items(right_id))

    def report_html(self, run_id: str) -> str:
        record = self._read_run(self.runs_dir / run_id)
        if record is None:
            raise FileNotFoundError(run_id)
        return render_report(record, self.items(run_id))

    def analyze(self, run_id: str, judges: list[dict], chairman: str | None = None) -> dict:
        record = self._read_run(self.runs_dir / run_id)
        if record is None:
            raise FileNotFoundError(run_id)
        items = self.items(run_id)
        analysis = run_council(
            record,
            items,
            judges,
            under_test=record.get("connection") or {},
            chairman_name=chairman,
        )
        record["analysis"] = analysis
        audit = record.setdefault("audit", {"events": []})
        audit["events"].append(
            {
                "at": _now(),
                "kind": "analysis",
                "mode": analysis.get("mode"),
                "number_guard": analysis.get("number_guard"),
                "source_label": analysis.get("source_label"),
            }
        )
        audit["number_guard"] = analysis.get("number_guard")
        audit["summary_source"] = analysis.get("mode")
        self._write_run(self.runs_dir / run_id, record)
        return analysis

    def _execute(
        self,
        *,
        run_id: str,
        run_dir: Path,
        connection: dict,
        expanded: dict,
        rows: list[dict],
        cancel: threading.Event,
        completed: set[str],
        existing_items: list[dict],
    ) -> None:
        items = list(existing_items)
        suites = []
        try:
            if self.model_factory is not None:
                model = self.model_factory(connection)
            else:
                from llm_eval_suite.connections import build_model

                model = build_model(connection)
            for suite in expanded["suites"]:
                if cancel.is_set():
                    break
                runner = RUNNERS[suite["name"]]
                progress = {
                    "name": suite["name"],
                    "done": 0,
                    "total": None,
                    "status": "running",
                }
                self._update_progress(run_dir, progress, suites)

                def on_item(item, progress=progress):
                    if item["id"] in completed:
                        return
                    completed.add(item["id"])
                    items.append(item)
                    self._append_item(run_dir, item)
                    progress["done"] = int(progress.get("done") or 0) + 1
                    self._update_progress(run_dir, progress, suites)

                ctx = SuiteContext(
                    model=model,
                    connection=connection,
                    dataset_rows=rows,
                    run_dir=run_dir,
                    cancel=cancel,
                    completed_ids=set(completed),
                    on_item=on_item,
                )
                # completed_ids at start should be the pre-resume set, not grow mid-run
                ctx.completed_ids = set(self._completed_ids_snapshot(existing_items))
                result = runner.run(ctx, suite)
                progress["status"] = "cancelled" if result.get("cancelled") or cancel.is_set() else "completed"
                progress["done"] = result.get("done", progress["done"])
                progress["total"] = result.get("total", progress["done"])
                progress["source"] = result.get("source")
                progress["label"] = result.get("label")
                suites.append(
                    {
                        "name": result["name"],
                        "source": result["source"],
                        "label": result["label"],
                        "notes": result["notes"],
                        "done": progress["done"],
                        "total": progress["total"],
                    }
                )
                self._update_progress(run_dir, progress, suites)
            status = "cancelled" if cancel.is_set() else "completed"
            self._finalize(run_dir, items, suites, status, error=None)
        except Exception as exc:  # noqa: BLE001 - stored on the run record
            self._finalize(run_dir, items, suites, "failed", error=str(exc))

    def _finalize(self, run_dir: Path, items: list[dict], suites: list[dict], status: str, error: str | None):
        record = self._read_run(run_dir) or {}
        record["status"] = status
        record["completed_at"] = _now()
        record["suites"] = suites
        record["scorecard"] = scorecard(items)
        record["error"] = error
        record["items_completed"] = len(items)
        self._write_run(run_dir, record)
        self._write_manifest(run_dir, record)

    def _update_progress(self, run_dir: Path, current: dict, suites: list[dict]) -> None:
        record = self._read_run(run_dir) or {}
        if record.get("status") == "cancel_requested":
            record["status"] = "running"
        known = {row["name"]: row for row in suites}
        known[current["name"]] = current
        record["progress"] = {"suites": list(known.values())}
        record["items_completed"] = self._count_items(run_dir)
        self._write_run(run_dir, record)

    def _completed_ids(self, run_dir: Path) -> set[str]:
        return {item["id"] for item in self._read_items(run_dir) if item.get("id")}

    def _completed_ids_snapshot(self, items: list[dict]) -> set[str]:
        return {item["id"] for item in items if item.get("id")}

    def _count_items(self, run_dir: Path) -> int:
        path = run_dir / "items.jsonl"
        if not path.is_file():
            return 0

        def read() -> int:
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

        return int(_retry_io(read) or 0)

    def _append_item(self, run_dir: Path, item: dict) -> None:
        path = run_dir / "items.jsonl"
        line = json.dumps(item) + "\n"

        def write() -> None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

        _retry_io(write)

    def _read_items(self, run_dir: Path) -> list[dict]:
        path = run_dir / "items.jsonl"
        if not path.is_file():
            return []

        def read() -> list[dict]:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return rows

        return _retry_io(read) or []

    def _read_run(self, run_dir: Path) -> dict | None:
        path = run_dir / "run.json"
        if not path.is_file():
            return None

        def read() -> dict:
            return json.loads(path.read_text(encoding="utf-8"))

        try:
            return _retry_io(read)
        except json.JSONDecodeError:
            time.sleep(0.05)
            return _retry_io(read)

    def _write_run(self, run_dir: Path, record: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / "run.json"
        temporary = run_dir / "run.json.tmp"
        payload = json.dumps(record, indent=2) + "\n"

        def write() -> None:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(target)

        _retry_io(write)

    def _write_manifest(self, run_dir: Path, record: dict) -> None:
        run_path = run_dir / "run.json"
        items_path = run_dir / "items.jsonl"
        if not items_path.is_file():
            items_path.write_text("", encoding="utf-8")
        manifest = {
            "run_id": record.get("run_id"),
            "created_at": _now(),
            "inputs": {
                "dataset_sha256": (record.get("dataset") or {}).get("sha256"),
                "preset": record.get("preset"),
                "preset_sha256": sha256_bytes(
                    json.dumps(record.get("preset_snapshot"), sort_keys=True).encode("utf-8")
                ),
                "connection": record.get("connection"),
            },
            "outputs": {
                "run_json_sha256": sha256_file(run_path),
                "items_jsonl_sha256": sha256_file(items_path),
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
