import json
from datetime import datetime
from pathlib import Path


class RunIndex:
    """
    Lightweight filesystem index of evaluation runs.

    Stores a JSON file at ``{output_dir}/.llm_eval_runs.json``.
    """

    INDEX_FILENAME = ".llm_eval_runs.json"

    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / self.INDEX_FILENAME

    def _load(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        with self.index_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data

    def _save(self, entries: list[dict]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)

    def register_run(
        self,
        run_id: str,
        run_name: str,
        status: str,
        *,
        dry_run: bool = False,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        config_path: str | None = None,
        artifact_paths: dict | None = None,
    ) -> dict:
        """Add or update a run entry in the index."""
        entries = self._load()
        started = started_at or datetime.now()
        entry = {
            "run_id": run_id,
            "run_name": run_name,
            "status": status,
            "dry_run": dry_run,
            "started_at": started.isoformat(),
            "completed_at": completed_at.isoformat() if completed_at else None,
            "config_path": config_path,
            "artifact_paths": artifact_paths,
        }

        updated = False
        for i, existing in enumerate(entries):
            if existing.get("run_id") == run_id:
                entries[i] = {**existing, **entry}
                updated = True
                break

        if not updated:
            entries.append(entry)

        self._save(entries)
        return entry

    def list_runs(self) -> list[dict]:
        """Return all indexed runs, newest first."""
        entries = self._load()
        return sorted(entries, key=lambda e: e.get("started_at", ""), reverse=True)

    def get_run(self, run_id: str) -> dict | None:
        """Look up a run by ID."""
        for entry in self._load():
            if entry.get("run_id") == run_id:
                return entry
        return None
