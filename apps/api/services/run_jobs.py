import logging
import threading
from datetime import datetime
from uuid import uuid4

from llm_eval.core.run_service import RunService

logger = logging.getLogger(__name__)


class RunJob:
    """In-memory live execution state for a run (not the long-term source of truth)."""

    def __init__(
        self,
        run_id: str,
        *,
        output_dir: str,
        dry_run: bool,
        run_name: str,
        created_at: datetime | None = None,
    ):
        self.run_id = run_id
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.run_name = run_name
        self.status = "pending"
        self.created_at = created_at or datetime.now()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error_message: str | None = None
        self.message: str | None = None


class RunJobManager:
    """
    Lightweight in-process job registry for local run execution.

    Designed for a single local API process. An in-memory dict tracks live
    pending/running state; the filesystem run index and audit JSON remain the
    long-term source of truth after a run finishes.
    """

    def __init__(self, run_service: RunService):
        self._run_service = run_service
        self._jobs: dict[str, RunJob] = {}
        self._lock = threading.RLock()

    def create_run_id(self) -> str:
        return uuid4().hex[:12]

    def submit(
        self,
        config: dict,
        *,
        dry_run: bool,
        compare: bool,
        run_id: str,
        output_dir: str,
        run_name: str,
    ) -> RunJob:
        job = RunJob(
            run_id=run_id,
            output_dir=output_dir,
            dry_run=dry_run,
            run_name=run_name,
        )
        with self._lock:
            self._jobs[run_id] = job

        thread = threading.Thread(
            target=self._execute,
            args=(job, config, dry_run, compare),
            name=f"run-job-{run_id}",
            daemon=True,
        )
        thread.start()
        return job

    def get_live_job(self, run_id: str) -> RunJob | None:
        with self._lock:
            return self._jobs.get(run_id)

    def list_live_jobs(self) -> list[RunJob]:
        with self._lock:
            return list(self._jobs.values())

    def get_merged_run(self, run_id: str, output_dir: str) -> dict | None:
        with self._lock:
            live = self._jobs.get(run_id)
        persisted = self._run_service.get_run(run_id, output_dir)
        if persisted is None and live is None:
            return None
        if persisted is None and live is not None:
            return self._job_to_dict(live)
        if live is not None and live.status in {"pending", "running"}:
            merged = dict(persisted)
            merged.update(self._job_to_dict(live))
            return merged
        return persisted

    def list_merged_runs(self, output_dir: str) -> list[dict]:
        persisted = {entry["run_id"]: entry for entry in self._run_service.list_runs(output_dir)}
        with self._lock:
            live_jobs = list(self._jobs.values())
        for job in live_jobs:
            if job.output_dir != output_dir:
                continue
            job_dict = self._job_to_dict(job)
            if job.run_id in persisted and job.status in {"pending", "running"}:
                merged = dict(persisted[job.run_id])
                merged.update(job_dict)
                persisted[job.run_id] = merged
            elif job.run_id not in persisted:
                persisted[job.run_id] = job_dict
        runs = list(persisted.values())
        runs.sort(key=lambda item: item.get("started_at") or item.get("created_at") or "", reverse=True)
        return runs

    def _execute(self, job: RunJob, config: dict, dry_run: bool, compare: bool) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = datetime.now()
        try:
            if dry_run:
                result = self._run_service.run_dry_run(config, run_id=job.run_id)
            else:
                result = self._run_service.start_run(config, compare=compare, run_id=job.run_id)
            with self._lock:
                job.status = result.status
                job.message = result.message
                job.error_message = result.error_message
        except Exception as exc:
            logger.exception("Background run %s failed unexpectedly", job.run_id)
            with self._lock:
                job.status = "failed_runtime"
                job.error_message = str(exc)
                job.message = f"Run failed: {exc}"
        finally:
            with self._lock:
                job.completed_at = datetime.now()

    @staticmethod
    def _job_to_dict(job: RunJob) -> dict:
        return {
            "run_id": job.run_id,
            "run_name": job.run_name,
            "status": job.status,
            "dry_run": job.dry_run,
            "output_dir": job.output_dir,
            "started_at": job.started_at.isoformat() if job.started_at else job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "created_at": job.created_at.isoformat(),
            "error_message": job.error_message,
            "message": job.message,
            "model_names": [],
            "evaluator_names": [],
        }
