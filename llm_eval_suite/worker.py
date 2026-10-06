"""Background run worker. Commands are argument lists, never shell strings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def build_worker_command(job_path: str | Path) -> list[str]:
    return [sys.executable, "-m", "llm_eval_suite.worker", str(job_path)]


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: python -m llm_eval_suite.worker JOB.json")
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    from llm_eval_suite.runs import RunManager

    manager = RunManager(job["runs_dir"], timing_path=job.get("timing_path") or None)
    manager.start(
        connection=job["connection"],
        preset_id=job["preset_id"],
        dataset_path=job["dataset_path"],
        resume_run_id=job.get("resume_run_id"),
        background=False,
        in_process=True,
        watch_cancel=True,
    )


if __name__ == "__main__":
    main()
