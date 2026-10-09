"""Run lifecycle: progress, cancel, resume, and hashed files on disk."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_eval.datasets.loader import load_dataset
from llm_eval.garak.live import is_empty_generation, validity_from_items
from llm_eval_suite.compare import compare_runs
from llm_eval_suite.council import run_council
from llm_eval_suite.presets import estimate_preset, expand_preset
from llm_eval_suite.report_html import render_report
from llm_eval_suite.scoring import live_failures, scorecard, withhold_category_scores
from llm_eval_suite.suites import RUNNERS, SuiteContext, planned_suite_total
from llm_eval_suite.timing import TimingStore
from llm_eval_suite.worker import build_worker_command

logger = logging.getLogger("llm_eval_suite.runs")

# Half or more empty fact-check answers is an invalid run. A thinking model
# that spends max_tokens inside <think> and returns no answer trips this.
FACTCHECK_EMPTY_INVALID_RATE = 0.5
ETA_SMOOTHING = 0.3
ETA_MIN_ITEMS = 3
# Early garak probes are slower than later ones. The live rate only takes
# over once most of the suite is done, so its weight is the completed
# fraction cubed.
ETA_LIVE_WEIGHT_POWER = 3


def finished_progress(progress: dict, elapsed_seconds: float | None) -> dict:
    """Keep a finished suite visible as done, with its elapsed time.

    The next suite replaces the active progress row. This snapshot is what
    stays in the list, so a finished garak row does not fall back to
    Estimating.
    """
    done = dict(progress)
    if done.get("status") != "cancelled":
        done["status"] = "completed"
    done["eta_seconds"] = None
    if elapsed_seconds is not None:
        done["elapsed_seconds"] = round(float(elapsed_seconds), 1)
    return done


def blend_forecast(live_seconds: float | None, prior_seconds: float | None, fraction: float) -> float | None:
    """Blend the live remaining time with the pre-run remaining time.

    ``fraction`` is how much of the suite is already done, from 0 to 1.
    The live forecast's weight is that fraction to the power
    ``ETA_LIVE_WEIGHT_POWER``. A missing side is the other side alone.
    """
    if live_seconds is None and prior_seconds is None:
        return None
    if live_seconds is None:
        return float(prior_seconds)
    if prior_seconds is None:
        return float(live_seconds)
    done_fraction = min(1.0, max(0.0, float(fraction)))
    live_weight = done_fraction ** ETA_LIVE_WEIGHT_POWER
    return (live_weight * float(live_seconds)) + ((1.0 - live_weight) * float(prior_seconds))


def displayed_elapsed(state: dict, now: float) -> float:
    """Elapsed seconds for the progress row. Frozen once the suite hits its total."""
    frozen = state.get("frozen_elapsed")
    if frozen is not None:
        return float(frozen)
    started = state.get("started")
    if started is None:
        return 0.0
    return max(0.0, now - float(started))


def smooth_item_seconds(previous: float | None, sample: float, *, fallback: float, alpha: float = ETA_SMOOTHING) -> float:
    """Blend the latest item time into the running per-item rate."""
    base = fallback if previous is None else previous
    return (alpha * max(0.0, sample)) + ((1.0 - alpha) * base)


def factcheck_empty_validity(items: list[dict]) -> dict:
    """INVALID when half or more of the live fact-check answers are empty."""
    rows = [item for item in items if item.get("suite") == "factcheck" and item.get("source") == "live"]
    empty = sum(1 for item in rows if item.get("empty") or is_empty_generation(item.get("response")))
    total = len(rows)
    if total and empty / total >= FACTCHECK_EMPTY_INVALID_RATE:
        return {
            "validity": "invalid",
            "reason": f"{empty} of {total} fact-check answers were empty",
        }
    return {"validity": "ok", "reason": ""}


def _merge_validity(current: dict, extra: dict) -> dict:
    if extra.get("validity") != "invalid":
        return current
    reason = (extra.get("reason") or "").strip()
    prior = (current.get("reason") or "").strip()
    if reason and reason not in prior:
        prior = "; ".join(part for part in (prior, reason) if part)
    elif not prior:
        prior = reason
    return {"validity": "invalid", "reason": prior}


SUITE_FIELDS = (
    "name",
    "source",
    "label",
    "notes",
    "done",
    "total",
    "validity",
    "validity_reason",
    "empty_generations",
    "attack_success_rate",
    "pass_rate",
    "pass_rate_label",
    "wording",
    "garak_runs_dir",
    "report_dir",
    "log_path",
)


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
        "precision": connection.get("precision") or "",
        "thinking_max_tokens": connection.get("thinking_max_tokens"),
        "trust_remote_code": bool(connection.get("trust_remote_code")),
        "use_chat_template": connection.get("use_chat_template"),
        "max_new_tokens": connection.get("max_new_tokens"),
        "prompt_budget": connection.get("prompt_budget"),
    }


class RunManager:
    def __init__(self, runs_dir: str | Path, *, model_factory=None, timing_path: str | Path | None = None):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.model_factory = model_factory
        self.timing = TimingStore(timing_path) if timing_path else None
        self._cancel: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._eta_state: dict[tuple[str, str], dict] = {}

    def start(
        self,
        *,
        connection: dict,
        preset_id: str,
        dataset_path: str,
        resume_run_id: str | None = None,
        background: bool = True,
        in_process: bool | None = None,
        watch_cancel: bool = False,
        offline: bool | None = None,
    ) -> dict:
        if preset_id == "dow_bench":
            from dow_bench.meta import exclusion_reason

            reason = exclusion_reason(connection.get("model") or connection.get("name"))
            if reason:
                raise ValueError(reason)
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
        if offline:
            expanded = {**expanded, "offline": True}
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
                "max_context": connection.get("max_context"),
                "prompt_budget": connection.get("prompt_budget"),
                "thinking_budget": connection.get("thinking_max_tokens"),
                "answer_cap": connection.get("max_new_tokens"),
            }
        )
        suite_rates = {}
        if self.timing is not None:
            for suite_name in ("garak", "factcheck", "robustness", "consistency"):
                rate = self.timing.rate_for(connection, suite_name)
                if rate is not None:
                    suite_rates[suite_name] = rate
        record["estimate"] = estimate_preset(
            _preset_body(expanded),
            dataset_rows=len(rows),
            suite_rates=suite_rates,
        )
        record["log_path"] = str(run_dir / "run.log")
        record["validity"] = "ok"
        record["validity_reason"] = ""
        self._write_run(run_dir, record)
        if in_process is None:
            in_process = self.model_factory is not None
        if background and not in_process:
            self._spawn_worker(
                run_id=run_id,
                run_dir=run_dir,
                connection=connection,
                preset_id=preset_id,
                dataset_path=str(dataset_path),
                offline=bool(expanded.get("offline")),
            )
            return self.get(run_id)
        from llm_eval.offline import activate_run_offline

        data_dir = self.timing.path.parent if self.timing is not None else None
        activate_run_offline(preset=expanded, data_dir=data_dir)
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
                watch_cancel=watch_cancel,
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
        record["failures"] = live_failures(self._read_items(self.runs_dir / run_id))[:100]
        return record

    def items(self, run_id: str) -> list[dict]:
        return self._read_items(self.runs_dir / run_id)

    def list_runs(self) -> list[dict]:
        rows = []
        if not self.runs_dir.is_dir():
            return rows
        for path in self.runs_dir.iterdir():
            record = self._read_run(path)
            if record is None:
                continue
            invalid = record.get("validity") == "invalid" or record.get("status") == "invalid"
            percent = None
            if not invalid:
                percent = (record.get("scorecard") or {}).get("overall_pass_percent")
            rows.append(
                {
                    "run_id": record.get("run_id"),
                    "status": record.get("status"),
                    "preset": record.get("preset"),
                    "model": (record.get("connection") or {}).get("model"),
                    "created_at": record.get("created_at"),
                    "overall_pass_percent": percent,
                }
            )
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
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

    def analyze(self, run_id: str, judges: list[dict], chairman: str | None = None, max_tokens: int = 1200) -> dict:
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
            max_tokens=max_tokens,
        )
        record["analysis"] = analysis
        rejections = analysis.get("number_rejections") or []
        audit = record.setdefault("audit", {"events": []})
        audit["events"].append(
            {
                "at": _now(),
                "kind": "analysis",
                "mode": analysis.get("mode"),
                "number_guard": analysis.get("number_guard"),
                "number_rejections": rejections,
                "source_label": analysis.get("source_label"),
            }
        )
        audit["number_guard"] = analysis.get("number_guard")
        audit["number_rejections"] = rejections
        audit["summary_source"] = analysis.get("mode")
        run_dir = self.runs_dir / run_id
        for row in rejections:
            line = f"Number check rejected {row.get('token')} in: {row.get('sentence')}"
            self._append_log(run_dir, line)
            logger.info(line)
            logging.getLogger("uvicorn.error").info(line)
        self._write_run(run_dir, record)
        return analysis

    def _spawn_worker(
        self,
        *,
        run_id: str,
        run_dir: Path,
        connection: dict,
        preset_id: str,
        dataset_path: str,
        offline: bool = False,
    ) -> None:
        job = {
            "runs_dir": str(self.runs_dir),
            "timing_path": str(self.timing.path) if self.timing else "",
            "data_dir": str(self.timing.path.parent) if self.timing else "",
            "connection": connection,
            "preset_id": preset_id,
            "offline": bool(offline),
            "dataset_path": dataset_path,
            "resume_run_id": run_id,
        }
        job_path = run_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        command = build_worker_command(job_path)
        try:
            subprocess.Popen(command, shell=False)
        except OSError as exc:
            record = self._read_run(run_dir) or {}
            record["status"] = "failed"
            record["error"] = str(exc)
            self._write_run(run_dir, record)
        self._append_log(run_dir, "Started background worker.")

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
        watch_cancel: bool = False,
    ) -> None:
        items = list(existing_items)
        suites = []
        started = time.perf_counter()
        new_live = 0
        model = None
        self._append_log(run_dir, f"Run {run_id} started.")
        if watch_cancel:
            threading.Thread(target=self._watch_cancel, args=(run_dir, cancel), daemon=True).start()
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
                suite_name = suite["name"]
                total = planned_suite_total(suite_name, suite, len(rows))
                now = time.perf_counter()
                self._eta_state[(run_id, suite_name)] = {"started": now, "last": now, "rate": None}
                progress = {
                    "name": suite_name,
                    "done": 0,
                    "total": total,
                    "status": "running",
                    "eta_seconds": None,
                }
                self._update_progress(run_dir, progress, suites)

                def on_item(item, progress=progress):
                    nonlocal new_live
                    if item["id"] in completed:
                        return
                    completed.add(item["id"])
                    items.append(item)
                    self._append_item(run_dir, item)
                    if item.get("source") == "live":
                        new_live += 1
                    # Garak progress is the cumulative attempt count from on_progress.
                    if suite_name != "garak":
                        progress["done"] = int(progress.get("done") or 0) + 1
                        self._stamp_eta(run_dir, progress, run_id)
                    self._update_progress(run_dir, progress, suites)

                def on_suite_progress(done, progress=progress):
                    progress["done"] = max(int(progress.get("done") or 0), int(done))
                    progress["status"] = "running"
                    self._stamp_eta(run_dir, progress, run_id)
                    self._update_progress(run_dir, progress, suites)

                ctx = SuiteContext(
                    model=model,
                    connection=connection,
                    dataset_rows=rows,
                    run_dir=run_dir,
                    cancel=cancel,
                    completed_ids=set(completed),
                    on_item=on_item,
                    on_progress=on_suite_progress,
                )
                # completed_ids at start should be the pre-resume set, not grow mid-run
                ctx.completed_ids = set(self._completed_ids_snapshot(existing_items))
                result = runner.run(ctx, suite)
                state = self._eta_state.get((run_id, suite_name)) or {}
                elapsed = None
                if state.get("started") is not None or state.get("frozen_elapsed") is not None:
                    elapsed = displayed_elapsed(state, time.perf_counter())
                progress["status"] = "cancelled" if result.get("cancelled") or cancel.is_set() else "completed"
                progress["done"] = result.get("done", progress["done"])
                progress["total"] = result.get("total", progress["done"])
                progress["source"] = result.get("source")
                progress["label"] = result.get("label")
                progress.update(finished_progress(progress, elapsed))
                summary = {key: result[key] for key in SUITE_FIELDS if key in result}
                summary["done"] = progress["done"]
                summary["total"] = progress["total"]
                summary["status"] = progress["status"]
                if progress.get("elapsed_seconds") is not None:
                    summary["elapsed_seconds"] = progress["elapsed_seconds"]
                suites.append(summary)
                if (
                    self.timing is not None
                    and progress.get("status") == "completed"
                    and int(progress.get("done") or 0) > 0
                ):
                    state = self._eta_state.get((run_id, suite_name))
                    if state is not None:
                        elapsed = time.perf_counter() - float(state["started"])
                        self.timing.record(
                            elapsed,
                            int(progress["done"]),
                            connection=connection,
                            suite=suite_name,
                        )
                self._append_log(run_dir, f"{result['name']} finished ({result.get('source')}).")
                self._update_progress(run_dir, progress, suites)
            status = "cancelled" if cancel.is_set() else "completed"
            self._finalize(
                run_dir, items, suites, status, error=None, started=started, new_live=new_live, model=model
            )
        except Exception as exc:  # noqa: BLE001 - stored on the run record
            self._append_log(run_dir, f"Run failed: {exc}")
            self._finalize(
                run_dir,
                items,
                suites,
                "failed",
                error=str(exc),
                started=started,
                new_live=new_live,
                model=model,
            )

    def _finalize(
        self,
        run_dir: Path,
        items: list[dict],
        suites: list[dict],
        status: str,
        error: str | None,
        started: float | None = None,
        new_live: int = 0,
        model=None,
    ):
        record = self._read_run(run_dir) or {}
        record["status"] = status
        record["completed_at"] = _now()
        record["suites"] = suites
        record["scorecard"] = scorecard(items)
        record["error"] = error
        record["items_completed"] = len(items)
        record["hit_token_cap_count"] = sum(1 for item in items if item.get("hit_token_cap") is True)
        record["think_truncated_count"] = sum(1 for item in items if item.get("think_truncated") is True)
        record["prompt_over_budget_count"] = sum(1 for item in items if item.get("prompt_over_budget") is True)
        record["stopped_on_newline_run_count"] = sum(
            1 for item in items if item.get("stopped_on_newline_run") is True
        )
        record["log_path"] = str(run_dir / "run.log")
        fact_validity = factcheck_empty_validity(items)
        if fact_validity["validity"] == "invalid":
            for suite in suites:
                if suite.get("name") == "factcheck":
                    suite["validity"] = "invalid"
                    suite["validity_reason"] = fact_validity["reason"]
                    note = suite.get("notes") or ""
                    if fact_validity["reason"] not in note:
                        suite["notes"] = ("INVALID fact-check. " + fact_validity["reason"] + " " + note).strip()
        validity = validity_from_items(
            [item for item in items if item.get("suite") == "garak" and item.get("source") == "live"]
        )
        validity = _merge_validity(validity, fact_validity)
        for suite in suites:
            if suite.get("validity") == "invalid":
                validity = _merge_validity(
                    validity,
                    {"validity": "invalid", "reason": suite.get("validity_reason") or ""},
                )
            if suite.get("garak_runs_dir"):
                record["garak_runs_dir"] = suite["garak_runs_dir"]
            if suite.get("pass_rate_label"):
                record["garak_pass_rate"] = suite.get("pass_rate")
                record["garak_attack_success_rate"] = suite.get("attack_success_rate")
                record["garak_pass_rate_label"] = suite.get("pass_rate_label")
                record["garak_wording"] = suite.get("wording")
        record["validity"] = validity.get("validity") or "ok"
        record["validity_reason"] = validity.get("reason") or ""
        device_name = getattr(model, "device_name", None)
        if device_name:
            record["device"] = device_name
        precision_used = getattr(model, "precision_used", None)
        if precision_used and not (record.get("connection") or {}).get("precision"):
            record["precision"] = precision_used
        elif (record.get("connection") or {}).get("precision"):
            record["precision"] = record["connection"]["precision"]
        mamba_path = getattr(model, "mamba_path", None)
        if mamba_path:
            record["mamba_path"] = mamba_path
        security = next(
            (
                row
                for row in (record.get("scorecard") or {}).get("categories") or []
                if row.get("category") == "security_jailbreak" and row.get("source") == "live"
            ),
            None,
        )
        if security and security.get("pass_rate") is not None:
            record["garak_pass_rate"] = security["pass_rate"]
            record["garak_attack_success_rate"] = round(1.0 - float(security["pass_rate"]), 4)
            record["garak_pass_rate_label"] = record.get("garak_pass_rate_label") or "Pass rate (1 - ASR)"
        if record["validity"] == "invalid" and status == "completed":
            record["status"] = "invalid"
        if record.get("validity") == "invalid" or record.get("status") == "invalid":
            record["scorecard"] = withhold_category_scores(record.get("scorecard") or {})
            record["garak_pass_rate"] = None
            record["garak_attack_success_rate"] = None
        self._write_run(run_dir, record)
        self._write_manifest(run_dir, record)

    def _stamp_eta(self, run_dir: Path, progress: dict, run_id: str) -> None:
        if progress.get("status") in {"completed", "cancelled"}:
            progress["eta_seconds"] = None
            return
        record = self._read_run(run_dir) or {}
        estimate = record.get("estimate") or {}
        total = progress.get("total")
        done = int(progress.get("done") or 0)
        now = time.perf_counter()
        key = (run_id, str(progress.get("name") or ""))
        state = self._eta_state.get(key)
        if state is None:
            state = {"started": now, "mark": now, "done": 0, "rate": None}
            self._eta_state[key] = state
        if "mark" not in state:
            state["mark"] = state.get("last", state.get("started", now))
        previous_done = int(state.get("done") or 0)
        # The clock for the rate starts at the first completed attempt.
        # Garak's process startup sits before that attempt and is not a prompt.
        # A later poll that reports a smaller count does not move the baseline.
        if done > previous_done and done > 0:
            if state.get("timed_at") is None:
                state["timed_at"] = now
                state["timed_done"] = done
            else:
                gained = done - int(state.get("timed_done") or 0)
                elapsed = max(0.0, now - float(state["timed_at"]))
                if gained > 0 and elapsed > 0:
                    state["rate"] = elapsed / gained
            state["mark"] = now
            state["done"] = done
        total_count = None if total is None else int(total)
        if total_count and done >= total_count and state.get("frozen_elapsed") is None:
            state["frozen_elapsed"] = max(0.0, now - float(state["started"]))
        suite_name = str(progress.get("name") or "")
        suite_rates = estimate.get("suite_rates") or {}
        known = suite_rates.get(suite_name)
        # Garak's stored suite rate still folds startup into every prompt, so
        # the early ETA runs about 40% high. Use the measured per-prompt rate
        # the pre-run estimate already shows, then blend toward the rate
        # observed after the first completed attempt.
        measured_rate = estimate.get("seconds_per_prompt")
        if (
            suite_name == "garak"
            and estimate.get("estimate_source") == "measured"
            and measured_rate is not None
        ):
            known = float(measured_rate)
        observed = state.get("rate") if done >= ETA_MIN_ITEMS and state.get("rate") else None
        rate = observed if observed is not None else (float(known) if known is not None else None)
        remaining = None if total_count is None else max(0, total_count - done)
        progress["elapsed_seconds"] = round(displayed_elapsed(state, now), 1)
        if remaining is None or remaining <= 0 or (observed is None and rate is None):
            progress["seconds_per_prompt"] = None if rate is None else round(rate, 3)
            progress["eta_seconds"] = None
            progress["eta_source"] = "estimating" if rate is None else "measured"
        elif observed is not None:
            live_eta = remaining * float(observed)
            prior_eta = None if known is None else remaining * float(known)
            fraction = 0.0 if not total_count else done / total_count
            blended = blend_forecast(live_eta, prior_eta, fraction)
            progress["eta_seconds"] = None if blended is None else round(blended, 1)
            progress["seconds_per_prompt"] = None if blended is None else round(blended / remaining, 3)
            progress["eta_source"] = "measured"
        else:
            progress["seconds_per_prompt"] = round(float(rate), 3)
            progress["eta_seconds"] = round(remaining * float(rate), 1)
            progress["eta_source"] = "matched"

    def _watch_cancel(self, run_dir: Path, cancel: threading.Event) -> None:
        while not cancel.is_set():
            record = self._read_run(run_dir) or {}
            if record.get("status") == "cancel_requested":
                cancel.set()
                return
            time.sleep(0.2)

    def _append_log(self, run_dir: Path, text: str) -> None:
        path = run_dir / "run.log"

        def write() -> None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text.rstrip() + "\n")

        _retry_io(write)

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


def _preset_body(expanded: dict) -> dict:
    body = {}
    for suite in expanded.get("suites") or []:
        name = suite.get("name")
        if not name:
            continue
        body[name] = {key: value for key, value in suite.items() if key != "name"}
    return body
