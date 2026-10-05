"""In-memory stand-in for the subset of the Dioptra JSON client this pilot uses.

The live client is ``connect_json_dioptra_client()`` against ``DIOPTRA_API``
(default http://localhost). This class records the same kind of calls and
never opens a socket.

Differences from Dioptra 1.1.0, on purpose:

- ``experiments.create`` takes entrypoint names, not integer entrypoint ids.
- ``group_id`` is not accepted. Dioptra requires one from a real deployment.
- Job ids are local strings. Dioptra assigns integers after submission.
"""

import math
from datetime import datetime, timezone

from llm_eval.dioptra.schema import (
    PILOT_NOTE,
    STATUS_IF_SUBMITTED,
    EntrypointRecord,
    ExperimentRecord,
    JobRecord,
    MetricRecord,
    PilotBundle,
    RestExample,
)


def normalize_metric_value(metric_value: float | str) -> float | str:
    """Match Dioptra's metric encoding: NaN and infinities become strings."""
    if isinstance(metric_value, str):
        if metric_value in {"nan", "inf", "-inf"}:
            return metric_value
        raise ValueError(
            "metric value string must be 'nan', 'inf', or '-inf'; "
            "otherwise pass a float"
        )
    try:
        numeric = float(metric_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric value must be a float") from exc
    if math.isnan(numeric):
        return "nan"
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    return numeric


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _ExperimentCollection:
    def __init__(self, client: "OfflineDioptraClient"):
        self._client = client

    def create(
        self,
        name: str,
        description: str | None = None,
        entrypoints: list[str] | None = None,
        *,
        group_label: str = "local",
        created_at: datetime | None = None,
    ) -> ExperimentRecord:
        if any(item.name == name for item in self._client._experiments):
            raise ValueError(f"experiment name already exists in this client: {name}")
        entrypoint_records = [
            EntrypointRecord(
                name=entrypoint_name,
                description="llm-eval-suite YAML evaluation config.",
                registered=False,
            )
            for entrypoint_name in (entrypoints or [])
        ]
        experiment = ExperimentRecord(
            name=name,
            description=description or "",
            group_label=group_label,
            entrypoints=entrypoint_records,
            created_at=created_at or _utcnow(),
        )
        self._client._experiments.append(experiment)
        return experiment


class _JobCollection:
    def __init__(self, client: "OfflineDioptraClient"):
        self._client = client

    def create(
        self,
        experiment_name: str,
        name: str,
        entrypoint_name: str,
        *,
        suite_status: str,
        parameters: dict | None = None,
        metrics_source: str,
        artifact_paths: list[str] | None = None,
        local_id: str | None = None,
    ) -> JobRecord:
        experiment = self._client.get_experiment(experiment_name)
        if not any(item.name == entrypoint_name for item in experiment.entrypoints):
            raise ValueError(
                f"entrypoint {entrypoint_name!r} is not on experiment {experiment_name!r}"
            )
        job_id = local_id or f"job-{len(experiment.jobs) + 1}"
        if any(item.local_id == job_id for item in experiment.jobs):
            raise ValueError(f"job local_id already exists: {job_id}")
        job = JobRecord(
            local_id=job_id,
            name=name,
            entrypoint_name=entrypoint_name,
            suite_status=suite_status,
            submitted=False,
            dioptra_status_if_submitted=STATUS_IF_SUBMITTED.get(suite_status),
            parameters=dict(parameters or {}),
            metrics=[],
            metrics_source=metrics_source,
            artifact_paths=list(artifact_paths or []),
        )
        experiment.jobs.append(job)
        return job

    def append_metric(
        self,
        job_local_id: str,
        metric_name: str,
        metric_value: float | str,
        metric_step: int | None = None,
        timestamp: datetime | None = None,
    ) -> MetricRecord:
        job = self._client.get_job(job_local_id)
        step = 0 if metric_step is None else metric_step
        if any(item.name == metric_name and item.step == step for item in job.metrics):
            raise ValueError(
                f"metric primary key already exists on {job_local_id}: "
                f"{metric_name} step {step}"
            )
        metric = MetricRecord(
            name=metric_name,
            value=normalize_metric_value(metric_value),
            step=step,
            timestamp=timestamp,
        )
        job.metrics.append(metric)
        return metric


class OfflineDioptraClient:
    """Records experiment, job, and metric calls in memory."""

    def __init__(self, source: str = "offline-client"):
        self.source = source
        self.warnings: list[str] = []
        self._experiments: list[ExperimentRecord] = []
        self.experiments = _ExperimentCollection(self)
        self.jobs = _JobCollection(self)

    def get_experiment(self, name: str) -> ExperimentRecord:
        for experiment in self._experiments:
            if experiment.name == name:
                return experiment
        raise KeyError(f"no experiment named {name!r}")

    def get_job(self, local_id: str) -> JobRecord:
        for experiment in self._experiments:
            for job in experiment.jobs:
                if job.local_id == local_id:
                    return job
        raise KeyError(f"no job with local_id {local_id!r}")

    def to_bundle(self) -> PilotBundle:
        if len(self._experiments) != 1:
            raise ValueError("pilot bundle expects exactly one experiment")
        experiment = self._experiments[0]
        return PilotBundle(
            source=self.source,
            notes=PILOT_NOTE,
            warnings=list(self.warnings),
            experiment=experiment,
            rest_examples=_rest_examples(experiment),
        )


def _rest_examples(experiment: ExperimentRecord) -> list[RestExample]:
    examples = [
        RestExample(
            method="POST",
            path="/api/v1/experiments",
            body={
                "group_id": None,
                "name": experiment.name,
                "description": experiment.description,
                "entrypoints": [],
            },
            blocked_reason=(
                "Dioptra requires an integer group_id and registered entrypoint "
                "ids from a live deployment. This pilot leaves both unset."
            ),
        )
    ]
    for job in experiment.jobs:
        for metric in job.metrics:
            body: dict = {"name": metric.name, "value": metric.value, "step": metric.step}
            if metric.timestamp is not None:
                body["timestamp"] = metric.timestamp.isoformat()
            examples.append(
                RestExample(
                    method="POST",
                    path="/api/v1/jobs/{job_id}/metrics",
                    body=body,
                    job_local_id=job.local_id,
                    blocked_reason=(
                        "Dioptra assigns an integer job id when a worker queue "
                        "accepts the job. This pilot does not submit jobs."
                    ),
                )
            )
    return examples
