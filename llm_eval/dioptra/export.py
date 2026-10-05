"""Map a suite audit record onto the offline Dioptra client."""

import csv
import json
from pathlib import Path

from llm_eval.dioptra.client import OfflineDioptraClient
from llm_eval.dioptra.schema import PilotBundle
from llm_eval.schemas.run_result import AuditMetadata

ENTRYPOINT_NAME = "yaml-eval"


def metrics_from_summary_csv(path: Path) -> list[tuple[str, float]]:
    """Read evaluator rows from results_summary.csv.

    Each row becomes two Dioptra-shaped metrics: ``{name}_mean_score`` and
    ``{name}_pass_rate``. Step stays 0 because the suite stores one aggregate
    per evaluator, not a time series.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"metric_name", "mean_score", "pass_rate"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing summary columns: {', '.join(sorted(missing))}"
            )
        metrics: list[tuple[str, float]] = []
        for row in reader:
            metric_name = (row.get("metric_name") or "").strip()
            if not metric_name:
                continue
            metrics.append((f"{metric_name}_mean_score", float(row["mean_score"])))
            metrics.append((f"{metric_name}_pass_rate", float(row["pass_rate"])))
        return metrics


def _artifact_paths_for_model(audit: AuditMetadata, model_name: str) -> list[str]:
    artifacts = audit.artifact_paths
    if artifacts is None:
        return []
    paths: list[str] = []
    for item in artifacts.model_artifacts:
        if item.model_name != model_name:
            continue
        for candidate in (item.summary_path, item.detailed_path, item.run_dir):
            if candidate:
                paths.append(candidate)
    return paths


def _summary_path_for_model(audit: AuditMetadata, model_name: str) -> str | None:
    artifacts = audit.artifact_paths
    if artifacts is None:
        return None
    for item in artifacts.model_artifacts:
        if item.model_name == model_name and item.summary_path:
            return item.summary_path
    return None


def record_audit(audit: AuditMetadata, client: OfflineDioptraClient | None = None) -> OfflineDioptraClient:
    """Record one experiment and one job per model from a suite audit."""
    client = client or OfflineDioptraClient(source="audit")
    description = f"llm-eval-suite run {audit.run_id}"
    if audit.config_path:
        description = f"{description}. Config: {audit.config_path}."
    client.experiments.create(
        name=audit.run_name,
        description=description,
        entrypoints=[ENTRYPOINT_NAME],
        created_at=audit.started_at,
    )

    model_names = list(audit.model_names) or [audit.run_name]
    providers = list(audit.model_providers)
    for index, model_name in enumerate(model_names):
        provider = providers[index] if index < len(providers) else ""
        summary_path = _summary_path_for_model(audit, model_name)
        artifact_paths = _artifact_paths_for_model(audit, model_name)
        metrics: list[tuple[str, float]] = []
        metrics_source = "none"
        if summary_path:
            summary_file = Path(summary_path)
            if summary_file.is_file():
                metrics = metrics_from_summary_csv(summary_file)
                metrics_source = "results_summary.csv"
            else:
                client.warnings.append(f"Summary path missing: {summary_path}")

        local_id = f"job-{index + 1}"
        client.jobs.create(
            audit.run_name,
            name=model_name,
            entrypoint_name=ENTRYPOINT_NAME,
            suite_status=audit.status,
            local_id=local_id,
            metrics_source=metrics_source,
            artifact_paths=artifact_paths,
            parameters={
                "dataset": audit.dataset_path,
                "dataset_sample_count": audit.dataset_sample_count,
                "model": model_name,
                "provider": provider,
                "evaluators": list(audit.evaluator_names),
                "config_hash": audit.config_hash,
                "config_path": audit.config_path,
                "dry_run": audit.dry_run,
                "compare": audit.compare,
            },
        )
        for metric_name, metric_value in metrics:
            client.jobs.append_metric(local_id, metric_name, metric_value, metric_step=0)
    return client


def export_audit_file(audit_path: str | Path) -> PilotBundle:
    path = Path(audit_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    audit = AuditMetadata.model_validate(payload)
    return record_audit(audit).to_bundle()


def write_bundle(bundle: PilotBundle, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
