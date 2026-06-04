from __future__ import annotations

import json
import logging
from pathlib import Path

from llm_eval.core.result_service import ResultService
from llm_eval.reporting.executive_summary import build_executive_summary
from llm_eval.schemas.executive_summary import ExecutiveSummary, ReportManifest, ReportManifestEntry
from llm_eval.schemas.run_result import RunArtifactPaths
from llm_eval.storage.artifacts import (
    EXECUTIVE_SUMMARY_FILENAME,
    REPORT_MANIFEST_FILENAME,
    artifact_label_for_kind,
    content_type_for_kind,
)

logger = logging.getLogger(__name__)


def reports_dir_for_run(output_dir: str, run_id: str) -> Path:
    return Path(output_dir) / "reports" / run_id


def load_executive_summary(path: str | Path) -> ExecutiveSummary:
    summary_path = Path(path)
    with summary_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return ExecutiveSummary.model_validate(data)


def load_report_manifest(path: str | Path) -> ReportManifest:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return ReportManifest.model_validate(data)


class ReportArtifactService:
    """Generate and persist reporting artifacts for completed runs."""

    def __init__(self, result_service: ResultService | None = None):
        self.result_service = result_service or ResultService()

    def generate_and_persist(
        self,
        *,
        run_id: str,
        run_name: str,
        dataset_path: str | None,
        model_names: list[str],
        evaluator_names: list[str],
        artifacts: RunArtifactPaths,
    ) -> RunArtifactPaths:
        model_results = self._load_model_results(artifacts)
        if not model_results:
            logger.info("Skipping executive summary for run %s: no model results available.", run_id)
            return artifacts

        summary = build_executive_summary(
            run_id=run_id,
            run_name=run_name,
            dataset_path=dataset_path,
            model_names=model_names,
            evaluator_names=evaluator_names,
            model_results=model_results,
        )

        reports_dir = reports_dir_for_run(artifacts.output_dir, run_id)
        reports_dir.mkdir(parents=True, exist_ok=True)

        executive_summary_path = reports_dir / EXECUTIVE_SUMMARY_FILENAME
        with executive_summary_path.open("w", encoding="utf-8") as handle:
            handle.write(summary.model_dump_json(indent=2))

        manifest = self._build_manifest(summary=summary, artifacts=artifacts, summary_path=executive_summary_path)
        manifest_path = reports_dir / REPORT_MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json(indent=2))

        return artifacts.model_copy(
            update={
                "reports_dir": str(reports_dir),
                "executive_summary_path": str(executive_summary_path),
                "report_manifest_path": str(manifest_path),
            }
        )

    def _load_model_results(self, artifacts: RunArtifactPaths) -> list[dict]:
        model_results: list[dict] = []
        for model_entry in artifacts.model_artifacts:
            payload = {
                "model_name": model_entry.model_name,
                "summary": None,
                "detailed": None,
            }
            if model_entry.summary_path:
                try:
                    payload["summary"] = self.result_service.load_summary(model_entry.summary_path)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    logger.warning("Could not load summary for %s: %s", model_entry.model_name, exc)
            if model_entry.detailed_path:
                try:
                    payload["detailed"] = self.result_service.load_detailed(model_entry.detailed_path)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    logger.warning("Could not load detailed results for %s: %s", model_entry.model_name, exc)
            if payload["summary"] or payload["detailed"]:
                model_results.append(payload)
        return model_results

    def _build_manifest(
        self,
        *,
        summary: ExecutiveSummary,
        artifacts: RunArtifactPaths,
        summary_path: Path,
    ) -> ReportManifest:
        entries: list[ReportManifestEntry] = [
            ReportManifestEntry(
                kind="executive_summary",
                path=str(summary_path),
                label=artifact_label_for_kind("executive_summary"),
                content_type=content_type_for_kind("executive_summary"),
            )
        ]

        for model_entry in artifacts.model_artifacts:
            for kind, path in (
                ("summary", model_entry.summary_path),
                ("detailed", model_entry.detailed_path),
            ):
                if not path:
                    continue
                label = artifact_label_for_kind(kind)
                if model_entry.model_name:
                    label = f"{label} ({model_entry.model_name})"
                entries.append(
                    ReportManifestEntry(
                        kind=kind,
                        path=path,
                        label=label,
                        content_type=content_type_for_kind(kind),
                    )
                )

        for kind, path in (
            ("comparison_summary", artifacts.comparison_summary_path),
            ("comparison_detailed", artifacts.comparison_detailed_path),
        ):
            if path:
                entries.append(
                    ReportManifestEntry(
                        kind=kind,
                        path=path,
                        label=artifact_label_for_kind(kind),
                        content_type=content_type_for_kind(kind),
                    )
                )

        return ReportManifest(
            run_id=summary.run_id,
            generated_at=summary.generated_at,
            artifacts=entries,
        )
