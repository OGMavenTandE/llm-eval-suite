from pathlib import Path

from llm_eval.storage.runs import RunDirectory

EXECUTIVE_SUMMARY_FILENAME = "executive-summary.json"
REPORT_MANIFEST_FILENAME = "report-manifest.json"

ARTIFACT_LABELS = {
    "executive_summary": "Executive summary",
    "report_manifest": "Report manifest",
    "audit": "Run audit record",
    "summary": "Score summary",
    "detailed": "Sample-by-sample results",
    "comparison_summary": "Comparison summary",
    "comparison_detailed": "Comparison details",
}

CONTENT_TYPES = {
    "executive_summary": "application/json",
    "report_manifest": "application/json",
    "audit": "application/json",
    "summary": "text/csv",
    "detailed": "application/json",
    "comparison_summary": "text/csv",
    "comparison_detailed": "application/json",
}


def artifact_label_for_kind(kind: str) -> str:
    return ARTIFACT_LABELS.get(kind, kind.replace("_", " ").title())


def content_type_for_kind(kind: str) -> str:
    return CONTENT_TYPES.get(kind, "application/octet-stream")


def standard_artifact_names() -> dict[str, str]:
    """Return canonical artifact filenames used across runs."""
    return {
        "summary": RunDirectory.SUMMARY_FILENAME,
        "detailed": RunDirectory.DETAILED_FILENAME,
        "comparison_summary": RunDirectory.COMPARISON_SUMMARY_FILENAME,
        "comparison_detailed": RunDirectory.COMPARISON_DETAILED_FILENAME,
        "metadata": RunDirectory.METADATA_FILENAME,
        "executive_summary": EXECUTIVE_SUMMARY_FILENAME,
        "report_manifest": REPORT_MANIFEST_FILENAME,
    }


def find_model_artifacts(run_dir: str | Path) -> dict[str, Path | None]:
    """Locate per-model summary and detailed outputs under a run directory."""
    base = Path(run_dir)
    summary = base / RunDirectory.SUMMARY_FILENAME
    detailed = base / RunDirectory.DETAILED_FILENAME
    return {
        "run_dir": base,
        "summary_path": summary if summary.exists() else None,
        "detailed_path": detailed if detailed.exists() else None,
    }


def find_comparison_artifacts(comparison_dir: str | Path) -> dict[str, Path | None]:
    """Locate comparison outputs under a comparison directory."""
    base = Path(comparison_dir)
    summary = base / RunDirectory.COMPARISON_SUMMARY_FILENAME
    detailed = base / RunDirectory.COMPARISON_DETAILED_FILENAME
    return {
        "comparison_dir": base,
        "comparison_summary_path": summary if summary.exists() else None,
        "comparison_detailed_path": detailed if detailed.exists() else None,
    }
