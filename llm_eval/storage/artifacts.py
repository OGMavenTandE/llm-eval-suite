from pathlib import Path

from llm_eval.storage.runs import RunDirectory


def standard_artifact_names() -> dict[str, str]:
    """Return canonical artifact filenames used across runs."""
    return {
        "summary": RunDirectory.SUMMARY_FILENAME,
        "detailed": RunDirectory.DETAILED_FILENAME,
        "comparison_summary": RunDirectory.COMPARISON_SUMMARY_FILENAME,
        "comparison_detailed": RunDirectory.COMPARISON_DETAILED_FILENAME,
        "metadata": RunDirectory.METADATA_FILENAME,
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
