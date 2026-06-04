import json
from pathlib import Path

from llm_eval.storage.artifacts import find_comparison_artifacts, find_model_artifacts


class ResultService:
    """Load summaries and detailed results from generated output files."""

    def load_summary(self, summary_path: str | Path) -> list[dict]:
        """Load a results_summary.csv file as a list of row dicts."""
        import csv

        path = Path(summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")

        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def load_detailed(self, detailed_path: str | Path) -> list[dict]:
        """Load a results_detailed.json file."""
        path = Path(detailed_path)
        if not path.exists():
            raise FileNotFoundError(f"Detailed results file not found: {path}")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}, got {type(data).__name__}.")
        return data

    def load_comparison_summary(self, comparison_summary_path: str | Path) -> list[dict]:
        """Load a comparison_summary.csv file."""
        import csv

        path = Path(comparison_summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Comparison summary not found: {path}")

        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def load_comparison_detailed(self, comparison_detailed_path: str | Path) -> list[dict]:
        """Load a comparison_detailed.json file."""
        path = Path(comparison_detailed_path)
        if not path.exists():
            raise FileNotFoundError(f"Comparison detailed file not found: {path}")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}, got {type(data).__name__}.")
        return data

    def resolve_model_artifacts(self, run_dir: str | Path) -> dict:
        """Return artifact paths for a model run directory."""
        return find_model_artifacts(run_dir)

    def resolve_comparison_artifacts(self, comparison_dir: str | Path) -> dict:
        """Return artifact paths for a comparison directory."""
        return find_comparison_artifacts(comparison_dir)
