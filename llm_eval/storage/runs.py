from datetime import datetime
from pathlib import Path
from uuid import uuid4


class RunDirectory:
    """Predictable layout for a single evaluation run."""

    SUMMARY_FILENAME = "results_summary.csv"
    DETAILED_FILENAME = "results_detailed.json"
    COMPARISON_SUMMARY_FILENAME = "comparison_summary.csv"
    COMPARISON_DETAILED_FILENAME = "comparison_detailed.json"
    METADATA_FILENAME = "run_metadata.json"

    def __init__(self, output_dir: str, run_name: str, run_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.run_id = run_id or uuid4().hex[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = self.output_dir / f"{run_name}_{timestamp}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def metadata_path(self) -> Path:
        return self.base_dir / self.METADATA_FILENAME

    def model_run_dir(self, model_name: str) -> Path:
        safe_name = model_name.replace(":", "_").replace("/", "_")
        return self.base_dir / safe_name

    def comparison_dir(self) -> Path:
        return self.base_dir / "comparison"


class RunStorage:
    """High-level helper for creating and resolving run directories."""

    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)

    def create_run_directory(self, run_name: str, run_id: str | None = None) -> RunDirectory:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return RunDirectory(str(self.output_dir), run_name, run_id=run_id)

    def resolve_output_dir(self, config_output_dir: str | None = None) -> Path:
        path = Path(config_output_dir or "results/")
        path.mkdir(parents=True, exist_ok=True)
        return path
