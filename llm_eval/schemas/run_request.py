from pydantic import BaseModel, Field

from llm_eval.schemas.dataset_info import DatasetInfo
from llm_eval.schemas.model_info import ModelInfo


class RunRequest(BaseModel):
    """Normalized evaluation run request."""

    run_name: str = "eval_run"
    output_dir: str = "results/"
    dataset: str
    models: list[ModelInfo]
    evaluators: list[dict]
    dry_run: bool = False
    verbose: bool = False
    compare: bool = False
    config_path: str | None = None


class RunValidationResult(BaseModel):
    """Outcome of config and dataset validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dataset: DatasetInfo | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    evaluator_names: list[str] = Field(default_factory=list)
    message: str | None = None
