import hashlib
import json
import sys
from pathlib import Path

from llm_eval.datasets.loader import load_dataset
from llm_eval.schemas.dataset_info import DatasetInfo
from llm_eval.schemas.model_info import ModelInfo
from llm_eval.schemas.run_request import RunRequest, RunValidationResult

REQUIRED_CONFIG_KEYS = ("models", "evaluators", "dataset")
SUPPORTED_PROVIDERS = ("ollama", "openai")
SUPPORTED_EVALUATORS = ("correctness", "latency", "robustness", "consistency", "cost")


class ConfigService:
    """Config loading, normalization, and validation helpers."""

    def load_yaml(self, path: str) -> dict:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is not installed. Install it with: pip install pyyaml"
            ) from exc

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML config must be a mapping (got {type(data).__name__}).")

        return data

    def normalize_config(
        self,
        config: dict,
        *,
        output_dir: str | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        compare: bool = False,
        config_path: str | None = None,
    ) -> RunRequest:
        """Convert a raw config dict into a typed RunRequest."""
        models = [
            ModelInfo(
                name=m["name"],
                provider=m["provider"],
                params=m.get("params", {}),
            )
            for m in config.get("models", [])
        ]

        return RunRequest(
            run_name=config.get("run_name", "eval_run"),
            output_dir=output_dir or config.get("output_dir", "results/"),
            dataset=config["dataset"],
            models=models,
            evaluators=list(config.get("evaluators", [])),
            dry_run=dry_run,
            verbose=verbose,
            compare=compare,
            config_path=config_path,
        )

    def validate_config_structure(self, config: dict) -> list[str]:
        """Return structural validation errors for a raw config dict."""
        errors: list[str] = []

        if not isinstance(config, dict):
            errors.append(f"Config must be a mapping (got {type(config).__name__}).")
            return errors

        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
        if missing:
            errors.append(f"Config is missing required key(s): {', '.join(missing)}")

        models = config.get("models", [])
        if not isinstance(models, list) or not models:
            errors.append("'models' must be a non-empty list.")

        for i, model in enumerate(models if isinstance(models, list) else []):
            if not isinstance(model, dict):
                errors.append(f"Model entry {i + 1} must be a mapping.")
                continue
            if "name" not in model:
                errors.append(f"Model entry {i + 1} is missing 'name'.")
            if "provider" not in model:
                errors.append(f"Model entry {i + 1} is missing 'provider'.")
            elif model["provider"] not in SUPPORTED_PROVIDERS:
                errors.append(
                    f"Model entry {i + 1} has unknown provider '{model['provider']}'. "
                    f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
                )

        evaluators = config.get("evaluators", [])
        if not isinstance(evaluators, list) or not evaluators:
            errors.append("'evaluators' must be a non-empty list.")

        for i, ev in enumerate(evaluators if isinstance(evaluators, list) else []):
            if not isinstance(ev, dict):
                errors.append(f"Evaluator entry {i + 1} must be a mapping.")
                continue
            name = ev.get("name")
            if not name:
                errors.append(f"Evaluator entry {i + 1} is missing 'name'.")
            elif name not in SUPPORTED_EVALUATORS:
                errors.append(
                    f"Evaluator entry {i + 1} has unknown name '{name}'. "
                    f"Supported: {', '.join(SUPPORTED_EVALUATORS)}."
                )

        dataset = config.get("dataset")
        if not dataset or not isinstance(dataset, str):
            errors.append("'dataset' must be a non-empty string path.")

        return errors

    def validate_run(self, config: dict) -> RunValidationResult:
        """Validate config structure and dataset without running inference."""
        errors = self.validate_config_structure(config)
        warnings: list[str] = []
        dataset_info: DatasetInfo | None = None
        models: list[ModelInfo] = []
        evaluator_names: list[str] = []

        if not errors:
            try:
                samples = load_dataset(config["dataset"])
                dataset_path = Path(config["dataset"])
                dataset_info = DatasetInfo(
                    path=config["dataset"],
                    format=dataset_path.suffix.lstrip(".").lower() or "unknown",
                    sample_count=len(samples),
                )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))

            models = [
                ModelInfo(
                    name=m["name"],
                    provider=m["provider"],
                    params=m.get("params", {}),
                )
                for m in config.get("models", [])
            ]
            evaluator_names = [
                ev["name"] for ev in config.get("evaluators", []) if ev.get("name")
            ]

        valid = len(errors) == 0
        message = None
        if valid:
            n = dataset_info.sample_count if dataset_info else 0
            message = (
                f"Config valid. {n} samples, {len(models)} model(s), "
                f"{len(evaluator_names)} evaluator(s)."
            )

        return RunValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            dataset=dataset_info,
            models=models,
            evaluator_names=evaluator_names,
            message=message,
        )

    @staticmethod
    def config_hash(config: dict) -> str:
        """Stable hash of a config dict for audit trails."""
        payload = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def exit_on_errors(errors: list[str]) -> None:
        """Print errors to stderr and exit with code 1 (CLI helper)."""
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        if errors:
            sys.exit(1)
