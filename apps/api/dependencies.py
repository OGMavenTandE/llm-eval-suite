from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    """Local API configuration."""

    workspace_root: Path
    output_dir: Path
    config_dir: Path
    datasets_dir: Path

    @property
    def run_index_path(self) -> Path:
        return self.output_dir / ".llm_eval_runs.json"


_settings_override: AppSettings | None = None


def get_workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache
def _default_settings() -> AppSettings:
    root = get_workspace_root()
    return AppSettings(
        workspace_root=root,
        output_dir=root / "results",
        config_dir=root / "config",
        datasets_dir=root / "datasets",
    )


def get_settings() -> AppSettings:
    if _settings_override is not None:
        return _settings_override
    return _default_settings()


def override_settings(settings: AppSettings) -> None:
    global _settings_override
    _settings_override = settings


def reset_settings() -> None:
    global _settings_override
    _settings_override = None
    _default_settings.cache_clear()


def get_run_service():
    from llm_eval.core.config_service import ConfigService
    from llm_eval.core.run_service import RunService

    return RunService(config_service=ConfigService())


def get_config_service():
    from llm_eval.core.config_service import ConfigService

    return ConfigService()


_run_job_manager = None


def get_run_job_manager():
    from apps.api.services.run_jobs import RunJobManager

    global _run_job_manager
    if _run_job_manager is None:
        _run_job_manager = RunJobManager(get_run_service())
    return _run_job_manager


def reset_run_job_manager() -> None:
    global _run_job_manager
    _run_job_manager = None
