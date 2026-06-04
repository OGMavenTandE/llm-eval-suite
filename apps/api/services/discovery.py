import requests

from apps.api.dependencies import AppSettings, get_config_service
from llm_eval.core.config_service import ConfigService


def discover_profiles(settings: AppSettings, config_service: ConfigService | None = None) -> list[dict]:
    config_service = config_service or get_config_service()
    profiles: list[dict] = []

    if not settings.config_dir.exists():
        return profiles

    for path in sorted(settings.config_dir.glob("*.yaml")) + sorted(settings.config_dir.glob("*.yml")):
        profile_id = path.stem
        item = {
            "profile_id": profile_id,
            "name": profile_id.replace("_", " ").replace("-", " ").title(),
            "description": None,
            "path": str(path),
            "evaluators": [],
            "model_names": [],
            "available": True,
            "valid": True,
            "validation_message": None,
        }
        try:
            config = config_service.load_yaml(str(path))
            validation = config_service.validate_run(config)
            item["evaluators"] = validation.evaluator_names
            item["model_names"] = [m.name for m in validation.models]
            item["valid"] = validation.valid
            item["validation_message"] = "; ".join(validation.errors) if validation.errors else None
        except Exception as exc:
            item["valid"] = False
            item["validation_message"] = str(exc)
        profiles.append(item)
    return profiles


def discover_datasets(settings: AppSettings) -> list[dict]:
    from llm_eval.datasets.loader import load_dataset

    datasets: list[dict] = []

    if not settings.datasets_dir.exists():
        return datasets

    for path in sorted(settings.datasets_dir.glob("*.jsonl")) + sorted(settings.datasets_dir.glob("*.csv")):
        dataset_id = path.stem
        item = {
            "dataset_id": dataset_id,
            "name": dataset_id.replace("_", " ").replace("-", " ").title(),
            "path": str(path),
            "format": path.suffix.lstrip(".").lower(),
            "sample_count": None,
            "valid": True,
            "validation_message": None,
        }
        try:
            rows = load_dataset(str(path))
            item["sample_count"] = len(rows)
        except Exception as exc:
            item["valid"] = False
            item["validation_message"] = str(exc)
        datasets.append(item)
    return datasets


def _probe_ollama(base_url: str = "http://localhost:11434", timeout: float = 1.0) -> tuple[bool, str, set[str]]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        names = {model.get("name", "") for model in data.get("models", []) if model.get("name")}
        return True, "reachable", names
    except requests.exceptions.ConnectionError:
        return False, "unreachable", set()
    except requests.exceptions.Timeout:
        return False, "timeout", set()
    except Exception:
        return False, "error", set()


def discover_models(settings: AppSettings, config_service: ConfigService | None = None) -> tuple[list[dict], list[str]]:
    config_service = config_service or get_config_service()
    seen: dict[tuple[str, str], dict] = {}
    providers: set[str] = set()

    for profile in discover_profiles(settings, config_service):
        try:
            config = config_service.load_yaml(profile["path"])
        except Exception:
            continue
        for model in config.get("models", []):
            provider = model.get("provider", "unknown")
            name = model.get("name", "unknown")
            providers.add(provider)
            key = (provider, name)
            if key not in seen:
                seen[key] = {
                    "provider": provider,
                    "name": name,
                    "available": False,
                    "connection_status": "unknown",
                    "warning_message": None,
                    "source_profile": profile["profile_id"],
                }

    ollama_ok, ollama_status, ollama_models = _probe_ollama()
    for (provider, name), item in seen.items():
        if provider == "ollama":
            item["connection_status"] = ollama_status
            if ollama_ok and name in ollama_models:
                item["available"] = True
            elif ollama_ok:
                item["available"] = False
                item["warning_message"] = f"Model '{name}' was not reported by the local Ollama instance."
            else:
                item["available"] = False
                item["warning_message"] = "Ollama is not reachable at http://localhost:11434."
        elif provider == "openai":
            item["connection_status"] = "configured"
            item["warning_message"] = "OpenAI availability is not probed automatically in offline mode."

    models = sorted(seen.values(), key=lambda item: (item["provider"], item["name"]))
    return models, sorted(providers)


def build_system_status(settings: AppSettings) -> dict:
    profiles = discover_profiles(settings)
    datasets = discover_datasets(settings)
    models, providers = discover_models(settings)
    warnings: list[str] = []

    if not settings.output_dir.exists():
        warnings.append(f"Output directory does not exist yet: {settings.output_dir}")
    if not profiles:
        warnings.append(f"No profiles found in {settings.config_dir}.")
    if not datasets:
        warnings.append(f"No datasets found in {settings.datasets_dir}.")
    if "ollama" in providers and not _probe_ollama()[0]:
        warnings.append("Ollama provider is referenced but not reachable locally.")

    return {
        "app_version": __import__("llm_eval").__version__,
        "output_dir": str(settings.output_dir),
        "run_index_path": str(settings.run_index_path),
        "profiles_count": len(profiles),
        "datasets_count": len(datasets),
        "model_providers": providers,
        "models_count": len(models),
        "warnings": warnings,
    }
