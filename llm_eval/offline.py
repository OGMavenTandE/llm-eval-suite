"""Offline switch for a closed network.

Set ``LLM_EVAL_OFFLINE=1``, save the settings flag, or mark a preset with
``offline: true``. A classified build sets ``AIRGAP_BUILD`` or
``LLM_EVAL_AIRGAP_BUILD=1``, which removes the Hugging Face hub branch.

Call :func:`apply_startup_offline` before importing ``transformers``,
``huggingface_hub``, or ``garak``. The Hugging Face environment variables are
read at import time, especially telemetry.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Classified builds set this True in the tree they ship. The environment
# variable does the same thing without editing this file.
AIRGAP_BUILD = False

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

OPENAI_BLOCKED = (
    "Offline mode blocks the OpenAI cloud API. "
    "Nothing was sent. Use a model on this computer."
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_BASE_FOLDER = "models/gpt2-medium"
DEFAULT_DETECTOR_ROOT = REPO_ROOT / "models" / "garak_detectors"

_settings_offline = False
_run_offline = False
_original_import = None
_env_before: dict[str, str | None] = {}
_guard_installed = False
_in_guard = False


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def airgap_build() -> bool:
    """True when this process is the classified build."""
    return bool(AIRGAP_BUILD) or _truthy(os.environ.get("LLM_EVAL_AIRGAP_BUILD"))


def offline_active() -> bool:
    """True when hub downloads and the OpenAI cloud API must be refused."""
    return (
        airgap_build()
        or _truthy(os.environ.get("LLM_EVAL_OFFLINE"))
        or _settings_offline
        or _run_offline
    )


def hub_allowed() -> bool:
    """False when a hub repo id must not be passed to ``from_pretrained``."""
    return not airgap_build() and not offline_active()


def is_local_endpoint(url: str | None) -> bool:
    if not url:
        return True
    host = (urlparse(url).hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "::1"}


def is_hub_repo_id(value: str | None) -> bool:
    """A hub id is ``name`` or ``org/name``. A relative folder such as ``models/gpt2-medium`` is not."""
    text = (value or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or text.startswith("."):
        return False
    if Path(text).is_dir():
        return False
    if "/" not in text:
        return True
    if text.count("/") != 1:
        return False
    org, name = text.split("/", 1)
    if not org or not name:
        return False
    # Configured weight folders live under these directory names.
    if org in {"models", "fixtures", "datasets", "config", "weights"}:
        return False
    return True


def hub_refused_message(model_id: str) -> str:
    name = (model_id or "").strip() or "unknown model"
    if airgap_build():
        return (
            f"This build disables Hugging Face hub downloads (LLM_EVAL_AIRGAP_BUILD). "
            f"Refused a download of {name}. "
            "Put the weights in a local folder that contains config.json. Nothing was downloaded."
        )
    return (
        f"Offline mode blocked a download of {name}. "
        "Put the weights in a local folder that contains config.json. Nothing was downloaded."
    )


def missing_folder_message(folder: str) -> str:
    place = folder or "the folder you configured"
    return (
        f"Offline mode refused to download weights. Local folder not found: {place}. "
        "Put config.json and the weight files in that folder. Nothing was downloaded."
    )


def refuse_remote_http(url: str | None, *, what: str = "network call") -> None:
    """Raise before any socket if offline mode forbids this URL."""
    if not offline_active() or is_local_endpoint(url):
        return
    target = url or what
    if what.lower().startswith("openai") or "api.openai.com" in (url or ""):
        raise RuntimeError(f"{OPENAI_BLOCKED} Refused {target}.")
    raise RuntimeError(
        f"Offline mode blocks a network call to {target} ({what}). Nothing was sent."
    )


def apply_offline_environment() -> None:
    """Set the Hugging Face offline variables. Safe to call more than once."""
    for key, value in OFFLINE_ENV.items():
        if key not in _env_before:
            _env_before[key] = os.environ.get(key)
        os.environ[key] = value


def apply_startup_offline() -> None:
    """Entry-point hook. Uses the environment and the air-gap build only.

    Settings and presets are applied later, once their files are known.
    This still runs before ``transformers``, ``huggingface_hub``, and ``garak``.
    """
    if airgap_build() or _truthy(os.environ.get("LLM_EVAL_OFFLINE")):
        apply_offline_environment()
        install_import_guard()


def settings_path(data_dir: str | Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / "settings.json"


def read_settings(data_dir: str | Path | None) -> dict:
    path = settings_path(data_dir)
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_settings(data_dir: str | Path | None) -> None:
    """Remember the saved offline flag for this process."""
    global _settings_offline
    _settings_offline = _truthy(read_settings(data_dir).get("offline"))
    if offline_active():
        apply_offline_environment()
        install_import_guard()


def write_settings_offline(data_dir: str | Path, enabled: bool) -> dict:
    """Save the settings flag. The environment and the air-gap build stay on."""
    global _settings_offline
    if not enabled and (airgap_build() or _truthy(os.environ.get("LLM_EVAL_OFFLINE"))):
        raise RuntimeError(
            "Offline mode is forced by LLM_EVAL_OFFLINE or the air-gap build. "
            "The OpenAI cloud API stays blocked."
        )
    path = settings_path(data_dir)
    if path is None:
        raise RuntimeError("No data directory for offline settings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = read_settings(data_dir)
    payload["offline"] = bool(enabled)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _settings_offline = bool(enabled)
    if offline_active():
        apply_offline_environment()
        install_import_guard()
    return offline_status(data_dir)


def activate_run_offline(
    *,
    preset: dict | None = None,
    config: dict | None = None,
    data_dir: str | Path | None = None,
) -> bool:
    """Turn the switch on for this run. Returns whether offline mode is on."""
    global _run_offline
    if data_dir is not None:
        load_settings(data_dir)
    if (preset or {}).get("offline") or (config or {}).get("offline"):
        _run_offline = True
    if offline_active():
        apply_offline_environment()
        install_import_guard()
        return True
    return False


def offline_forced_by() -> str | None:
    if airgap_build():
        return "airgap"
    if _truthy(os.environ.get("LLM_EVAL_OFFLINE")):
        return "env"
    return None


def offline_status(data_dir: str | Path | None = None) -> dict:
    if data_dir is not None:
        # Read the file without clearing a run flag that is already on.
        saved = _truthy(read_settings(data_dir).get("offline"))
        global _settings_offline
        _settings_offline = saved
    on = offline_active()
    forced = offline_forced_by()
    if on and forced == "airgap":
        message = (
            "Offline mode is on for this air-gap build. "
            "The OpenAI cloud API is blocked. Weight downloads are refused."
        )
    elif on and forced == "env":
        message = (
            "Offline mode is on (LLM_EVAL_OFFLINE). "
            "The OpenAI cloud API is blocked. Weight downloads are refused."
        )
    elif on:
        message = (
            "Offline mode is on. "
            "The OpenAI cloud API is blocked. Weight downloads are refused."
        )
    else:
        message = (
            "Offline mode is off. "
            "The OpenAI cloud API is blocked only after you turn offline mode on."
        )
    return {
        "offline": on,
        "forced": forced,
        "airgap_build": airgap_build(),
        "openai_cloud_blocked": on,
        "message": message,
    }


def demo_base_folder(configured: str | None = None) -> str:
    """Local gpt2-medium folder. ``LLM_EVAL_DEMO_BASE_FOLDER`` wins."""
    override = (os.environ.get("LLM_EVAL_DEMO_BASE_FOLDER") or "").strip()
    if override:
        return override
    text = (configured or "").strip()
    if not text or text == "gpt2-medium":
        return DEFAULT_DEMO_BASE_FOLDER
    return text


def demo_base_warning(folder: str | None = None) -> str:
    place = folder or DEFAULT_DEMO_BASE_FOLDER
    return (
        f"Model B is a local gpt2-medium folder. Put config.json and the weight files in {place}. "
        "Set LLM_EVAL_DEMO_BASE_FOLDER to use a different folder. This preset does not download weights."
    )


def offline_preset_note() -> str:
    return (
        "Offline mode is on for this preset. "
        "The OpenAI cloud API is blocked and weights are not downloaded."
    )


@dataclass(frozen=True)
class DetectorPin:
    """One garak detector that would otherwise download a hub model."""

    key: str
    garak_module: str
    garak_class: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    env_var: str

    def directory(self) -> Path:
        override = (os.environ.get(self.env_var) or "").strip()
        if override:
            return Path(override)
        root = (os.environ.get("LLM_EVAL_GARAK_DETECTORS_DIR") or "").strip()
        base = Path(root) if root else DEFAULT_DETECTOR_ROOT
        return base / self.key

    def ready(self) -> Path | None:
        folder = self.directory()
        if folder.is_dir() and (folder / "config.json").is_file():
            return folder
        return None

    def missing_message(self) -> str:
        files = ", ".join(self.files)
        place = self.directory()
        if offline_active():
            lead = (
                f"Offline mode skipped garak detector {self.garak_module}.{self.garak_class}"
            )
        else:
            lead = f"Skipped garak detector {self.garak_module}.{self.garak_class}"
        return (
            f"{lead} because {self.repo_id} at revision {self.revision} is not in {place}. "
            f"Copy these files into that folder: {files}. Nothing was downloaded."
        )


# Revisions are the Hub commit SHAs for these repos (checked October 8, 2026).
DETECTOR_PINS: tuple[DetectorPin, ...] = (
    DetectorPin(
        key="refutation_detector_distilbert",
        garak_module="misleading",
        garak_class="MustRefuteClaimModel",
        repo_id="garak-llm/refutation_detector_distilbert",
        revision="906ac60cba379abc1ad1ed328acebeb17357adac",
        files=(
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
        env_var="LLM_EVAL_GARAK_REFUTATION_DIR",
    ),
    DetectorPin(
        key="roberta_nli",
        garak_module="misleading",
        garak_class="MustContradictNLI",
        repo_id="garak-llm/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        revision="75044664e962c6237d48ec4d72fa189fb8723fdc",
        files=(
            "config.json",
            "merges.txt",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        env_var="LLM_EVAL_GARAK_NLI_DIR",
    ),
)


def detector_for_id(model_id: str) -> DetectorPin | None:
    text = (model_id or "").strip()
    if not text:
        return None
    for pin in DETECTOR_PINS:
        if text == pin.repo_id or text == pin.key:
            return pin
        try:
            if Path(text).resolve() == pin.directory().resolve():
                return pin
        except OSError:
            continue
    return None


def ready_local_dir(model_id: str) -> Path | None:
    text = str(model_id)
    path = Path(text)
    if path.is_dir() and (path / "config.json").is_file():
        return path
    pin = detector_for_id(text)
    if pin is None:
        return None
    return pin.ready()


def resolve_pretrained_source(model_id: str) -> tuple[str | None, str | None]:
    """Return ``(local_path, error)``. An error means do not touch the network."""
    text = str(model_id)
    local = ready_local_dir(text)
    if local is not None:
        return str(local), None
    pin = detector_for_id(text)
    if pin is not None:
        return None, pin.missing_message()
    if offline_active():
        if is_hub_repo_id(text):
            return None, hub_refused_message(text)
        return None, missing_folder_message(text)
    return None, None


def probes_use_hf_detectors(probes: list[str] | str | None) -> bool:
    if probes is None:
        return False
    if isinstance(probes, str):
        text = probes.strip().lower()
    else:
        text = ",".join(str(part) for part in probes).lower()
    if not text:
        return False
    return text == "all" or "misleading" in text


def detector_skip_notes(probes: list[str] | str | None = None) -> list[str]:
    """Visible notes for detector weights that are not on disk.

    Pass ``probes`` to limit the note to a run that would load them.
    ``None`` lists every missing pin.
    """
    if probes is not None and not probes_use_hf_detectors(probes):
        return []
    notes = []
    for pin in DETECTOR_PINS:
        if pin.ready() is None:
            notes.append(pin.missing_message())
    return notes


def garak_detector_config() -> dict:
    """``plugins.detectors`` block. A ready folder is pinned. A missing one is skipped."""
    grouped: dict[str, dict] = {}
    for pin in DETECTOR_PINS:
        ready = pin.ready()
        if ready is not None:
            entry = {
                "detector_model_path": str(ready),
                "hf_args": {"local_files_only": True},
            }
        else:
            entry = {"skip": True}
        grouped.setdefault(pin.garak_module, {})[pin.garak_class] = entry
    return grouped


def garak_child_env(env: dict, work_dir: str | Path) -> dict:
    """Environment for a garak subprocess, including the import guard."""
    child = dict(env)
    if offline_active():
        child.update(OFFLINE_ENV)
    hook = Path(work_dir) / "_llm_eval_offline_hook"
    hook.mkdir(parents=True, exist_ok=True)
    (hook / "sitecustomize.py").write_text(
        "try:\n"
        "    import llm_eval.offline as _llm_eval_offline\n"
        "    _llm_eval_offline.install_import_guard()\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )
    parts = [str(hook), str(REPO_ROOT)]
    prior = child.get("PYTHONPATH") or ""
    if prior:
        parts.append(prior)
    child["PYTHONPATH"] = os.pathsep.join(parts)
    return child


def install_import_guard() -> None:
    """Refuse hub downloads the moment ``transformers`` or ``huggingface_hub`` is used."""
    global _guard_installed, _original_import

    if not _guard_installed:
        _original_import = builtins.__import__

        def _guarded(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            module = _original_import(name, globals, locals, fromlist, level)
            # Never import from this hook. A nested import would call it again.
            if not _in_guard and _should_wrap(name):
                _enter_wrap(name)
            return module

        builtins.__import__ = _guarded
        _guard_installed = True
    for loaded in ("transformers", "huggingface_hub", "garak.detectors.base"):
        if loaded in sys.modules:
            _enter_wrap(loaded)


def _should_wrap(name: str) -> bool:
    return (
        name == "transformers"
        or name.startswith("transformers.")
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
        or name == "garak.detectors.base"
    )


def _enter_wrap(name: str) -> None:
    global _in_guard
    _in_guard = True
    try:
        _wrap_imported(name)
    finally:
        _in_guard = False


def _wrap_imported(name: str) -> None:
    module = sys.modules.get(name)
    if module is None:
        return
    if name == "transformers":
        _wrap_transformers(module)
    elif name == "huggingface_hub" or name.startswith("huggingface_hub."):
        _wrap_huggingface_hub(sys.modules.get("huggingface_hub", module))
    elif name == "garak.detectors.base":
        _patch_hf_detector(module)


def _wrap_transformers(module) -> None:
    for class_name in (
        "AutoConfig",
        "AutoTokenizer",
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForSequenceClassification",
    ):
        cls = getattr(module, class_name, None)
        if cls is None or not hasattr(cls, "from_pretrained"):
            continue
        current = cls.from_pretrained
        if getattr(current, "_llm_eval_guard", False):
            continue
        original = current

        def _wrapped(model_id, *args, _original=original, **kwargs):
            local, error = resolve_pretrained_source(str(model_id))
            if error:
                raise RuntimeError(error)
            if local is not None:
                kwargs = dict(kwargs)
                kwargs["local_files_only"] = True
                return _original(local, *args, **kwargs)
            return _original(model_id, *args, **kwargs)

        _wrapped._llm_eval_guard = True
        setattr(cls, "from_pretrained", staticmethod(_wrapped))


def _wrap_huggingface_hub(module) -> None:
    for name in ("hf_hub_download", "snapshot_download"):
        original = getattr(module, name, None)
        if original is None or getattr(original, "_llm_eval_guard", False):
            continue

        def _wrapped(*args, _original=original, **kwargs):
            repo = kwargs.get("repo_id")
            if repo is None and args:
                repo = args[0]
            text = str(repo or "")
            _local, error = resolve_pretrained_source(text)
            if error:
                raise RuntimeError(error)
            pin = detector_for_id(text)
            if pin is not None and pin.ready() is not None:
                raise RuntimeError(
                    f"Refused a hub download of {pin.repo_id}. "
                    f"Load the local folder {pin.ready()} instead. Nothing was downloaded."
                )
            if offline_active():
                raise RuntimeError(hub_refused_message(text or "unknown model"))
            return _original(*args, **kwargs)

        _wrapped._llm_eval_guard = True
        setattr(module, name, _wrapped)


def _patch_hf_detector(module) -> None:
    """Skip a missing detector before ``from_pretrained`` can download it.

    ``Detector.__init__`` sets ``_instance_configured``. The original
    ``HFDetector.__init__`` calls that again, and the second load is a no-op,
    so a local path assigned in between is the path that gets loaded.
    """
    detector_cls = getattr(module, "HFDetector", None)
    base = getattr(module, "Detector", None)
    if detector_cls is None or base is None:
        return
    if getattr(detector_cls, "_llm_eval_patched", False):
        return
    original = detector_cls.__init__

    def __init__(self, config_root=None):
        if config_root is None:
            from garak import _config as config_root

        base.__init__(self, config_root)
        path = str(getattr(self, "detector_model_path", "") or "")
        local = ready_local_dir(path)
        if local is None and (detector_for_id(path) is not None or offline_active()):
            self.skip = True
            pin = detector_for_id(path)
            note = pin.missing_message() if pin is not None else hub_refused_message(path)
            self.offline_note = note
            print(note, flush=True)
            return
        if local is not None:
            self.detector_model_path = str(local)
        original(self, config_root)

    detector_cls.__init__ = __init__
    detector_cls._llm_eval_patched = True


def reset_for_tests() -> None:
    """Drop process flags and restore environment keys this module set."""
    global _settings_offline, _run_offline, _guard_installed, _original_import, _in_guard
    _settings_offline = False
    _run_offline = False
    _in_guard = False
    if _guard_installed and _original_import is not None:
        builtins.__import__ = _original_import
    _guard_installed = False
    _original_import = None
    for key, previous in _env_before.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    _env_before.clear()
