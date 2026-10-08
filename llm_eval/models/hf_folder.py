"""In-process Hugging Face folder adapter.

``torch`` and ``transformers`` are optional. Install them with
``pip install -e ".[hf]"``. Tests that need those packages must
``pytest.importorskip`` them.

Stage-1 Department of War models need transformers 5.5 or newer
(gemma4 >= 5.5.0, nemotron_h >= 5.3.0, lfm2 >= 5.0). Loading one of
those folders fails with a clear message when the installed
transformers is older. ``trust_remote_code`` stays off unless the
connection opts in. No stage-1 model needs it. Phi-4-mini-instruct
and Nemotron ship ``auto_map`` Python files. Do not download them.

Gemma 4 E2B is ``Gemma4ForConditionalGeneration``. The loader reads
``config.architectures`` and uses the image-text-to-text class with
text-only input.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from llm_eval.offline import (
    apply_startup_offline,
    hub_allowed,
    hub_refused_message,
    is_hub_repo_id,
    missing_folder_message,
)

apply_startup_offline()

from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.models.context import strip_think_blocks, truncate_to_token_budget

logger = logging.getLogger("llm_eval.models.hf_folder")

STAGE1_MIN_TRANSFORMERS = (5, 5)
MULTIMODAL_ARCHITECTURES = {"Gemma4ForConditionalGeneration"}
STAGE1_MODEL_TYPES = {"gemma4", "nemotron_h", "lfm2"}
MAMBA_MODEL_TYPES = {"nemotron_h", "granitemoehybrid"}

_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN_TAG = re.compile(r"<think\b", re.IGNORECASE)


def torch_device_name(torch_module) -> str:
    """``cuda`` when this torch build can see a GPU, otherwise ``cpu``."""
    cuda = getattr(torch_module, "cuda", None)
    available = getattr(cuda, "is_available", None)
    if callable(available) and available():
        return "cuda"
    return "cpu"


def parse_transformers_version(text: str | None) -> tuple[int, int]:
    parts: list[int] = []
    for piece in str(text or "").split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


def transformers_too_old_message(found: str | None) -> str:
    seen = found or "unknown"
    return (
        f"This environment has transformers {seen}. "
        "Stage-1 models need transformers>=5.5 "
        "(gemma4 >=5.5.0, nemotron_h >=5.3.0, lfm2 >=5.0 for LFM2.5-2.6B). "
        "Load with trust_remote_code=False. No stage-1 model needs remote code. "
        "Phi-4-mini-instruct and Nemotron ship auto_map .py files. Do not download them."
    )


def _stage1_config(config: dict, name: str) -> bool:
    try:
        from dow_bench.meta import match_model
    except ImportError:
        match_model = None
    if match_model is not None and match_model(name):
        return True
    name_or_path = str(config.get("_name_or_path") or "")
    if match_model is not None and name_or_path and match_model(name_or_path):
        return True
    if str(config.get("model_type") or "") in STAGE1_MODEL_TYPES:
        return True
    architectures = config.get("architectures") or []
    return any(arch in MULTIMODAL_ARCHITECTURES for arch in architectures)


def assert_transformers_for_stage1(module, config: dict, name: str) -> None:
    """Fail closed when a stage-1 folder is loaded on transformers older than 5.5."""
    if not _stage1_config(config, name):
        return
    raw = getattr(module, "__version__", None)
    if raw is None or parse_transformers_version(str(raw)) < STAGE1_MIN_TRANSFORMERS:
        raise RuntimeError(transformers_too_old_message(None if raw is None else str(raw)))


def read_config(source: str | Path) -> dict:
    path = Path(source) / "config.json" if not str(source).endswith("config.json") else Path(source)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def select_model_class(transformers_module, config: dict):
    """Pick the loader class from ``config.architectures``.

    Gemma 4 E2B is multimodal. Text-only prompts still go through
    ``AutoModelForImageTextToText`` (or ``AutoModelForMultimodalLM``).
    """
    architectures = list(config.get("architectures") or [])
    if any(name in MULTIMODAL_ARCHITECTURES for name in architectures):
        cls = getattr(transformers_module, "AutoModelForImageTextToText", None)
        if cls is None:
            cls = getattr(transformers_module, "AutoModelForMultimodalLM", None)
        if cls is None:
            raise RuntimeError(
                "Gemma 4 E2B is Gemma4ForConditionalGeneration. "
                "Load it with AutoModelForImageTextToText on transformers>=5.5, "
                "not AutoModelForCausalLM. This transformers build has neither class."
            )
        return cls, "image-text-to-text"
    return transformers_module.AutoModelForCausalLM, "causal-lm"


def mamba_execution_path(model_type: str, name: str = "", *, platform: str | None = None) -> str | None:
    """Which hybrid-Mamba path will run. Never imports or requires mamba-ssm.

    The kernels are Linux-only. Windows, and a Linux box without the
    packages, uses the torch path.
    """
    hybrid = model_type in MAMBA_MODEL_TYPES
    if not hybrid and name:
        try:
            from dow_bench.meta import hybrid_mamba

            hybrid = hybrid_mamba(name)
        except ImportError:
            hybrid = False
    if not hybrid:
        return None
    plat = platform if platform is not None else sys.platform
    if plat == "win32":
        return "torch"
    ssm = importlib.util.find_spec("mamba_ssm")
    conv = importlib.util.find_spec("causal_conv1d")
    if ssm is not None and conv is not None:
        return "mamba-ssm"
    return "torch"


def prompt_opened_think(prompt: str) -> bool:
    """True when the chat template left an unclosed ``<think>`` on the prompt."""
    tail = (prompt or "")[-120:].lower()
    return "<think>" in tail and "</think>" not in tail


def answer_after_think(raw: str, *, prompt_opened_think: bool = False) -> str:
    """Scored text. Drops a closed think block and an unterminated one.

    When the template opened ``<think>`` in the prompt, the completion
    starts inside that block. Text through ``</think>`` is dropped. If
    the close tag never arrives, the completion is still thinking.
    """
    text = raw or ""
    if prompt_opened_think and not _THINK_OPEN_TAG.search(text):
        close = _THINK_CLOSE.search(text)
        if close is None:
            return ""
        text = text[close.end() :]
    return strip_think_blocks(text)


class HuggingFaceFolderModel(BaseModel):
    """Load a local transformers folder and generate text.

    Chat mode is used only when the tokenizer has a chat template. Otherwise
    the folder is treated as a base model (completions). GPT-2's 1,024-token
    window is the default cap: the prompt is truncated and new tokens are capped.
    """

    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        folder = params.get("folder") or params.get("base_url") or name
        self.folder = str(folder)
        self.max_context = int(params.get("max_context") or 1024)
        self.max_new_tokens = int(params.get("max_new_tokens") or 64)
        self.mode = params.get("mode") or "auto"
        self.hub = bool(params.get("hub"))
        self.precision = str(params.get("precision") or "").strip().lower()
        self.trust_remote_code = bool(params.get("trust_remote_code"))
        self.thinking_max_tokens = params.get("thinking_max_tokens")
        self.use_chat_template = params.get("use_chat_template")
        self.precision_used = ""
        self.loader_kind = ""
        self.mamba_path = ""
        self._tokenizer = None
        self._model = None
        self.device_name = "cpu"
        self._device = "cpu"

    def _resolve_source(self) -> tuple[str | Path, bool]:
        """Return the load path and whether it must stay on local files.

        A hub id or a missing folder is refused before ``transformers`` is
        imported, so offline mode names what would have been downloaded.
        An existing directory is checked for ``config.json`` after the import,
        which keeps the optional-extra error for a local folder.
        """
        path = Path(self.folder) if self.folder else None
        if path is not None and path.is_dir():
            return path, True
        source_name = str(self.name or self.folder or "")
        if self.hub and hub_allowed() and source_name:
            return source_name, False
        if not hub_allowed() and (self.hub or is_hub_repo_id(source_name)):
            raise RuntimeError(hub_refused_message(source_name))
        if not hub_allowed():
            raise RuntimeError(missing_folder_message(str(self.folder or source_name)))
        raise RuntimeError(f"Hugging Face folder not found: {path}")

    def _load(self):
        if self._model is not None:
            return
        source, local_only = self._resolve_source()
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Loading a Hugging Face folder needs the optional hf extra: "
                'pip install -e ".[hf]"'
            ) from exc
        if isinstance(source, Path) and not (source / "config.json").is_file():
            raise RuntimeError(
                f"{source} has no config.json. If this is a nanoGPT ckpt.pt, convert it first."
            )
        config = read_config(source) if isinstance(source, Path) else {}
        assert_transformers_for_stage1(transformers, config, self.name)
        if self.trust_remote_code and _stage1_config(config, self.name):
            logger.warning(
                "trust_remote_code is on for %s. No stage-1 model needs it. "
                "Phi-4-mini-instruct and Nemotron ship auto_map .py files that are not downloaded.",
                self.name,
            )
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.device_name = torch_device_name(torch)
        self._device = torch.device(self.device_name)
        load_kwargs = {
            "local_files_only": local_only,
            "trust_remote_code": self.trust_remote_code,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        model_cls, self.loader_kind = select_model_class(transformers, config)
        if self.precision == "8bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "8-bit loading needs bitsandbytes and BitsAndBytesConfig from transformers."
                ) from exc
            quant = BitsAndBytesConfig(load_in_8bit=True)
            self._model = model_cls.from_pretrained(
                source,
                quantization_config=quant,
                device_map="auto",
                **load_kwargs,
            )
            self.precision_used = "8bit"
        else:
            self._model = model_cls.from_pretrained(source, **load_kwargs)
            if self.precision == "bf16":
                self._model.to(self._device, dtype=torch.bfloat16)
                self.precision_used = "bf16"
            elif self.precision == "fp16":
                self._model.to(self._device, dtype=torch.float16)
                self.precision_used = "fp16"
            elif self.device_name == "cuda":
                self._model.to(self._device, dtype=torch.float16)
                self.precision_used = "fp16"
            else:
                self._model.to(self._device)
                self.precision_used = "fp32"
        self._model.eval()
        self._torch = torch
        model_type = str(config.get("model_type") or "")
        path_used = mamba_execution_path(model_type, self.name)
        self.mamba_path = path_used or ""
        if path_used:
            logger.info(
                "Hybrid Mamba weights for %s are using the %s path. mamba-ssm is not required.",
                self.name,
                path_used,
            )

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self._load()
        torch = self._torch
        answer_cap = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or self.max_new_tokens)
        answer_cap = max(1, answer_cap)
        think_cap = kwargs.get("thinking_max_tokens", self.thinking_max_tokens)
        try:
            think_cap = int(think_cap) if think_cap else 0
        except (TypeError, ValueError):
            think_cap = 0
        thinking = _model_thinks(self.name) or think_cap > 0
        if thinking and think_cap > 0:
            max_new = answer_cap + think_cap
        else:
            max_new = answer_cap
        max_new = max(1, min(max_new, self.max_context - 1))
        budget = max(1, self.max_context - max_new)
        tokenizer = self._tokenizer
        if self.use_chat_template is True:
            use_chat = bool(getattr(tokenizer, "chat_template", None))
        elif self.use_chat_template is False:
            use_chat = False
        else:
            use_chat = self.mode == "chat" or (
                self.mode == "auto" and getattr(tokenizer, "chat_template", None)
            )
        if use_chat and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        opened_think = prompt_opened_think(prompt)
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) > budget:
            token_ids = token_ids[-budget:]
        elif len(prompt.split()) > budget and not token_ids:
            prompt = truncate_to_token_budget(prompt, budget)
            token_ids = tokenizer.encode(prompt)
        if not token_ids:
            token_ids = tokenizer.encode(prompt or " ")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        attention = torch.ones_like(input_ids)
        start = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        latency_ms = (time.perf_counter() - start) * 1000
        new_tokens = output[0][input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        limited = _trim_thinking(raw_text, tokenizer, think_cap) if think_cap > 0 else raw_text
        text = answer_after_think(limited, prompt_opened_think=opened_think)
        completion_tokens = int(new_tokens.shape[-1])
        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_used=completion_tokens,
            metadata={
                "model": self.name,
                "completion_tokens": completion_tokens,
                "max_context": self.max_context,
                "max_new_tokens": answer_cap,
                "thinking_max_tokens": think_cap or None,
                "mode": "chat" if use_chat else "completions",
                "device": self.device_name,
                "precision": self.precision_used,
                "trust_remote_code": self.trust_remote_code,
                "loader": self.loader_kind,
                "raw_text": raw_text,
                "thinking_default": thinking,
                "mamba_path": self.mamba_path,
            },
        )


def _model_thinks(name: str) -> bool:
    try:
        from dow_bench.meta import thinking_default
    except ImportError:
        return False
    return thinking_default(name)


def _trim_thinking(text: str, tokenizer, cap: int) -> str:
    """Keep at most ``cap`` tokens inside a think block."""
    match = re.search(r"(?is)<think>(.*?)</think>", text)
    if not match:
        return text
    body = match.group(1)
    tokens = tokenizer.encode(body)
    if len(tokens) <= cap:
        return text
    trimmed = tokenizer.decode(tokens[:cap])
    return text[: match.start(1)] + trimmed + text[match.end(1) :]
