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
from llm_eval.models.context import strip_think_blocks

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


_THINK_MARKERS = (
    "</think>",
    "</thinking>",
    "</reasoning>",
    "<|end_of_thought|>",
    "<|end_think|>",
)


def end_of_thinking_marker(tokenizer) -> str:
    """Close tag from the chat template or tokenizer, otherwise ``</think>``."""
    blobs: list[str] = []
    template = getattr(tokenizer, "chat_template", None)
    if template:
        blobs.append(str(template))
    for attr in ("additional_special_tokens", "all_special_tokens"):
        values = getattr(tokenizer, attr, None) or []
        blobs.extend(str(value) for value in values)
    encoder = getattr(tokenizer, "added_tokens_encoder", None) or {}
    if isinstance(encoder, dict):
        blobs.extend(str(key) for key in encoder)
    haystack = "\n".join(blobs)
    lowered = haystack.lower()
    for marker in _THINK_MARKERS:
        index = lowered.find(marker.lower())
        if index >= 0:
            return haystack[index : index + len(marker)]
    return "</think>"


def _flatten_ids(raw) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    values = list(raw)
    if values and isinstance(values[0], (list, tuple)):
        values = list(values[0])
    return [int(value) for value in values]


def _encode_ids(tokenizer, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return _flatten_ids(ids)


def _token_list(tokens) -> list[int]:
    if tokens is None:
        return []
    if hasattr(tokens, "tolist"):
        return _flatten_ids(tokens.tolist())
    if hasattr(tokens, "data"):
        return _flatten_ids(tokens.data)
    return _flatten_ids(tokens)


def _token_count(tokens) -> int:
    shape = getattr(tokens, "shape", None)
    if shape:
        return int(shape[-1])
    return len(_token_list(tokens))


def _think_split(text: str, marker: str) -> tuple[bool, str]:
    """Whether ``marker`` closed the think block, and the text after it."""
    if not text or not marker:
        return False, ""
    index = text.lower().find(marker.lower())
    if index < 0:
        return False, ""
    return True, text[index + len(marker) :]


def _scored_after_thinking(raw_text: str, marker: str) -> str:
    """Answer text after the think block. The block itself is not scored."""
    if marker.lower() == "</think>":
        return answer_after_think(raw_text, prompt_opened_think=True)
    closed, after = _think_split(raw_text, marker)
    if closed:
        return strip_think_blocks(after)
    return ""


class HuggingFaceFolderModel(BaseModel):
    """Load a local transformers folder and generate text.

    Chat mode is used only when the tokenizer has a chat template. Otherwise
    the folder is treated as a base model (completions). GPT-2's 1,024-token
    window is the default context size. An eval prompt that does not fit its
    budget is refused. It is not left-truncated.
    """

    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        folder = params.get("folder") or params.get("base_url") or name
        self.folder = str(folder)
        self.max_context = int(params.get("max_context") or 1024)
        self.max_new_tokens = int(params.get("max_new_tokens") or 64)
        raw_budget = params.get("prompt_budget", None)
        self.prompt_budget = None if raw_budget in (None, "") else int(raw_budget)
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
        answer_cap = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or self.max_new_tokens)
        answer_cap = max(1, answer_cap)
        think_cap = self._resolved_think_cap(kwargs.get("thinking_max_tokens", self.thinking_max_tokens))
        use_thinking = think_cap > 0
        generation_tokens = answer_cap + (think_cap if use_thinking else 0)
        override = kwargs.get("prompt_budget", self.prompt_budget)
        allowed, residual = self._prompt_limit(generation_tokens, override)
        tokenizer = self._tokenizer
        use_chat = self._use_chat(tokenizer)
        prepared = self.prepare_prompt(prompt)
        opened_think = prompt_opened_think(prepared)
        token_ids = _encode_ids(tokenizer, prepared)
        if not token_ids:
            token_ids = _encode_ids(tokenizer, prepared or " ")
        if len(token_ids) > allowed:
            message = (
                f"Prompt is {len(token_ids)} tokens. The prompt budget is {allowed}. "
                f"max_context is {self.max_context}, the thinking budget is {think_cap}, "
                f"and the answer cap is {answer_cap}. "
                f"The context window leaves {max(0, residual)} tokens beside the reserved generation. "
                "The prompt was not truncated."
            )
            logger.error("%s", message)
            return self._response(
                text="",
                raw_text="",
                latency_ms=0.0,
                completion_tokens=0,
                answer_cap=answer_cap,
                think_cap=think_cap,
                use_chat=use_chat,
                thinking=_model_thinks(self.name),
                prompt_tokens=len(token_ids),
                prompt_budget=allowed,
                hit_token_cap=False,
                think_truncated=False,
                prompt_over_budget=True,
                prompt_budget_error=message,
            )
        if use_thinking:
            return self._generate_with_thinking(
                token_ids,
                answer_cap=answer_cap,
                think_cap=think_cap,
                use_chat=use_chat,
                prompt_budget=allowed,
            )
        new_tokens, raw_text, latency_ms = self._generate_new(token_ids, answer_cap)
        completion_tokens = _token_count(new_tokens)
        text = answer_after_think(raw_text, prompt_opened_think=opened_think)
        return self._response(
            text=text,
            raw_text=raw_text,
            latency_ms=latency_ms,
            completion_tokens=completion_tokens,
            answer_cap=answer_cap,
            think_cap=0,
            use_chat=use_chat,
            thinking=_model_thinks(self.name),
            prompt_tokens=len(token_ids),
            prompt_budget=allowed,
            hit_token_cap=completion_tokens >= answer_cap,
            think_truncated=False,
            prompt_over_budget=False,
            prompt_budget_error="",
        )

    def _resolved_think_cap(self, explicit) -> int:
        """Thinking models use their cap. Models with no thinking mode stay one-shot."""
        if not _model_thinks(self.name):
            return 0
        if explicit in (None, ""):
            return _metadata_thinking_cap(self.name)
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            return _metadata_thinking_cap(self.name)

    def _prompt_limit(self, generation_tokens: int, override) -> tuple[int, int]:
        residual = int(self.max_context) - int(generation_tokens)
        declared = override if override is not None else self.prompt_budget
        allowed = residual if declared is None else min(int(declared), residual)
        return max(0, allowed), residual

    def _generate_new(self, token_ids: list[int], max_new: int, eos_token_id=None):
        torch = self._torch
        tokenizer = self._tokenizer
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        attention = torch.ones_like(input_ids)
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "max_new_tokens": max(1, int(max_new)),
            "do_sample": False,
            "pad_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        start = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(**generate_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        new_tokens = output[0][input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return new_tokens, raw_text, latency_ms

    def _generate_with_thinking(
        self,
        token_ids: list[int],
        *,
        answer_cap: int,
        think_cap: int,
        use_chat: bool,
        prompt_budget: int,
    ) -> ModelResponse:
        """Think up to the thinking cap, then generate the answer under its own cap.

        If the think block is still open, append the model's end-of-thinking
        marker and a newline before the answer phase. ``hit_token_cap`` is true
        only when that answer phase fills the answer cap.
        """
        tokenizer = self._tokenizer
        marker = end_of_thinking_marker(tokenizer)
        marker_ids = _encode_ids(tokenizer, marker)
        phase1_eos = getattr(tokenizer, "eos_token_id", None)
        if len(marker_ids) == 1:
            if phase1_eos is None:
                phase1_eos = marker_ids[0]
            else:
                phase1_eos = [phase1_eos, marker_ids[0]]
        think_tokens, think_text, think_latency = self._generate_new(token_ids, think_cap, eos_token_id=phase1_eos)
        think_ids = _token_list(think_tokens)
        control = ""
        try:
            control = tokenizer.decode(think_tokens, skip_special_tokens=False)
        except TypeError:
            control = think_text
        closed, after = _think_split(control or think_text, marker)
        if not closed:
            closed, after = _think_split(think_text, marker)
        think_truncated = False
        answer_ids: list[int] = []
        answer_latency = 0.0
        if not closed:
            think_truncated = True
            prefix = list(token_ids) + think_ids + _encode_ids(tokenizer, marker + "\n")
            answer_tokens, answer_text, answer_latency = self._generate_new(prefix, answer_cap)
            answer_ids = _token_list(answer_tokens)
            raw_text = f"{think_text}{marker}\n{answer_text}"
        elif not after.strip():
            prefix = list(token_ids) + think_ids
            answer_tokens, answer_text, answer_latency = self._generate_new(prefix, answer_cap)
            answer_ids = _token_list(answer_tokens)
            raw_text = f"{think_text}{answer_text}"
        elif len(think_ids) >= think_cap:
            prefix = list(token_ids) + think_ids
            answer_tokens, answer_text, answer_latency = self._generate_new(prefix, answer_cap)
            answer_ids = _token_list(answer_tokens)
            raw_text = f"{think_text}{answer_text}"
        else:
            raw_text = think_text
        text = _scored_after_thinking(raw_text, marker)
        return self._response(
            text=text,
            raw_text=raw_text,
            latency_ms=think_latency + answer_latency,
            completion_tokens=len(answer_ids),
            answer_cap=answer_cap,
            think_cap=think_cap,
            use_chat=use_chat,
            thinking=True,
            prompt_tokens=len(token_ids),
            prompt_budget=prompt_budget,
            hit_token_cap=len(answer_ids) >= answer_cap,
            think_truncated=think_truncated,
            prompt_over_budget=False,
            prompt_budget_error="",
        )

    def _response(
        self,
        *,
        text: str,
        raw_text: str,
        latency_ms: float,
        completion_tokens: int,
        answer_cap: int,
        think_cap: int,
        use_chat: bool,
        thinking: bool,
        prompt_tokens: int,
        prompt_budget: int,
        hit_token_cap: bool,
        think_truncated: bool,
        prompt_over_budget: bool,
        prompt_budget_error: str,
    ) -> ModelResponse:
        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_used=completion_tokens,
            metadata={
                "model": self.name,
                "completion_tokens": completion_tokens,
                "max_context": self.max_context,
                "max_new_tokens": answer_cap,
                "prompt_budget": prompt_budget,
                "prompt_tokens": prompt_tokens,
                "thinking_max_tokens": think_cap or None,
                "mode": "chat" if use_chat else "completions",
                "device": self.device_name,
                "precision": self.precision_used,
                "trust_remote_code": self.trust_remote_code,
                "loader": self.loader_kind,
                "raw_text": raw_text,
                "thinking_default": thinking,
                "mamba_path": self.mamba_path,
                "hit_token_cap": hit_token_cap,
                "think_truncated": think_truncated,
                "prompt_over_budget": prompt_over_budget,
                "prompt_budget_error": prompt_budget_error,
            },
        )

    def _use_chat(self, tokenizer) -> bool:
        if self.use_chat_template is True:
            return bool(getattr(tokenizer, "chat_template", None))
        if self.use_chat_template is False:
            return False
        return self.mode == "chat" or (self.mode == "auto" and getattr(tokenizer, "chat_template", None))

    def prepare_prompt(self, prompt: str) -> str:
        """Apply the chat template when this folder uses one. Otherwise return the prompt."""
        tokenizer = self._tokenizer
        if tokenizer is not None and self._use_chat(tokenizer) and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def count_prompt_tokens(self, prompt: str) -> int:
        """Token count of the prompt the way ``generate`` will send it."""
        self._load()
        prepared = self.prepare_prompt(prompt)
        return len(self._tokenizer.encode(prepared or " "))


def _model_thinks(name: str) -> bool:
    try:
        from dow_bench.meta import thinking_default
    except ImportError:
        return False
    return thinking_default(name)


def _metadata_thinking_cap(name: str) -> int:
    try:
        from dow_bench.meta import thinking_cap
    except ImportError:
        return 0
    return thinking_cap(name)
