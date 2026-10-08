"""In-process Hugging Face folder adapter.

``torch`` and ``transformers`` are optional. Install them with
``pip install -e ".[hf]"``. Tests that need those packages must
``pytest.importorskip`` them.
"""

from __future__ import annotations

import os
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


def torch_device_name(torch_module) -> str:
    """``cuda`` when this torch build can see a GPU, otherwise ``cpu``."""
    cuda = getattr(torch_module, "cuda", None)
    available = getattr(cuda, "is_available", None)
    if callable(available) and available():
        return "cuda"
    return "cpu"


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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Loading a Hugging Face folder needs the optional hf extra: "
                'pip install -e ".[hf]"'
            ) from exc
        if isinstance(source, Path) and not (source / "config.json").is_file():
            raise RuntimeError(
                f"{source} has no config.json. If this is a nanoGPT ckpt.pt, convert it first."
            )
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.device_name = torch_device_name(torch)
        self._device = torch.device(self.device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        self._model = AutoModelForCausalLM.from_pretrained(source, local_files_only=local_only)
        if self.device_name == "cuda":
            self._model.to(self._device, dtype=torch.float16)
        else:
            self._model.to(self._device)
        self._model.eval()
        self._torch = torch

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self._load()
        torch = self._torch
        max_new = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or self.max_new_tokens)
        max_new = max(1, min(max_new, self.max_context - 1))
        budget = max(1, self.max_context - max_new)
        tokenizer = self._tokenizer
        use_chat = self.mode == "chat" or (
            self.mode == "auto" and getattr(tokenizer, "chat_template", None)
        )
        if use_chat and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
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
        text = strip_think_blocks(tokenizer.decode(new_tokens, skip_special_tokens=True))
        completion_tokens = int(new_tokens.shape[-1])
        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_used=completion_tokens,
            metadata={
                "model": self.name,
                "completion_tokens": completion_tokens,
                "max_context": self.max_context,
                "mode": "chat" if use_chat else "completions",
                "device": self.device_name,
            },
        )
