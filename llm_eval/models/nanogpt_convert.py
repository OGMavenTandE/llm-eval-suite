"""Detect a nanoGPT ``ckpt.pt`` and convert it to a Hugging Face GPT-2 folder.

nanoGPT stores attention and MLP projections as ``nn.Linear`` weights
``(out, in)``. Hugging Face GPT-2 uses Conv1D weights ``(in, out)``, so
``attn.c_attn``, ``attn.c_proj``, ``mlp.c_fc``, and ``mlp.c_proj`` are
transposed. A ``torch.compile`` prefix ``_orig_mod.`` is stripped.

The GPT-2 tokenizer is not downloaded. Copy ``vocab.json``, ``merges.txt``,
and ``tokenizer_config.json`` from a local GPT-2 tokenizer into the output
folder before loading it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CONV1D_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

# Fixed prompt. Token ids are GPT-2 BPE for "The capital of France is."
# The check compares both forwards on these ids and does not download a tokenizer.
FIXED_PROMPT = "The capital of France is."
FIXED_PROMPT_TOKEN_IDS = (464, 3139, 286, 4881, 318, 13)
LOGITS_TOLERANCE = 1e-4


def _torch_load(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def looks_like_nanogpt_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.is_file() and file_path.suffix == ".pt"


def describe_model_path(path: str | Path) -> dict:
    """Sniff a folder or file the connect screen can offer an action for."""
    target = Path(path)
    if target.is_dir() and (target / "config.json").is_file():
        return {
            "kind": "hf",
            "convertible": False,
            "message": "This is a Hugging Face folder. Connect it with the Hugging Face folder option.",
        }
    if target.is_file() and target.suffix == ".gguf":
        return {
            "kind": "gguf",
            "convertible": False,
            "message": (
                "This is a GGUF file. Serve it with Ollama or a llama.cpp server, "
                "then connect with Ollama or an OpenAI-compatible URL."
            ),
        }
    if looks_like_nanogpt_file(target):
        confirmed = _checkpoint_has_model_args(target)
        if confirmed is False:
            return {
                "kind": "unknown",
                "convertible": False,
                "message": "This .pt file does not look like a nanoGPT checkpoint (no model_args).",
            }
        note = "Convert to Hugging Face folder"
        if confirmed is None:
            note = (
                "This looks like a PyTorch checkpoint. "
                "Install the hf extra to confirm it and convert it."
            )
        return {
            "kind": "nanogpt",
            "convertible": True,
            "message": (
                f"{note}. nanoGPT checkpoints are converted to a GPT-2 folder "
                "(Conv1D weights transposed, _orig_mod. removed). "
                "Copy a local GPT-2 tokenizer into the output folder afterwards."
            ),
        }
    return {
        "kind": "unknown",
        "convertible": False,
        "message": "Could not tell what this path is. Expected a Hugging Face folder, a .gguf file, or ckpt.pt.",
    }


def _checkpoint_has_model_args(path: Path) -> bool | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        checkpoint = _torch_load(torch, path)
    except Exception:
        return False
    return isinstance(checkpoint, dict) and "model_args" in checkpoint and "model" in checkpoint


def should_transpose(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in CONV1D_SUFFIXES)


def strip_compile_prefix(key: str) -> str:
    return key.replace("_orig_mod.", "")


def remap_state_dict(state: dict) -> dict:
    """Strip compile prefixes and transpose Conv1D weights. Does not need a real model."""
    remapped = {}
    for raw_key, value in state.items():
        key = strip_compile_prefix(str(raw_key))
        if should_transpose(key):
            value = _transpose(value)
        remapped[key] = value
    if "lm_head.weight" not in remapped and "transformer.wte.weight" in remapped:
        remapped["lm_head.weight"] = remapped["transformer.wte.weight"]
    return remapped


def _transpose(value):
    if hasattr(value, "transpose") and getattr(value, "ndim", None) == 2:
        return value.transpose(0, 1)
    if hasattr(value, "t") and not isinstance(value, (str, bytes)):
        try:
            return value.t()
        except Exception:
            return value
    return value


def gpt2_config_from_model_args(model_args: dict) -> dict:
    n_positions = int(model_args.get("block_size") or model_args.get("n_positions") or 1024)
    dropout = float(model_args.get("dropout") or 0.0)
    vocab = int(model_args.get("vocab_size") or 50257)
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_layer": int(model_args["n_layer"]),
        "n_head": int(model_args["n_head"]),
        "n_embd": int(model_args["n_embd"]),
        "n_positions": n_positions,
        "n_ctx": n_positions,
        "vocab_size": vocab,
        "bos_token_id": vocab - 1 if vocab < 50257 else 50256,
        "eos_token_id": vocab - 1 if vocab < 50257 else 50256,
        "attn_pdrop": dropout,
        "embd_pdrop": dropout,
        "resid_pdrop": dropout,
        "layer_norm_epsilon": 1e-5,
        "tie_word_embeddings": True,
        "activation_function": "gelu",
        "transformers_version": "4.0.0",
    }


def logits_check_passed(max_abs_diff: float, tolerance: float = LOGITS_TOLERANCE) -> bool:
    """True when the converted checkpoint matches the source within ``tolerance``."""
    return float(max_abs_diff) <= float(tolerance)


def prompt_token_ids_for_vocab(vocab_size: int) -> list[int]:
    """Fixed prompt ids, shortened so a tiny test vocab still has a stable sequence."""
    ids = [int(token) for token in FIXED_PROMPT_TOKEN_IDS]
    if vocab_size > max(ids):
        return ids
    width = max(1, min(5, vocab_size - 1))
    return list(range(1, width + 1))


def build_conversion_report(
    *,
    source: str,
    output: str,
    config: dict,
    transposed_keys: list[str],
    logits_check: dict,
) -> dict:
    return {
        "source": source,
        "output": output,
        "activation_function": config.get("activation_function"),
        "n_ctx": config.get("n_ctx"),
        "n_positions": config.get("n_positions"),
        "transposed_keys": list(transposed_keys),
        "logits_check": logits_check,
    }


def write_conversion_report(destination: str | Path, report: dict) -> Path:
    path = Path(destination)
    if path.is_dir():
        path = path / "conversion_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def convert_nanogpt_to_hf(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    """Convert a nanoGPT checkpoint to a Hugging Face GPT-2 folder.

    Requires PyTorch. The tokenizer files are not fetched from the network.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'Converting a nanoGPT checkpoint needs the optional hf extra: pip install -e ".[hf]"'
        ) from exc

    source = Path(ckpt_path)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = _torch_load(torch, source)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "model_args" not in checkpoint:
        raise ValueError(
            "Expected a nanoGPT checkpoint with 'model' and 'model_args'. "
            "This converter does not change other checkpoint formats."
        )
    raw_state = checkpoint["model"]
    state = remap_state_dict(raw_state)
    config = gpt2_config_from_model_args(checkpoint["model_args"])
    transposed = sorted(key for key in state if should_transpose(key))
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save(state, destination / "pytorch_model.bin")
    note = (
        "Tokenizer files were not downloaded. Copy vocab.json, merges.txt, and "
        "tokenizer_config.json from a local GPT-2 tokenizer into this folder "
        "before loading the model.\n"
    )
    (destination / "TOKENIZER.txt").write_text(note, encoding="utf-8")
    token_ids = prompt_token_ids_for_vocab(int(config["vocab_size"]))
    try:
        max_abs = max_logit_diff(
            raw_state,
            state,
            config,
            token_ids,
            torch_module=torch,
        )
    except Exception as exc:
        logits_check = {
            "ran": True,
            "passed": False,
            "max_abs_diff": None,
            "tolerance": LOGITS_TOLERANCE,
            "prompt": FIXED_PROMPT,
            "prompt_token_ids": token_ids,
            "error": str(exc),
        }
        report = build_conversion_report(
            source=str(source),
            output=str(destination),
            config=config,
            transposed_keys=transposed,
            logits_check=logits_check,
        )
        write_conversion_report(destination, report)
        raise RuntimeError(
            f"Logits check failed ({exc}). See {destination / 'conversion_report.json'}."
        ) from exc
    passed = logits_check_passed(max_abs, LOGITS_TOLERANCE)
    logits_check = {
        "ran": True,
        "passed": passed,
        "max_abs_diff": max_abs,
        "tolerance": LOGITS_TOLERANCE,
        "prompt": FIXED_PROMPT,
        "prompt_token_ids": token_ids,
    }
    report = build_conversion_report(
        source=str(source),
        output=str(destination),
        config=config,
        transposed_keys=transposed,
        logits_check=logits_check,
    )
    write_conversion_report(destination, report)
    if not passed:
        raise RuntimeError(
            "Logits check failed: max abs diff "
            f"{max_abs:.3e} exceeds tolerance {LOGITS_TOLERANCE:.0e}. "
            f"See {destination / 'conversion_report.json'}."
        )
    return destination


def _linear_state(state: dict) -> dict:
    """Strip compile prefixes and tie lm_head. Do not transpose."""
    cleaned = {strip_compile_prefix(str(key)): value for key, value in state.items()}
    if "lm_head.weight" not in cleaned and "transformer.wte.weight" in cleaned:
        cleaned["lm_head.weight"] = cleaned["transformer.wte.weight"]
    return cleaned


def _gelu_exact(tensor):
    import torch

    return 0.5 * tensor * (1.0 + torch.erf(tensor / math.sqrt(2.0)))


def _gelu_new(tensor):
    import torch

    inner = math.sqrt(2.0 / math.pi) * (tensor + 0.044715 * torch.pow(tensor, 3.0))
    return 0.5 * tensor * (1.0 + torch.tanh(inner))


def _activation(name: str, tensor):
    if name == "gelu":
        return _gelu_exact(tensor)
    if name in {"gelu_new", "gelu_pytorch_tanh"}:
        return _gelu_new(tensor)
    raise ValueError(f"Unsupported activation_function {name!r}. Expected 'gelu'.")


def _apply(tensor, weight, bias, layout: str):
    import torch

    if layout == "conv1d":
        weight = weight.transpose(0, 1)
    return torch.nn.functional.linear(tensor, weight, bias)


def _gpt2_logits(state: dict, config: dict, token_ids: list[int], *, layout: str):
    """One GPT-2 forward. ``linear`` is nanoGPT. ``conv1d`` is the Hugging Face layout."""
    import torch

    activation_name = "gelu" if layout == "linear" else str(config.get("activation_function") or "gelu")
    n_head = int(config["n_head"])
    n_layer = int(config["n_layer"])
    eps = float(config.get("layer_norm_epsilon") or 1e-5)
    ids = torch.tensor([token_ids], dtype=torch.long)
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    hidden = state["transformer.wte.weight"][ids] + state["transformer.wpe.weight"][positions]
    for layer in range(n_layer):
        prefix = f"transformer.h.{layer}"
        normed = torch.nn.functional.layer_norm(
            hidden,
            (hidden.shape[-1],),
            state[f"{prefix}.ln_1.weight"],
            state[f"{prefix}.ln_1.bias"],
            eps,
        )
        qkv = _apply(
            normed,
            state[f"{prefix}.attn.c_attn.weight"],
            state[f"{prefix}.attn.c_attn.bias"],
            layout,
        )
        query, key, value = qkv.chunk(3, dim=-1)
        batch, steps, channels = query.shape
        head = channels // n_head
        def _heads(tensor):
            return tensor.view(batch, steps, n_head, head).transpose(1, 2)

        query, key, value = _heads(query), _heads(key), _heads(value)
        scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head))
        causal = torch.tril(torch.ones(steps, steps, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, channels)
        attended = _apply(
            mixed,
            state[f"{prefix}.attn.c_proj.weight"],
            state[f"{prefix}.attn.c_proj.bias"],
            layout,
        )
        hidden = hidden + attended
        normed = torch.nn.functional.layer_norm(
            hidden,
            (hidden.shape[-1],),
            state[f"{prefix}.ln_2.weight"],
            state[f"{prefix}.ln_2.bias"],
            eps,
        )
        fed = _apply(
            normed,
            state[f"{prefix}.mlp.c_fc.weight"],
            state[f"{prefix}.mlp.c_fc.bias"],
            layout,
        )
        fed = _activation(activation_name, fed)
        fed = _apply(
            fed,
            state[f"{prefix}.mlp.c_proj.weight"],
            state[f"{prefix}.mlp.c_proj.bias"],
            layout,
        )
        hidden = hidden + fed
    hidden = torch.nn.functional.layer_norm(
        hidden,
        (hidden.shape[-1],),
        state["transformer.ln_f.weight"],
        state["transformer.ln_f.bias"],
        eps,
    )
    return torch.nn.functional.linear(hidden, state["lm_head.weight"], None)


def max_logit_diff(source_state: dict, converted_state: dict, config: dict, token_ids: list[int], torch_module=None) -> float:
    """Compare nanoGPT linear logits with converted Conv1D logits on one prompt."""
    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    linear = _linear_state(source_state)
    converted = _linear_state(converted_state)
    with torch.no_grad():
        left = _gpt2_logits(linear, config, token_ids, layout="linear")
        right = _gpt2_logits(converted, config, token_ids, layout="conv1d")
        return float((left - right).abs().max().item())
