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
from pathlib import Path

CONV1D_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


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
        "transformers_version": "4.0.0",
    }


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
    state = remap_state_dict(checkpoint["model"])
    config = gpt2_config_from_model_args(checkpoint["model_args"])
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save(state, destination / "pytorch_model.bin")
    note = (
        "Tokenizer files were not downloaded. Copy vocab.json, merges.txt, and "
        "tokenizer_config.json from a local GPT-2 tokenizer into this folder "
        "before loading the model.\n"
    )
    (destination / "TOKENIZER.txt").write_text(note, encoding="utf-8")
    return destination
