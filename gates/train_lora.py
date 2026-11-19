"""Train a LoRA adapter for a classification workload on a local base model.

Needs ``torch``, ``transformers`` and ``peft`` (``pip install -e ".[hf,lora]"``)
and runs offline against a base-model folder on disk. It trains only on the
``train`` split, so the held-out ``test`` split stays untouched for Gate 5.

    python -m gates.train_lora --spec examples/message_triage/spec.yaml \\
        --items examples/message_triage/items.jsonl \\
        --base path/to/base-model --out runs/triage-lora --merge

Writes the adapter, a ``train_record.json`` (base model, data hash, settings,
loss by epoch), and with ``--merge`` a merged folder that ``gates predict
--provider hf`` can load directly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from gates.data import load_items
from gates.report import file_digest
from gates.spec import load_spec


def build_examples(spec, items, tokenizer, max_len: int):
    """Prompt tokens are masked from the loss; only the label and EOS are learned."""
    examples = []
    for item in items:
        prompt = spec.prompt_for(item["text"])
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(" " + item["label"] if not prompt.endswith((" ", "\n")) else item["label"],
                               add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
        prompt_ids = prompt_ids[-(max_len - len(answer_ids)):]
        examples.append({"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids})
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--base", required=True, help="Local base-model folder (config.json and tokenizer)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--merge", action="store_true", help="Also write a merged model folder")
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = load_spec(args.spec)
    train = load_items(args.items, split="train")
    if not train:
        raise SystemExit("no items with split 'train'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, torch_dtype=dtype).to(device)
    target = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=target, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable:,} of {total:,} parameters on {device}")

    examples = build_examples(spec, train, tokenizer, args.max_len)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    losses = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(examples), args.batch):
            batch = examples[start:start + args.batch]
            width = max(len(e["input_ids"]) for e in batch)
            ids = torch.full((len(batch), width), tokenizer.pad_token_id)
            labels = torch.full((len(batch), width), -100)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, e in enumerate(batch):
                n = len(e["input_ids"])
                ids[row, :n] = torch.tensor(e["input_ids"])
                labels[row, :n] = torch.tensor(e["labels"])
                mask[row, :n] = 1
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss += float(out.loss)
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
        print(f"[lora] epoch {epoch}/{args.epochs} loss {losses[-1]:.4f}")

    out = Path(args.out)
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    record = {
        "workload": spec.workload,
        "base_model_folder": Path(args.base).name,
        "base_config_sha256": file_digest(Path(args.base) / "config.json")["sha256"],
        "items": file_digest(args.items),
        "spec": file_digest(args.spec),
        "train_items": len(train),
        "settings": {k: v for k, v in vars(args).items() if k not in ("spec", "items", "base", "out")},
        "trainable_parameters": trainable,
        "loss_by_epoch": losses,
        "device": device,
    }
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged")
        tokenizer.save_pretrained(out / "merged")
        record["merged_folder"] = "merged"
    (out / "train_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[lora] wrote {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
