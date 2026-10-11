"""LoRA fine-tuning of an open-weight causal LM on sentence -> rule pairs, data-parallel
across GPUs (Sec. 4.3b, row "Rules from the literature", training step; Sec. 4.3c,
"Literature rules": LoRA on the curated rules of [1] plus their source sentences).

Each GPU holds the full (frozen) base model plus trainable LoRA adapters and sees a
disjoint shard of the pairs; Hugging Face Trainer runs DistributedDataParallel under
torchrun or `accelerate launch`:

    torchrun --standalone --nproc_per_node=4 llm/lora_finetune.py \
        --model-name <HF id of the 4-8B open-weight model chosen in Sec. 4.3c> \
        --data llm/data/rules_curated.jsonl --out runs/llm_lora
    accelerate launch --multi_gpu --num_processes 4 llm/lora_finetune.py ...   # equivalent

    torchrun --standalone --nproc_per_node=2 llm/lora_finetune.py --model-name tiny --epochs 2   # CPU / CI

A 30B-class model fits in one 288 GB Rubin GPU in bf16, so for the final pass the same
script runs with --model-name <30B id>; FSDP is not needed at this scale.

`--model-name tiny` builds a 2-layer Llama-architecture model and a word-level tokenizer
from the training data so the pipeline can be tested offline.

Data format: JSONL with "sentence" and "rule" ("cell type, signal, direction, behavior").
Output: <out>/adapter (LoRA weights) and, for tiny, <out>/model (merged) + vocab.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from common import WordTokenizer, load_tokenizer, make_prompt, read_jsonl  # noqa: E402


class RulePairs(torch.utils.data.Dataset):
    """Prompt tokens are masked out of the loss (-100); only the rule tokens are learned."""

    def __init__(self, rows: list[dict], tok, max_len: int = 128):
        self.items = []
        for r in rows:
            prompt = tok.encode(make_prompt(r["sentence"]), add_special_tokens=True)
            target = tok.encode(" " + r["rule"].strip(), add_special_tokens=False) + [tok.eos_token_id]
            ids = (prompt + target)[:max_len]
            labels = ([-100] * len(prompt) + target)[:max_len]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def make_collator(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


def build_model_and_tokenizer(name: str, rows: list[dict]):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    if name == "tiny":
        tok = WordTokenizer.build([make_prompt(r["sentence"]) + " " + r["rule"] for r in rows])
        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                          pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
        return LlamaForCausalLM(cfg), tok
    tok = load_tokenizer(name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model, tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="tiny")
    p.add_argument("--data", default=os.path.join(_HERE, "data", "rules_toy.jsonl"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=None, help="default 2e-4 (pretrained) or 5e-3 (tiny, from scratch)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/llm_lora")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    rows = read_jsonl(args.data)
    base, tok = build_model_and_tokenizer(args.model_name, rows)
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      # the tiny model starts from random weights, so its embeddings must train too;
                      # a pretrained base keeps them frozen (LoRA only)
                      modules_to_save=["embed_tokens", "lm_head"] if args.model_name == "tiny" else None)
    model = get_peft_model(base, lora)
    if args.lr is None:
        args.lr = 5e-3 if args.model_name == "tiny" else 2e-4
    ds = RulePairs(rows, tok)

    cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        use_cpu=not cuda,
        ddp_backend="nccl" if cuda else "gloo",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=make_collator(tok.pad_token_id))
    if targs.process_index == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] world_size={targs.world_size} device={torch.cuda.get_device_name(0) if cuda else 'cpu'} "
              f"pairs={len(ds)} trainable={trainable/1e6:.2f}M of {total/1e6:.1f}M params", flush=True)

    t0 = time.time()
    out = trainer.train()

    if targs.process_index == 0:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(os.path.join(args.out, "adapter"))
        if args.model_name == "tiny":                       # keep a loadable merged copy for extract_sharded.py
            merged = model.merge_and_unload()
            merged.save_pretrained(os.path.join(args.out, "model"))
            tok.save(os.path.join(args.out, "model", "vocab.json"))
        metrics = {"world_size": targs.world_size, "model": args.model_name, "n_pairs": len(ds),
                   "epochs": args.epochs, "wall_time_s": round(time.time() - t0, 2),
                   "train_loss": round(out.metrics.get("train_loss", 0.0), 4),
                   "train_samples_per_s": round(out.metrics.get("train_samples_per_second", 0.0), 2)}
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("[lora] " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
