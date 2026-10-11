"""Data-parallel fine-tuning of a Geneformer-style transformer on rank-value encoded cells
(Sec. 4.3b, row "Rules from omics data": Geneformer on the dissociated atlases).

Hugging Face `Trainer` wraps the model in DistributedDataParallel automatically when it is
launched with torchrun, so the same script runs on 1, 4 or 8 GPUs (or 2 CPU processes in CI):

    torchrun --standalone --nproc_per_node=4 omics/finetune_ddp.py --model-name ctheodoris/Geneformer --data cells.pt
    torchrun --standalone --nproc_per_node=2 omics/finetune_ddp.py --model-name tiny --n-cells 2000 --epochs 1

Input format (`--data`): a torch file with "input_ids" (N, L) of gene tokens ordered by
descending expression (Geneformer's rank-value encoding), "attention_mask" (N, L) and
"labels" (N,) — the cell-state / polarisation labels whose markers we want to learn.
Without `--data`, a synthetic rank-value dataset with three cell states is generated, and
`--model-name tiny` builds a 2-layer BERT from scratch so nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

N_GENES = 2000           # synthetic vocabulary size (real Geneformer: ~25k genes)
SEQ_LEN = 128            # top-L genes kept per cell (real Geneformer: 2048 / 4096)
PAD_ID, CLS_ID, GENE_OFFSET = 0, 1, 2
N_CLASSES = 3


class RankValueCells(Dataset):
    """Synthetic cells: expression ~ lognormal, class-specific marker genes up-weighted,
    genes ranked by expression, top-L gene tokens kept (Geneformer encoding)."""

    def __init__(self, n_cells: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        labels = torch.arange(n_cells) % N_CLASSES
        markers = torch.randperm(N_GENES, generator=g)[: N_CLASSES * 40].view(N_CLASSES, 40)
        expr = torch.randn(n_cells, N_GENES, generator=g).exp()
        for c in range(N_CLASSES):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            expr[idx[:, None], markers[c][None, :]] *= 3.0
        ranked = expr.argsort(dim=1, descending=True)[:, :SEQ_LEN] + GENE_OFFSET
        self.input_ids = torch.cat([torch.full((n_cells, 1), CLS_ID), ranked], dim=1)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = labels

    @classmethod
    def from_file(cls, path: str) -> "RankValueCells":
        d = torch.load(path, map_location="cpu")
        obj = cls.__new__(cls)
        obj.input_ids, obj.attention_mask, obj.labels = d["input_ids"], d["attention_mask"], d["labels"]
        return obj

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i], "labels": self.labels[i]}


def build_model(name: str, n_labels: int):
    from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification

    if name == "tiny":
        cfg = BertConfig(vocab_size=N_GENES + GENE_OFFSET, hidden_size=128, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=256,
                         max_position_embeddings=SEQ_LEN + 2, num_labels=n_labels, pad_token_id=PAD_ID)
        return BertForSequenceClassification(cfg)
    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_labels)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="tiny", help="'tiny' or a Hugging Face id, e.g. ctheodoris/Geneformer")
    p.add_argument("--data", default=None, help="torch file with input_ids / attention_mask / labels")
    p.add_argument("--n-cells", type=int, default=6000, help="synthetic cells when --data is not given")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=32, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=None, help="default 5e-5 (pretrained) or 1e-3 (tiny, from scratch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/omics_ddp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import Trainer, TrainingArguments

    ds = RankValueCells.from_file(args.data) if args.data else RankValueCells(args.n_cells, args.seed)
    n_val = max(1, len(ds) // 10)
    val_ds, train_ds = torch.utils.data.Subset(ds, range(n_val)), torch.utils.data.Subset(ds, range(n_val, len(ds)))
    n_labels = int(ds.labels.max().item()) + 1
    model = build_model(args.model_name, n_labels)
    if args.lr is None:
        args.lr = 1e-3 if args.model_name == "tiny" else 5e-5

    cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=args.seed,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        use_cpu=not cuda,
        ddp_backend="nccl" if cuda else "gloo",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    def accuracy(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((logits.argmax(-1) == labels).mean())}

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=accuracy)
    if targs.process_index == 0:
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
        print(f"[omics] world_size={targs.world_size} device={dev} train={len(train_ds)} val={len(val_ds)} "
              f"model={args.model_name} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    t0 = time.time()
    train_out = trainer.train()
    eval_out = trainer.evaluate()
    if targs.process_index == 0:
        metrics = {
            "world_size": targs.world_size,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "model": args.model_name,
            "n_train": len(train_ds),
            "epochs": args.epochs,
            "wall_time_s": round(time.time() - t0, 2),
            "train_samples_per_s": round(train_out.metrics.get("train_samples_per_second", 0.0), 2),
            "train_loss": round(train_out.metrics.get("train_loss", 0.0), 4),
            "eval_accuracy": round(eval_out.get("eval_accuracy", 0.0), 4),
        }
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("[omics] " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
