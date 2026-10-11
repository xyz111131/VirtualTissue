"""One spatial section per GPU: a PyTorch Geometric graph-attention model on a cell–cell
neighbourhood graph (Sec. 4.3b, row "Rules from omics data": NicheCompass / CellNEST on
the Xenium and Visium sections).

NicheCompass and CellNEST both train one PyG model per section on a single GPU, so the
parallelism across the ~10 sections and the hyper-parameter grid is a Slurm job array:

    sbatch --array=0-7 slurm/job_array_single_gpu.sbatch omics/niche_job_array.py --n-sections 8

Each array task takes the sections dealt to it by `huangtests.shard_indices`, builds the
k-nearest-neighbour graph, trains a 2-layer GAT to predict niche labels, and writes
`<out>/niche_task<k>.json`. The synthetic sections stand in for the real ones; swap
`make_section` for an AnnData loader and the rest is unchanged. This is also the ARM-port
check for the PyG stack named as the first Q1 task in Sec. 4.5b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from huangtests import shard_indices  # noqa: E402


def make_section(seed: int, n_cells: int, n_genes: int = 50, n_niches: int = 3, k: int = 10):
    """Cells in the unit square, three spatial niches, niche-specific marker genes, kNN graph."""
    from torch_geometric.data import Data

    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n_cells, 2, generator=g)
    centres = torch.rand(n_niches, 2, generator=g)
    niche = torch.cdist(pos, centres).argmin(dim=1)
    markers = torch.randn(n_niches, n_genes, generator=g) * 0.8
    x = torch.randn(n_cells, n_genes, generator=g) + markers[niche]
    # kNN edges (both directions). cdist is fine at section scale; use FAISS/torch-cluster for >1e5 cells.
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    nbr = d.topk(k, largest=False).indices                      # (n_cells, k)
    src = torch.arange(n_cells).repeat_interleave(k)
    edge_index = torch.stack([torch.cat([src, nbr.reshape(-1)]), torch.cat([nbr.reshape(-1), src])])
    perm = torch.randperm(n_cells, generator=g)
    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[perm[: int(0.7 * n_cells)]] = True
    return Data(x=x, edge_index=edge_index, y=niche, pos=pos, train_mask=train_mask, test_mask=~train_mask)


class GAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, heads: int = 4):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_dim, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, n_classes, heads=1)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        return self.conv2(F.dropout(h, p=0.2, training=self.training), edge_index)


def train_section(section_id: int, args: argparse.Namespace, device: torch.device) -> dict:
    data = make_section(seed=section_id, n_cells=args.n_cells).to(device)
    model = GAT(data.num_node_features, 32, int(data.y.max()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return {"section": section_id, "n_cells": int(data.num_nodes), "n_edges": int(data.num_edges),
            "test_accuracy": round(acc, 4), "train_time_s": round(time.time() - t0, 2)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-sections", type=int, default=8, help="sections 0..n-1 are dealt across array tasks")
    p.add_argument("--n-cells", type=int, default=4000, help="cells per synthetic section")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out", default="runs/niche_array")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    mine, shard, n_shards = shard_indices(args.n_sections)
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"[niche] task {shard}/{n_shards} on {dev_name}: sections {mine}", flush=True)

    results = [train_section(s, args, device) for s in mine]
    for r in results:
        print(f"[niche]   section {r['section']}: {r['n_cells']} cells, {r['n_edges']} edges, "
              f"test acc {r['test_accuracy']:.3f}, {r['train_time_s']}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"niche_task{shard}.json"), "w") as f:
        json.dump({"task": shard, "n_tasks": n_shards, "device": dev_name, "sections": results}, f, indent=2)


if __name__ == "__main__":
    main()
