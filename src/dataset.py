"""
PyTorch Dataset for plan graphs.

Loads JSON plans, builds graphs, and returns batched Data objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.feature_encoder import OperatorVocabulary, node_features_to_tensor
from src.graph_builder import build_graph


class PlanGraphDataset(Dataset):
    """Dataset of query execution plans as graphs."""

    def __init__(
        self,
        plan_dir: str | Path,
        vocab: OperatorVocabulary | None = None,
        max_depth: int = 32,
        target: str = "execution_time",
    ):
        self.plan_dir = Path(plan_dir)
        self.max_depth = max_depth
        self.target_key = target
        self.vocab = vocab or OperatorVocabulary()

        self.files: list[Path] = []
        for f in sorted(self.plan_dir.glob("*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if "plan" in data:
                    self.files.append(f)
            except (json.JSONDecodeError, KeyError):
                pass

    def __len__(self) -> int:
        return len(self.files)

    def _load_one(self, idx: int) -> dict[str, Any] | None:
        path = self.files[idx]
        with open(path) as f:
            data = json.load(f)
        plan = data.get("plan")
        target_val = data.get(self.target_key)
        if plan is None or target_val is None:
            return None

        try:
            target_val = float(target_val)
        except (TypeError, ValueError):
            return None

        graph = build_graph(plan)
        nodes = graph["nodes"]
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)

        feats = node_features_to_tensor(nodes, self.vocab, self.max_depth)
        x_op = feats["operator_ids"]
        x_cont = feats["continuous"]
        depth = feats["depth"]

        # Cardinality = root's actual_rows (for multi-task)
        cardinality = nodes[0].get("actual_rows") if nodes else 0.0
        if cardinality is not None:
            cardinality = float(cardinality)
        else:
            cardinality = 0.0

        return {
            "x_op": x_op,
            "x_cont": x_cont,
            "depth": depth,
            "edge_index": edge_index,
            "batch_mask": torch.ones(len(nodes), dtype=torch.bool),
            "y": torch.tensor([target_val], dtype=torch.float32),
            "cardinality": torch.tensor([cardinality], dtype=torch.float32),
            "query_id": data.get("query_id", idx),
            "num_nodes": len(nodes),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        out = self._load_one(idx)
        if out is None:
            # Fallback to a dummy sample if loading fails
            out = {
                "x_op": torch.zeros(1, dtype=torch.long),
                "x_cont": torch.zeros(1, 5),
                "depth": torch.zeros(1, dtype=torch.long),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "batch_mask": torch.ones(1, dtype=torch.bool),
                "y": torch.tensor([0.0]),
                "cardinality": torch.tensor([0.0]),
                "query_id": idx,
                "num_nodes": 1,
            }
        return out


def collate_plan_graphs(batch: list[dict]) -> dict[str, Any]:
    """
    Collate multiple plan graphs into a single batched graph.
    Uses batch assignment for batching (node i belongs to batch[batch[i]]).
    """
    x_ops = []
    x_conts = []
    depths = []
    edge_indices = []
    ys = []
    cardinalities = []
    batch_assign = []
    offset = 0

    for i, sample in enumerate(batch):
        n = sample["num_nodes"]
        x_ops.append(sample["x_op"])
        x_conts.append(sample["x_cont"])
        depths.append(sample["depth"])
        ys.append(sample["y"])
        cardinalities.append(sample.get("cardinality", torch.tensor([0.0])))
        batch_assign.extend([i] * n)

        ei = sample["edge_index"]
        if ei.dim() == 2 and ei.shape[1] > 0:
            edge_indices.append(ei + offset)
        offset += n

    return {
        "x_op": torch.cat(x_ops, dim=0),
        "x_cont": torch.cat(x_conts, dim=0),
        "depth": torch.cat(depths, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2, 0, dtype=torch.long),
        "batch": torch.tensor(batch_assign, dtype=torch.long),
        "y": torch.cat(ys, dim=0),
        "cardinality": torch.cat(cardinalities, dim=0) if cardinalities else torch.zeros(len(batch)),
        "batch_size": len(batch),
    }


class PlanPairDataset(Dataset):
    """
    Wrapper that yields pairs (plan_i, plan_j) for ranking loss.
    Pairs are sampled from same query_id where runtime_i < runtime_j.
    """

    def __init__(self, base_dataset: PlanGraphDataset):
        self.base = base_dataset
        self._build_groups()

    def _build_groups(self) -> None:
        """Group indices by query_id."""
        self.groups: dict[int, list[int]] = {}
        for i in range(len(self.base)):
            sample = self.base[i]
            qid = sample.get("query_id", i)
            self.groups.setdefault(qid, []).append(i)

        # Pairs: (faster_idx, slower_idx) where y[faster] < y[slower]
        self.pairs: list[tuple[int, int]] = []
        for qid, indices in self.groups.items():
            if len(indices) < 2:
                continue
            for i in indices:
                for j in indices:
                    if i == j:
                        continue
                    yi = self.base[i]["y"].item()
                    yj = self.base[j]["y"].item()
                    if yi < yj:
                        self.pairs.append((i, j))

    def __len__(self) -> int:
        return len(self.pairs) if self.pairs else len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.pairs:
            i, j = self.pairs[idx % len(self.pairs)]
            si = self.base[i]
            sj = self.base[j]
            return {
                "plan_fast": si,
                "plan_slow": sj,
                "y_fast": si["y"],
                "y_slow": sj["y"],
            }
        s = self.base[idx]
        return {"plan_fast": s, "plan_slow": s, "y_fast": s["y"], "y_slow": s["y"]}


def collate_plan_pairs(batch: list[dict]) -> dict[str, Any]:
    """Collate pairs: concatenate fast and slow plans with batch indices."""
    fast_batches = [b["plan_fast"] for b in batch]
    slow_batches = [b["plan_slow"] for b in batch]
    fast_collated = collate_plan_graphs(fast_batches)
    slow_collated = collate_plan_graphs(slow_batches)
    return {
        "fast": fast_collated,
        "slow": slow_collated,
        "y_fast": torch.stack([b["y_fast"] for b in batch]).squeeze(-1),
        "y_slow": torch.stack([b["y_slow"] for b in batch]).squeeze(-1),
    }
