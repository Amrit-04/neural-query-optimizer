"""
Feature encoding for plan graph nodes.

- Operator type -> learnable embedding
- Numeric features (cost, rows, width, depth)
- Depth positional encoding
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Canonical operator types (extend as needed for your plans)
OPERATOR_TYPES = [
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "Nested Loop",
    "Merge Join",
    "Hash Join",
    "Aggregate",
    "Hash Aggregate",
    "Sort",
    "Limit",
    "Subquery Scan",
    "Materialize",
    "Unique",
    "Set Op",
    "Append",
    "ModifyTable",
    "Memoize",
    "Gather",
    "Gather Merge",
    "CTE Scan",
    "Unknown",
]


class OperatorVocabulary:
    """Maps operator type strings to integer IDs."""

    def __init__(self, operator_types: list[str] | None = None):
        self.types = operator_types or OPERATOR_TYPES
        self.type_to_id = {t: i for i, t in enumerate(self.types)}
        self.unk_id = self.type_to_id.get("Unknown", len(self.types) - 1)

    def __len__(self) -> int:
        return len(self.types)

    def get_id(self, node_type: str) -> int:
        return self.type_to_id.get(node_type, self.unk_id)


def encode_numeric(
    nodes: list[dict],
    max_depth: int = 32,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Encode numeric features for each node.
    Returns (node_features, actual_times).
    actual_times is per-node; root holds execution time.
    """
    vocab = OperatorVocabulary()
    feats = []
    actual_times = []

    for n in nodes:
        node_type = n.get("node_type", "Unknown")
        op_id = vocab.get_id(node_type)
        depth = min(int(n.get("depth", 0)), max_depth - 1)
        startup = np.log1p(max(0.0, float(n.get("startup_cost") or 0.0)))
        total_cost = np.log1p(max(0.0, float(n.get("total_cost") or 0.0)))
        plan_rows = np.log1p(max(0.0, float(n.get("plan_rows") or 0.0)))
        plan_width = np.log1p(max(0.0, float(n.get("plan_width") or 0.0)))
        actual_time = n.get("actual_time")

        feat = [
            op_id,
            depth,
            startup,
            total_cost,
            plan_rows,
            plan_width,
        ]
        feats.append(feat)
        actual_times.append(actual_time)

    X = np.array(feats, dtype=np.float32)
    at = np.array(actual_times, dtype=np.float32)
    return X, at


def node_features_to_tensor(
    nodes: list[dict],
    vocab: OperatorVocabulary,
    max_depth: int = 32,
    include_cost: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Build tensors for GNN input.
    Separates operator IDs (for embedding) from continuous features.
    Only planning-time features are used.
    If include_cost=True: continuous dimension is 4 (startup_cost, total_cost, plan_rows, plan_width).
    If include_cost=False: continuous dimension is 2 (plan_rows, plan_width).
    """
    op_ids = []
    cont_feats = []
    depths = []
    for n in nodes:
        op_ids.append(vocab.get_id(n.get("node_type", "Unknown")))
        depth = min(int(n.get("depth", 0)), max_depth - 1)
        depths.append(depth)
        plan_rows = np.log1p(max(0.0, float(n.get("plan_rows") or 0.0)))
        plan_width = np.log1p(max(0.0, float(n.get("plan_width") or 0.0)))
        if include_cost:
            startup = np.log1p(max(0.0, float(n.get("startup_cost") or 0.0)))
            total_cost = np.log1p(max(0.0, float(n.get("total_cost") or 0.0)))
            cont_feats.append([startup, total_cost, plan_rows, plan_width])
        else:
            cont_feats.append([plan_rows, plan_width])

    return {
        "operator_ids": torch.tensor(op_ids, dtype=torch.long),
        "continuous": torch.tensor(cont_feats, dtype=torch.float32),
        "depth": torch.tensor(depths, dtype=torch.long),
    }


def extract_flat_features(nodes: list[dict], vocab: OperatorVocabulary) -> np.ndarray:
    """
    Extract flat/tabular feature vector for baseline tree-based models (e.g. GBDT).
    """
    if not nodes:
        return np.zeros(6 + len(vocab), dtype=np.float32)

    root = nodes[0]
    num_nodes = float(len(nodes))
    max_depth = float(max(int(n.get("depth", 0)) for n in nodes))

    root_startup = np.log1p(max(0.0, float(root.get("startup_cost") or 0.0)))
    root_total = np.log1p(max(0.0, float(root.get("total_cost") or 0.0)))
    root_rows = np.log1p(max(0.0, float(root.get("plan_rows") or 0.0)))
    root_width = np.log1p(max(0.0, float(root.get("plan_width") or 0.0)))

    # Count frequencies of each operator
    op_counts = np.zeros(len(vocab), dtype=np.float32)
    for n in nodes:
        op_id = vocab.get_id(n.get("node_type", "Unknown"))
        if 0 <= op_id < len(op_counts):
            op_counts[op_id] += 1.0

    return np.concatenate([
        np.array([num_nodes, max_depth, root_startup, root_total, root_rows, root_width], dtype=np.float32),
        op_counts,
    ])
