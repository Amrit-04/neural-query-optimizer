"""
Feature encoding for plan graph nodes.

- Operator type → learnable embedding
- Numeric features (cost, rows, depth)
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
        startup = n.get("startup_cost") or 0.0
        total_cost = n.get("total_cost") or 0.0
        plan_rows = np.log1p(n.get("plan_rows") or 0.0)
        plan_width = int(n.get("plan_width") or 0)
        actual_rows = np.log1p(n.get("actual_rows") or 0.0)
        actual_time = n.get("actual_time")

        # Raw numeric features (6 dims)
        feat = [
            op_id,
            depth,
            startup,
            total_cost,
            plan_rows,
            plan_width,
            actual_rows,
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
) -> dict[str, torch.Tensor]:
    """
    Build tensors for GNN input.
    Separates operator IDs (for embedding) from continuous features.
    """
    op_ids = []
    cont_feats = []
    depths = []
    for n in nodes:
        op_ids.append(vocab.get_id(n.get("node_type", "Unknown")))
        depth = min(int(n.get("depth", 0)), max_depth - 1)
        depths.append(depth)
        startup = n.get("startup_cost") or 0.0
        total_cost = n.get("total_cost") or 0.0
        plan_rows = np.log1p(n.get("plan_rows") or 0.0)
        plan_width = float(n.get("plan_width") or 0)
        actual_rows = np.log1p(n.get("actual_rows") or 0.0)
        cont_feats.append([startup, total_cost, plan_rows, plan_width, actual_rows])

    return {
        "operator_ids": torch.tensor(op_ids, dtype=torch.long),
        "continuous": torch.tensor(cont_feats, dtype=torch.float32),
        "depth": torch.tensor(depths, dtype=torch.long),
    }
