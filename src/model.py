"""
Graph Neural Network cost models.

- GCN baseline (PlanGCN)
- GAT variant with operator embeddings (PlanGAT)
- GraphSAGE variant (PlanSAGE)
- Global pooling: mean, add, attention
- Embedding extraction for t-SNE
- Multi-task: runtime + cardinality
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax


def global_attention_pool(x: torch.Tensor, batch: torch.Tensor, gate: nn.Linear) -> torch.Tensor:
    """
    Attention-based global pooling: weighted sum of nodes by learned gate.
    Vectorized and ONNX-compatible using PyG's softmax and global_add_pool.
    """
    scores = gate(x)  # (N, 1)
    weights = pyg_softmax(scores, batch)  # (N, 1) normalized per graph
    pooled = global_add_pool(x * weights, batch)
    return pooled


class PlanGCN(nn.Module):
    """GCN model with operator embeddings and global pooling."""

    def __init__(
        self,
        num_operators: int,
        operator_embed_dim: int,
        cont_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: Literal["mean", "add", "attention"] = "mean",
        predict_cardinality: bool = False,
    ):
        super().__init__()
        self.operator_embed = nn.Embedding(num_operators, operator_embed_dim)
        self.cont_proj = nn.Linear(cont_dim, hidden_dim - operator_embed_dim)
        input_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.dropout = dropout
        self.pooling = pooling
        self.attention_gate = nn.Linear(hidden_dim, 1) if pooling == "attention" else None
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.predict_cardinality = predict_cardinality
        self.cardinality_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            if predict_cardinality
            else None
        )

    def get_embedding(
        self,
        x_op: torch.Tensor,
        x_cont: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph-level embedding before final MLP."""
        x_emb = self.operator_embed(x_op)
        x_c = self.cont_proj(x_cont)
        x = torch.cat([x_emb, x_c], dim=-1)

        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "attention" and self.attention_gate is not None:
            x = global_attention_pool(x, batch, self.attention_gate)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return x

    def forward(
        self,
        x_op: torch.Tensor,
        x_cont: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        emb = self.get_embedding(x_op, x_cont, edge_index, batch)
        pred = self.mlp(emb).squeeze(-1)
        if self.predict_cardinality and self.cardinality_head is not None:
            card = self.cardinality_head(emb).squeeze(-1)
            return pred, card
        return pred


class PlanGAT(nn.Module):
    """Graph Attention Network with operator embeddings and global pooling."""

    def __init__(
        self,
        num_operators: int,
        operator_embed_dim: int,
        cont_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 4,
        pooling: Literal["mean", "add", "attention"] = "mean",
        predict_cardinality: bool = False,
    ):
        super().__init__()
        self.operator_embed = nn.Embedding(num_operators, operator_embed_dim)
        self.cont_proj = nn.Linear(cont_dim, hidden_dim - operator_embed_dim)
        in_channels = hidden_dim

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            out_channels = hidden_dim // heads
            self.convs.append(
                GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )
            in_channels = out_channels * heads

        self.hidden_dim = in_channels
        self.dropout = dropout
        self.pooling = pooling
        self.attention_gate = nn.Linear(in_channels, 1) if pooling == "attention" else None
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.predict_cardinality = predict_cardinality
        self.cardinality_head = (
            nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            if predict_cardinality
            else None
        )

    def get_embedding(
        self,
        x_op: torch.Tensor,
        x_cont: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph-level embedding before final MLP."""
        x_emb = self.operator_embed(x_op)
        x_c = self.cont_proj(x_cont)
        x = torch.cat([x_emb, x_c], dim=-1)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "attention" and self.attention_gate is not None:
            x = global_attention_pool(x, batch, self.attention_gate)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return x

    def forward(
        self,
        x_op: torch.Tensor,
        x_cont: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        emb = self.get_embedding(x_op, x_cont, edge_index, batch)
        pred = self.mlp(emb).squeeze(-1)
        if self.predict_cardinality and self.cardinality_head is not None:
            card = self.cardinality_head(emb).squeeze(-1)
            return pred, card
        return pred


class PlanSAGE(nn.Module):
    """GraphSAGE model with operator embeddings and global pooling."""

    def __init__(
        self,
        num_operators: int,
        operator_embed_dim: int,
        cont_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: Literal["mean", "add", "attention"] = "mean",
        predict_cardinality: bool = False,
    ):
        super().__init__()
        self.operator_embed = nn.Embedding(num_operators, operator_embed_dim)
        self.cont_proj = nn.Linear(cont_dim, hidden_dim - operator_embed_dim)
        input_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.dropout = dropout
        self.pooling = pooling
        self.attention_gate = nn.Linear(hidden_dim, 1) if pooling == "attention" else None
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.predict_cardinality = predict_cardinality
        self.cardinality_head = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            if predict_cardinality
            else None
        )

    def get_embedding(
        self,
        x_op: torch.Tensor,
        x_cont: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph-level embedding before final MLP."""
        x_emb = self.operator_embed(x_op)
        x_c = self.cont_proj(x_cont)
        x = torch.cat([x_emb, x_c], dim=-1)

        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "attention" and self.attention_gate is not None:
            x = global_attention_pool(x, batch, self.attention_gate)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return x

    def forward(
        self,
        x_op: torch.Tensor,
        x_cont: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        emb = self.get_embedding(x_op, x_cont, edge_index, batch)
        pred = self.mlp(emb).squeeze(-1)
        if self.predict_cardinality and self.cardinality_head is not None:
            card = self.cardinality_head(emb).squeeze(-1)
            return pred, card
        return pred


def build_model(config: dict, num_operators: int) -> nn.Module:
    """Build model from config."""
    model_cfg = config.get("model", {})
    graph_cfg = config.get("graph", {})
    model_type = model_cfg.get("type", "gcn").lower()
    op_dim = graph_cfg.get("operator_embedding_dim", 16)
    hidden = model_cfg.get("hidden_dim", 64)
    layers = model_cfg.get("num_layers", 2)
    dropout = model_cfg.get("dropout", 0.1)
    pooling = model_cfg.get("pooling", "mean")
    predict_cardinality = model_cfg.get("predict_cardinality", False)
    cont_dim = 4

    if model_type == "gat":
        return PlanGAT(
            num_operators=num_operators,
            operator_embed_dim=op_dim,
            cont_dim=cont_dim,
            hidden_dim=hidden,
            num_layers=layers,
            dropout=dropout,
            pooling=pooling,
            predict_cardinality=predict_cardinality,
        )
    elif model_type in ("sage", "graphsage"):
        return PlanSAGE(
            num_operators=num_operators,
            operator_embed_dim=op_dim,
            cont_dim=cont_dim,
            hidden_dim=hidden,
            num_layers=layers,
            dropout=dropout,
            pooling=pooling,
            predict_cardinality=predict_cardinality,
        )
    return PlanGCN(
        num_operators=num_operators,
        operator_embed_dim=op_dim,
        cont_dim=cont_dim,
        hidden_dim=hidden,
        num_layers=layers,
        dropout=dropout,
        pooling=pooling,
        predict_cardinality=predict_cardinality,
    )
