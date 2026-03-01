"""
Convert query execution plan tree → graph representation.

Nodes: operators
Edges: parent → child (plan structure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlanNode:
    """Single operator node in the plan tree."""

    node_type: str
    depth: int
    relation_name: str | None
    startup_cost: float | None
    total_cost: float | None
    plan_rows: float | None
    plan_width: int | None
    actual_rows: float | None
    actual_time: float | None
    children: list[PlanNode] = field(default_factory=list)


def parse_plan_tree(plan: dict, depth: int = 0) -> PlanNode:
    """Recursively parse plan JSON into PlanNode tree."""
    node = PlanNode(
        node_type=plan.get("Node Type", "Unknown"),
        depth=depth,
        relation_name=plan.get("Relation Name"),
        startup_cost=_safe_float(plan.get("Startup Cost")),
        total_cost=_safe_float(plan.get("Total Cost")),
        plan_rows=_safe_float(plan.get("Plan Rows")),
        plan_width=int(plan.get("Plan Width") or 0),
        actual_rows=_safe_float(plan.get("Actual Rows")),
        actual_time=_safe_float(plan.get("Actual Total Time")),
        children=[],
    )
    for child in plan.get("Plans", []):
        node.children.append(parse_plan_tree(child, depth + 1))
    return node


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tree_to_adjacency(node: PlanNode) -> tuple[list[PlanNode], list[tuple[int, int]]]:
    """
    Flatten tree to node list and edge list.
    Returns (nodes, edges) where edges are (parent_idx, child_idx).
    """
    nodes: list[PlanNode] = []
    edges: list[tuple[int, int]] = []

    def dfs(n: PlanNode, parent_idx: int | None) -> int:
        idx = len(nodes)
        nodes.append(n)
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for child in n.children:
            dfs(child, idx)
        return idx

    dfs(node, None)
    return nodes, edges


def build_graph(plan: dict) -> dict:
    """
    Build graph representation from plan dict.
    Returns dict with: nodes (list of node dicts), edge_index (2 x E array).
    """
    root = parse_plan_tree(plan)
    nodes, edge_list = tree_to_adjacency(root)

    node_dicts = [
        {
            "node_type": n.node_type,
            "depth": n.depth,
            "relation_name": n.relation_name,
            "startup_cost": n.startup_cost,
            "total_cost": n.total_cost,
            "plan_rows": n.plan_rows,
            "plan_width": n.plan_width,
            "actual_rows": n.actual_rows,
            "actual_time": n.actual_time,
        }
        for n in nodes
    ]

    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {"nodes": node_dicts, "edge_index": edge_index, "root": root}
