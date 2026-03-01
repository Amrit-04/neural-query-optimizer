"""
Generate synthetic plan JSON for testing without PostgreSQL.

Creates diverse plan trees with varying operators, costs, and execution times.
Useful for Phase 1 development before TPC-H data is available.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OPERATORS = [
    "Seq Scan",
    "Index Scan",
    "Nested Loop",
    "Hash Join",
    "Merge Join",
    "Aggregate",
    "Hash Aggregate",
    "Sort",
    "Limit",
]


def random_plan(depth: int = 0, max_depth: int = 4) -> dict:
    """Generate a random plan subtree."""
    op = random.choice(OPERATORS)
    has_children = depth < max_depth and op in ("Nested Loop", "Hash Join", "Merge Join", "Aggregate", "Sort")

    rows = random.uniform(10, 100000)
    cost = random.uniform(1, 1000) + rows * 0.01
    actual_time = cost * random.uniform(0.5, 2.0)

    plan = {
        "Node Type": op,
        "Startup Cost": round(cost * 0.2, 2),
        "Total Cost": round(cost, 2),
        "Plan Rows": round(rows, 0),
        "Plan Width": random.randint(4, 64),
        "Actual Rows": round(rows * random.uniform(0.5, 1.5), 0),
        "Actual Total Time": round(actual_time, 2),
    }

    if has_children:
        n_children = 2 if op in ("Hash Join", "Merge Join", "Nested Loop") else 1
        plan["Plans"] = [random_plan(depth + 1, max_depth) for _ in range(n_children)]

    return plan


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/plans_synthetic")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        plan = random_plan(max_depth=random.randint(2, 5))
        exec_time = plan.get("Actual Total Time") or sum_actual_time(plan)
        record = {
            "query_id": i % 20,
            "hint_id": i // 20,
            "sql": f"SELECT ... -- synthetic {i}",
            "plan": plan,
            "execution_time": exec_time,
        }
        with open(out_dir / f"plan_synth_{i}.json", "w") as f:
            json.dump(record, f, indent=2)
    print(f"Generated {args.num} synthetic plans in {out_dir}")


def sum_actual_time(plan: dict) -> float:
    """Sum actual times (root dominates; use root for simplicity)."""
    t = plan.get("Actual Total Time") or 0.0
    for child in plan.get("Plans", []):
        t = max(t, sum_actual_time(child))
    return t


if __name__ == "__main__":
    main()
