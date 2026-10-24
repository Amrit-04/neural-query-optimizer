"""
Query-level dataset splitting utilities to guarantee zero data leakage.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def get_query_split(
    plan_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> dict[str, list[Any]]:
    """
    Split dataset strictly by query_id.
    Returns dict with 'train', 'val', 'test' lists of query IDs.
    """
    plan_path = Path(plan_dir)
    query_ids_set = set()

    for p in plan_path.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            if "query_id" in data:
                query_ids_set.add(data["query_id"])
        except Exception:
            pass

    query_ids = sorted(list(query_ids_set))
    rng = random.Random(seed)
    rng.shuffle(query_ids)

    n_total = len(query_ids)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * (1.0 - train_ratio - val_ratio)))
    n_train = n_total - n_val - n_test

    train_q = query_ids[:n_train]
    val_q = query_ids[n_train : n_train + n_val]
    test_q = query_ids[n_train + n_val :]

    split_dict = {
        "train": sorted(train_q),
        "val": sorted(val_q),
        "test": sorted(test_q),
    }

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(split_dict, f, indent=2)

    return split_dict
