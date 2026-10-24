"""
Evaluation metrics computation.

Computes:
- Per-query Spearman rank correlation (averaged across queries)
- Per-query Kendall's tau (averaged across queries)
- Top-1 selection accuracy (% of queries where lowest predicted cost is fastest plan)
- Suboptimality regret (ms extra runtime compared to the best plan)
- Global regression metrics: MAE, RMSE, R2, MRE
- Pooled Spearman & Kendall (for comparison with legacy metrics)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Any


def evaluate_predictions(
    query_ids: list[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute full suite of ranking and regression metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 1. Global Regression Metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    mre = float(np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6)))

    # 2. Legacy Pooled Correlation (across all queries)
    if len(y_true) > 1:
        sp_pooled, _ = stats.spearmanr(y_true, y_pred)
        kt_pooled, _ = stats.kendalltau(y_true, y_pred)
        spearman_pooled = float(sp_pooled) if not np.isnan(sp_pooled) else 0.0
        kendall_pooled = float(kt_pooled) if not np.isnan(kt_pooled) else 0.0
    else:
        spearman_pooled = kendall_pooled = 0.0

    # 3. Per-Query Ranking Metrics
    groups: dict[Any, list[int]] = {}
    for idx, qid in enumerate(query_ids):
        groups.setdefault(qid, []).append(idx)

    spearmans: list[float] = []
    kendalls: list[float] = []
    top1_correct: list[int] = []
    regrets: list[float] = []

    for qid, indices in groups.items():
        if len(indices) < 2:
            continue

        q_true = y_true[indices]
        q_pred = y_pred[indices]

        # Spearman & Kendall within this query
        sp, _ = stats.spearmanr(q_true, q_pred)
        kt, _ = stats.kendalltau(q_true, q_pred)
        if not np.isnan(sp):
            spearmans.append(float(sp))
        if not np.isnan(kt):
            kendalls.append(float(kt))

        # Top-1 selection: plan with minimum predicted cost
        best_true_local_idx = int(np.argmin(q_true))
        best_pred_local_idx = int(np.argmin(q_pred))

        is_top1 = int(best_true_local_idx == best_pred_local_idx)
        top1_correct.append(is_top1)

        # Regret: extra runtime incurred over optimal plan
        regret = float(q_true[best_pred_local_idx] - q_true[best_true_local_idx])
        regrets.append(max(0.0, regret))

    mean_spearman = float(np.mean(spearmans)) if spearmans else 0.0
    mean_kendall = float(np.mean(kendalls)) if kendalls else 0.0
    top1_acc = float(np.mean(top1_correct)) if top1_correct else 0.0
    mean_regret = float(np.mean(regrets)) if regrets else 0.0
    median_regret = float(np.median(regrets)) if regrets else 0.0

    return {
        "spearman_per_query": mean_spearman,
        "kendall_per_query": mean_kendall,
        "top1_accuracy": top1_acc,
        "mean_regret_ms": mean_regret,
        "median_regret_ms": median_regret,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mre": mre,
        "spearman_pooled": spearman_pooled,
        "kendall_pooled": kendall_pooled,
        "num_queries_evaluated": len(top1_correct),
        "num_plans_evaluated": len(y_true),
    }
