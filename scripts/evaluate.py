"""
Comprehensive evaluation script for neural cost model and baselines.

Evaluates on the held-out query test split:
1. Learned GNN Model (checkpoint)
2. Baseline 1: PostgreSQL Native Cost Optimizer (EXPLAIN total cost)
3. Baseline 2: Flat-Feature Tabular Model (HistGradientBoosting on tree summary stats)

Computes per-query Spearman, Kendall, Top-1 selection accuracy, regret, and regression metrics.
Saves results to results/evaluation_results.json and results/baseline_comparison.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import PlanGraphDataset, collate_plan_graphs
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.metrics import evaluate_predictions
from src.model import build_model
from src.split_utils import get_query_split

logger = logging.getLogger("neural_optimizer")


def load_trained_model(checkpoint_path: Path, device: torch.device):
    """Load model and configuration from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    vocab_len = ckpt.get("vocab_len", 19)
    model = build_model(config, num_operators=vocab_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, ckpt.get("split")


@torch.no_grad()
def predict_gnn(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Extract model predictions, ground truths, native costs, flat features, and query IDs."""
    model.eval()
    y_preds = []
    y_trues = []
    native_costs = []
    flat_feats = []
    query_ids = []

    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)

        pred = model(x_op, x_cont, edge_index, batch_idx)
        if isinstance(pred, tuple):
            pred = pred[0]

        y_preds.append(pred.cpu().numpy())
        y_trues.append(batch["y"].numpy())
        native_costs.append(batch["root_cost"].numpy())
        flat_feats.append(batch["flat_features"].numpy())
        query_ids.extend(batch["query_ids"])

    return (
        np.concatenate(y_preds),
        np.concatenate(y_trues),
        np.concatenate(native_costs),
        np.concatenate(flat_feats),
        query_ids,
    )


def main() -> None:
    start_time = time.time()
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate model and baselines on held-out test split")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data", default="data/plans_synthetic")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--split_file", default="results/query_split.json")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model and split
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Run train.py first.", ckpt_path)
        sys.exit(1)

    model, config, split_info = load_trained_model(ckpt_path, device)
    if split_info is None and Path(args.split_file).exists():
        with open(args.split_file) as f:
            split_info = json.load(f)
    elif split_info is None:
        split_info = get_query_split(args.data, seed=42)

    train_q = set(split_info["train"])
    test_q = set(split_info["test"])
    logger.info("Evaluating on %d held-out test queries (trained on %d queries)", len(test_q), len(train_q))

    vocab = OperatorVocabulary()
    train_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=train_q)
    test_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=test_q)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_plan_graphs)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_plan_graphs)

    # 2. GNN Model Evaluation on Test Set
    y_pred_gnn, y_true_test, y_native_test, X_flat_test, test_qids = predict_gnn(model, test_loader, device)
    gnn_metrics = evaluate_predictions(test_qids, y_true_test, y_pred_gnn)

    # 3. Baseline 1: PostgreSQL Native Cost Optimizer
    native_metrics = evaluate_predictions(test_qids, y_true_test, y_native_test)

    # 4. Baseline 2: Flat-Feature Model (Gradient Boosting)
    # Collect train flat features and labels
    _, y_true_train, _, X_flat_train, _ = predict_gnn(model, train_loader, device)
    gbdt = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    gbdt.fit(X_flat_train, y_true_train)
    y_pred_gbdt = gbdt.predict(X_flat_test)
    gbdt_metrics = evaluate_predictions(test_qids, y_true_test, y_pred_gbdt)

    # 5. Improvement Calculations
    def calc_improvement(val_m, val_b):
        if val_b == 0:
            return 0.0
        return ((val_m - val_b) / abs(val_b)) * 100.0

    def calc_regret_reduction(reg_m, reg_b):
        if reg_b == 0:
            return 0.0
        return ((reg_b - reg_m) / reg_b) * 100.0

    top1_gain_vs_native = calc_improvement(gnn_metrics["top1_accuracy"], native_metrics["top1_accuracy"])
    top1_gain_vs_gbdt = calc_improvement(gnn_metrics["top1_accuracy"], gbdt_metrics["top1_accuracy"])
    regret_reduction_vs_native = calc_regret_reduction(gnn_metrics["mean_regret_ms"], native_metrics["mean_regret_ms"])
    regret_reduction_vs_gbdt = calc_regret_reduction(gnn_metrics["mean_regret_ms"], gbdt_metrics["mean_regret_ms"])

    elapsed = time.time() - start_time
    full_results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "test_queries_count": len(test_q),
        "test_plans_count": len(y_true_test),
        "gnn_model": gnn_metrics,
        "baseline_postgres_native": native_metrics,
        "baseline_gradient_boosting": gbdt_metrics,
        "improvements": {
            "top1_gain_pct_vs_postgres_native": round(top1_gain_vs_native, 2),
            "top1_gain_pct_vs_gradient_boosting": round(top1_gain_vs_gbdt, 2),
            "regret_reduction_pct_vs_postgres_native": round(regret_reduction_vs_native, 2),
            "regret_reduction_pct_vs_gradient_boosting": round(regret_reduction_vs_gbdt, 2),
        },
    }

    # Save JSON results
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # Generate Markdown Table
    md_content = f"""# Benchmark & Baseline Comparison Report

**Date:** {full_results['evaluation_timestamp']}  
**Evaluation Set:** {len(test_q)} Held-Out Queries ({len(y_true_test)} Plans), Zero Query Leakage  
**Evaluation Duration:** {elapsed:.2f}s  

---

## 1. Primary Ranking & Selection Metrics (Within-Query)

| Metric | GNN Model (Learned) | Baseline 1: PostgreSQL Native Cost | Baseline 2: Flat GBDT | Improvement vs. Postgres Native | Improvement vs. Flat GBDT |
|---|---|---|---|---|---|
| **Top-1 Selection Accuracy** | **{gnn_metrics['top1_accuracy']*100:.1f}%** | {native_metrics['top1_accuracy']*100:.1f}% | {gbdt_metrics['top1_accuracy']*100:.1f}% | **{top1_gain_vs_native:+.1f}%** | **{top1_gain_vs_gbdt:+.1f}%** |
| **Mean Regret (ms)** | **{gnn_metrics['mean_regret_ms']:.2f} ms** | {native_metrics['mean_regret_ms']:.2f} ms | {gbdt_metrics['mean_regret_ms']:.2f} ms | **{regret_reduction_vs_native:+.1f}%** | **{regret_reduction_vs_gbdt:+.1f}%** |
| **Median Regret (ms)** | **{gnn_metrics['median_regret_ms']:.2f} ms** | {native_metrics['median_regret_ms']:.2f} ms | {gbdt_metrics['median_regret_ms']:.2f} ms | - | - |
| **Spearman Rank Corr. (Per Query)** | **{gnn_metrics['spearman_per_query']:.3f}** | {native_metrics['spearman_per_query']:.3f} | {gbdt_metrics['spearman_per_query']:.3f} | {calc_improvement(gnn_metrics['spearman_per_query'], native_metrics['spearman_per_query']):+.1f}% | {calc_improvement(gnn_metrics['spearman_per_query'], gbdt_metrics['spearman_per_query']):+.1f}% |
| **Kendall's tau (Per Query)** | **{gnn_metrics['kendall_per_query']:.3f}** | {native_metrics['kendall_per_query']:.3f} | {gbdt_metrics['kendall_per_query']:.3f} | {calc_improvement(gnn_metrics['kendall_per_query'], native_metrics['kendall_per_query']):+.1f}% | {calc_improvement(gnn_metrics['kendall_per_query'], gbdt_metrics['kendall_per_query']):+.1f}% |

---

## 2. Global Regression & Pooled Metrics

| Metric | GNN Model | PostgreSQL Native | Flat GBDT |
|---|---|---|---|
| **MAE (ms)** | {gnn_metrics['mae']:.2f} ms | {native_metrics['mae']:.2f} ms | {gbdt_metrics['mae']:.2f} ms |
| **RMSE (ms)** | {gnn_metrics['rmse']:.2f} ms | {native_metrics['rmse']:.2f} ms | {gbdt_metrics['rmse']:.2f} ms |
| **R² Score** | {gnn_metrics['r2']:.3f} | {native_metrics['r2']:.3f} | {gbdt_metrics['r2']:.3f} |
| **Mean Relative Error (MRE)** | {gnn_metrics['mre']*100:.1f}% | {native_metrics['mre']*100:.1f}% | {gbdt_metrics['mre']*100:.1f}% |
| **Spearman (Pooled across queries)** | {gnn_metrics['spearman_pooled']:.3f} | {native_metrics['spearman_pooled']:.3f} | {gbdt_metrics['spearman_pooled']:.3f} |
| **Kendall (Pooled across queries)** | {gnn_metrics['kendall_pooled']:.3f} | {native_metrics['kendall_pooled']:.3f} | {gbdt_metrics['kendall_pooled']:.3f} |
"""

    with open(out_dir / "baseline_comparison.md", "w") as f:
        f.write(md_content)

    logger.info("Evaluation Complete! Results saved to %s", out_dir)
    print(md_content)


if __name__ == "__main__":
    main()
