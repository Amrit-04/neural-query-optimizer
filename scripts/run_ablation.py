"""
Automated ablation runner across GNN architectures, pooling methods, and embedding dimensions.

Ablation Grid:
- Architectures: GCN, GAT, GraphSAGE
- Pooling: Mean, Attention
- Operator Embedding Dimension: 16, 32
- Seeds: 3 random seeds (42, 43, 44) to compute Mean and Standard Deviation.

Logs all runs to MLflow and outputs:
- results/ablation_results.json
- results/ablation_results.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, set_seed
from src.dataset import PlanGraphDataset, PlanPairDataset, collate_plan_graphs, collate_plan_pairs
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.metrics import evaluate_predictions
from src.model import build_model
from src.split_utils import get_query_split
from scripts.train import train_epoch_ranking, evaluate_dataset

logger = logging.getLogger("neural_optimizer")


def run_single_train_eval(
    model_type: str,
    pooling: str,
    embed_dim: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    vocab: OperatorVocabulary,
    train_ds: PlanGraphDataset,
    val_ds: PlanGraphDataset,
    test_ds: PlanGraphDataset,
) -> dict[str, float]:
    """Train and evaluate a single model configuration on given seed."""
    set_seed(seed, deterministic=True)

    config = {
        "model": {
            "type": model_type,
            "pooling": pooling,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "predict_cardinality": False,
        },
        "graph": {
            "operator_embedding_dim": embed_dim,
            "max_depth": 32,
        },
    }

    pair_ds = PlanPairDataset(train_ds)
    train_loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_plan_pairs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_plan_graphs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_plan_graphs)

    model = build_model(config, num_operators=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_sp = -float("inf")
    best_weights = None
    patience = 8
    patience_cnt = 0

    for epoch in range(epochs):
        train_epoch_ranking(model, train_loader, optimizer, device, margin=0.1)
        val_true, val_pred, val_qids = evaluate_dataset(model, val_loader, device)
        val_metrics = evaluate_predictions(val_qids, val_true, val_pred)

        if val_metrics["spearman_per_query"] > best_val_sp:
            best_val_sp = val_metrics["spearman_per_query"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_true, test_pred, test_qids = evaluate_dataset(model, test_loader, device)
    test_metrics = evaluate_predictions(test_qids, test_true, test_pred)
    return test_metrics


def main() -> None:
    start_time = time.time()
    setup_logging()

    parser = argparse.ArgumentParser(description="Run complete GNN ablation study")
    parser.add_argument("--data", default="data/plans_synthetic")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1 seed, 4 configs, 5 epochs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        model_types = ["gcn", "gat"]
        poolings = ["mean", "attention"]
        embed_dims = [16]
        seeds = [42]
        epochs = 6
        batch_size = 16
    else:
        model_types = ["gcn", "gat", "sage"]
        poolings = ["mean", "attention"]
        embed_dims = [16, 32]
        seeds = [42, 43, 44]
        epochs = 25
        batch_size = 32

    # Query split
    split_info = get_query_split(args.data, seed=42)
    vocab = OperatorVocabulary()
    train_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=set(split_info["train"]))
    val_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=set(split_info["val"]))
    test_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=set(split_info["test"]))

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("neural_query_optimizer_ablation")

    grid = list(itertools.product(model_types, poolings, embed_dims))
    total_runs = len(grid) * len(seeds)
    logger.info("Starting Ablation: %d configurations × %d seeds = %d total runs", len(grid), len(seeds), total_runs)

    results_by_config = []
    run_idx = 0

    for m_type, pool, edim in grid:
        cfg_name = f"{m_type.upper()}_{pool}_dim{edim}"
        seed_metrics = []

        for seed in seeds:
            run_idx += 1
            t0 = time.time()
            metrics = run_single_train_eval(
                model_type=m_type,
                pooling=pool,
                embed_dim=edim,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=0.001,
                device=device,
                vocab=vocab,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
            )
            dur = time.time() - t0

            # Log to MLflow
            with mlflow.start_run(run_name=f"{cfg_name}_s{seed}"):
                mlflow.log_params({
                    "model_type": m_type,
                    "pooling": pool,
                    "embed_dim": edim,
                    "seed": seed,
                    "epochs": epochs,
                })
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"test_{k}", v)
                mlflow.log_metric("duration_seconds", dur)

            seed_metrics.append(metrics)
            logger.info(
                "[%d/%d] %s (seed %d): Sp=%.3f, Top1=%.1f%%, Regret=%.1fms (%.1fs)",
                run_idx,
                total_runs,
                cfg_name,
                seed,
                metrics["spearman_per_query"],
                metrics["top1_accuracy"] * 100,
                metrics["mean_regret_ms"],
                dur,
            )

        # Aggregate across seeds
        def agg(key):
            vals = [m[key] for m in seed_metrics]
            return float(np.mean(vals)), float(np.std(vals))

        sp_m, sp_s = agg("spearman_per_query")
        kt_m, kt_s = agg("kendall_per_query")
        t1_m, t1_s = agg("top1_accuracy")
        rg_m, rg_s = agg("mean_regret_ms")
        mae_m, mae_s = agg("mae")
        r2_m, r2_s = agg("r2")

        results_by_config.append({
            "model_type": m_type,
            "pooling": pool,
            "embed_dim": edim,
            "config_name": cfg_name,
            "seeds": seeds,
            "spearman_mean": round(sp_m, 4),
            "spearman_std": round(sp_s, 4),
            "kendall_mean": round(kt_m, 4),
            "kendall_std": round(kt_s, 4),
            "top1_acc_mean": round(t1_m, 4),
            "top1_acc_std": round(t1_s, 4),
            "regret_ms_mean": round(rg_m, 2),
            "regret_ms_std": round(rg_s, 2),
            "mae_mean": round(mae_m, 2),
            "mae_std": round(mae_s, 2),
            "r2_mean": round(r2_m, 4),
            "r2_std": round(r2_s, 4),
        })

    # Sort to determine winner: highest Top-1 accuracy, then highest Spearman
    results_by_config.sort(key=lambda x: (x["top1_acc_mean"], x["spearman_mean"]), reverse=True)
    winner = results_by_config[0]

    elapsed = time.time() - start_time
    output_data = {
        "ablation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(elapsed, 2),
        "total_runs": total_runs,
        "seeds": seeds,
        "winning_configuration": winner,
        "configurations": results_by_config,
    }

    # Save JSON
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Format Markdown Table
    md_lines = [
        "# GNN Architecture & Feature Ablation Study Results",
        f"**Date:** {output_data['ablation_timestamp']}  ",
        f"**Total Runs:** {total_runs} across seeds {seeds}  ",
        f"**Total Duration:** {elapsed:.2f}s ({elapsed/60:.2f} min)  ",
        f"**Winning Configuration:** **{winner['config_name']}** (Top-1: {winner['top1_acc_mean']*100:.1f}±{winner['top1_acc_std']*100:.1f}%, Spearman: {winner['spearman_mean']:.3f}±{winner['spearman_std']:.3f})  \n",
        "| Architecture | Pooling | Embedding Dim | Top-1 Accuracy (%) | Mean Regret (ms) | Spearman (Per Query) | Kendall's tau | MAE (ms) | R² Score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results_by_config:
        md_lines.append(
            f"| {r['model_type'].upper()} | {r['pooling']} | {r['embed_dim']} | "
            f"{r['top1_acc_mean']*100:.1f}% ± {r['top1_acc_std']*100:.1f}% | "
            f"{r['regret_ms_mean']:.1f} ± {r['regret_ms_std']:.1f} | "
            f"{r['spearman_mean']:.3f} ± {r['spearman_std']:.3f} | "
            f"{r['kendall_mean']:.3f} ± {r['kendall_std']:.3f} | "
            f"{r['mae_mean']:.1f} ± {r['mae_std']:.1f} | "
            f"{r['r2_mean']:.3f} ± {r['r2_std']:.3f} |"
        )

    md_content = "\n".join(md_lines)
    with open(out_dir / "ablation_results.md", "w") as f:
        f.write(md_content)

    logger.info("Ablation Study Finished in %.2fs! Results saved to %s", elapsed, out_dir)
    print(md_content)


if __name__ == "__main__":
    main()
