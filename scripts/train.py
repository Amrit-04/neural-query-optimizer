"""
Training script for the neural cost model.

Features:
- Query-level train/validation/test split (zero data leakage).
- Pairwise margin ranking loss (fixed target sign) and regression (MSE).
- MLflow experiment tracking.
- Fast execution with --quick mode (<15s) and full budget (<60s).
- Logs elapsed time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config, set_seed
from src.dataset import (
    PlanGraphDataset,
    PlanPairDataset,
    collate_plan_graphs,
    collate_plan_pairs,
)
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.metrics import evaluate_predictions
from src.model import build_model
from src.split_utils import get_query_split

logger = logging.getLogger("neural_optimizer")


def _unwrap_pred(pred, model):
    if isinstance(pred, tuple):
        return pred[0], pred[1]
    return pred, None


def train_epoch_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train single epoch with regression (MSE) loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred = model(x_op, x_cont, edge_index, batch_idx)
        pred_runtime, _ = _unwrap_pred(pred, model)

        loss = F.mse_loss(pred_runtime, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def train_epoch_ranking(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 0.1,
) -> float:
    """
    Train single epoch with pairwise ranking loss.
    Math: want pred_fast < pred_slow.
    margin_ranking_loss(pred_slow, pred_fast, target=1, margin=m)
    Loss = max(0, -(pred_slow - pred_fast) + m) = max(0, pred_fast - pred_slow + m)
    """
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        fast = batch["fast"]
        slow = batch["slow"]

        def run_plan(pl):
            p = model(
                pl["x_op"].to(device),
                pl["x_cont"].to(device),
                pl["edge_index"].to(device),
                pl["batch"].to(device),
            )
            return _unwrap_pred(p, model)[0]

        pred_fast = run_plan(fast)
        pred_slow = run_plan(slow)

        target = torch.ones_like(pred_slow)
        loss = F.margin_ranking_loss(pred_slow, pred_fast, target, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(pred_fast)
        n += len(pred_fast)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Run model inference and return (y_true, y_pred, query_ids)."""
    model.eval()
    preds = []
    trues = []
    qids = []
    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)
        y = batch["y"].to(device)

        p = model(x_op, x_cont, edge_index, batch_idx)
        pred_runtime, _ = _unwrap_pred(p, model)

        preds.append(pred_runtime.cpu().numpy())
        trues.append(y.cpu().numpy())
        qids.extend(batch["query_ids"])

    y_pred = np.concatenate(preds, axis=0) if preds else np.array([])
    y_true = np.concatenate(trues, axis=0) if trues else np.array([])
    return y_true, y_pred, qids


def main() -> None:
    start_time = time.time()
    setup_logging()

    parser = argparse.ArgumentParser(description="Train neural cost model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data", default="data/plans_synthetic", help="Plan JSON directory")
    parser.add_argument("--checkpoint", default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--mode", choices=["regression", "ranking"], default=None)
    parser.add_argument("--model_type", choices=["gcn", "gat", "sage"], default=None)
    parser.add_argument("--pooling", choices=["mean", "attention"], default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing (<15s)")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed, config.get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s | Seed: %d", device, seed)

    # 1. Query-level Split
    split_info = get_query_split(args.data, seed=seed, save_path="results/query_split.json")
    train_q, val_q, test_q = split_info["train"], split_info["val"], split_info["test"]
    logger.info(
        "Split: %d train queries, %d val queries, %d test queries (Zero plan leakage)",
        len(train_q),
        len(val_q),
        len(test_q),
    )

    vocab = OperatorVocabulary()
    train_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=set(train_q))
    val_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=set(val_q))
    test_ds = PlanGraphDataset(args.data, vocab=vocab, allowed_query_ids=set(test_q))

    train_cfg = config.get("training", {})
    mode = args.mode or train_cfg.get("mode", "ranking")
    epochs = 10 if args.quick else (args.epochs or train_cfg.get("epochs", 40))
    batch_size = 16 if args.quick else train_cfg.get("batch_size", 32)
    lr = train_cfg.get("lr", 0.001)
    margin = train_cfg.get("ranking_margin", 0.1)

    # Apply CLI model overrides if given
    if args.model_type:
        config.setdefault("model", {})["type"] = args.model_type
    if args.pooling:
        config.setdefault("model", {})["pooling"] = args.pooling
    if args.embed_dim:
        config.setdefault("graph", {})["operator_embedding_dim"] = args.embed_dim

    model_type = config.get("model", {}).get("type", "gat")
    pooling_type = config.get("model", {}).get("pooling", "attention")
    embed_dim = config.get("graph", {}).get("operator_embedding_dim", 16)

    # Data loaders
    if mode == "ranking":
        pair_ds = PlanPairDataset(train_ds)
        train_loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_plan_pairs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_plan_graphs)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_plan_graphs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_plan_graphs)

    model = build_model(config, num_operators=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-5))

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # MLflow Setup
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("neural_query_optimizer")
        mlflow.start_run(run_name=f"{model_type}_{pooling_type}_d{embed_dim}_s{seed}")
        mlflow.log_params({
            "model_type": model_type,
            "pooling": pooling_type,
            "embed_dim": embed_dim,
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "quick_mode": args.quick,
        })

    if mode == "ranking":
        best_val_score = -float("inf")
    else:
        best_val_score = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        if mode == "ranking":
            loss = train_epoch_ranking(model, train_loader, optimizer, device, margin=margin)
        else:
            loss = train_epoch_regression(model, train_loader, optimizer, device)

        val_true, val_pred, val_qids = evaluate_dataset(model, val_loader, device)
        val_metrics = evaluate_predictions(val_qids, val_true, val_pred)

        if use_mlflow:
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_spearman", val_metrics["spearman_per_query"], step=epoch)
            mlflow.log_metric("val_top1_acc", val_metrics["top1_accuracy"], step=epoch)
            mlflow.log_metric("val_mae", val_metrics["mae"], step=epoch)

        if mode == "ranking":
            is_best = (val_metrics["spearman_per_query"] > best_val_score)
            if is_best:
                best_val_score = val_metrics["spearman_per_query"]
        else:
            is_best = (val_metrics["mae"] < best_val_score)
            if is_best:
                best_val_score = val_metrics["mae"]

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %2d/%2d | loss=%.4f | val_sp=%.4f | val_top1=%.2f%% | val_mae=%.2f ms",
                epoch + 1,
                epochs,
                loss,
                val_metrics["spearman_per_query"],
                val_metrics["top1_accuracy"] * 100,
                val_metrics["mae"],
            )

        if is_best:
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocab_len": len(vocab),
                    "split": split_info,
                    "val_metrics": val_metrics,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience and not args.quick:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    # Load best checkpoint for final test evaluation
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_true, test_pred, test_qids = evaluate_dataset(model, test_loader, device)
    test_metrics = evaluate_predictions(test_qids, test_true, test_pred)

    elapsed = time.time() - start_time
    test_metrics["training_time_seconds"] = round(elapsed, 2)
    test_metrics["model_type"] = model_type
    test_metrics["pooling"] = pooling_type
    test_metrics["embed_dim"] = embed_dim
    test_metrics["seed"] = seed

    if use_mlflow:
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.end_run()

    # Save metrics to results/
    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "best_model_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("==================================================")
    logger.info("Training Finished in %.2fs!", elapsed)
    logger.info("Test Per-Query Spearman: %.4f", test_metrics["spearman_per_query"])
    logger.info("Test Per-Query Kendall:  %.4f", test_metrics["kendall_per_query"])
    logger.info("Test Top-1 Accuracy:     %.2f%%", test_metrics["top1_accuracy"] * 100)
    logger.info("Test Mean Regret:        %.2f ms", test_metrics["mean_regret_ms"])
    logger.info("Test MAE:                %.2f ms", test_metrics["mae"])
    logger.info("Test R2:                 %.4f", test_metrics["r2"])
    logger.info("==================================================")


if __name__ == "__main__":
    main()
