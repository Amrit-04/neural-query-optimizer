"""
Training script for the neural cost model.

Phase 1: Regression (MSE) on execution time.
Phase 2: Pairwise ranking loss, multi-task (cardinality).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.config import load_config, set_seed
from src.dataset import (
    PlanGraphDataset,
    PlanPairDataset,
    collate_plan_graphs,
    collate_plan_pairs,
)
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.model import build_model

logger = logging.getLogger("neural_optimizer")


def _unwrap_pred(pred, model):
    """Handle model returning (pred, card) for multi-task."""
    if isinstance(pred, tuple):
        return pred[0], pred[1]
    return pred, None


def train_epoch_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cardinality_weight: float = 0.0,
) -> float:
    """Single epoch of regression training (optionally with cardinality)."""
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
        pred_runtime, pred_card = _unwrap_pred(pred, model)

        loss = F.mse_loss(pred_runtime, y)
        if pred_card is not None and "cardinality" in batch and cardinality_weight > 0:
            card = batch["cardinality"].to(device)
            loss = loss + cardinality_weight * F.mse_loss(pred_card, card)

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
    """Single epoch of pairwise ranking training."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        fast = batch["fast"]
        slow = batch["slow"]

        def run_plan(pl):
            pred = model(
                pl["x_op"].to(device),
                pl["x_cont"].to(device),
                pl["edge_index"].to(device),
                pl["batch"].to(device),
            )
            return _unwrap_pred(pred, model)[0]

        pred_fast = run_plan(fast)
        pred_slow = run_plan(slow)
        # Margin ranking: want pred_fast < pred_slow (lower cost = better)
        loss = F.margin_ranking_loss(pred_fast, pred_slow, torch.ones_like(pred_fast, device=device), margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(pred_fast)
        n += len(pred_fast)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute validation loss (MSE) and MAE."""
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)
        y = batch["y"].to(device)

        pred = model(x_op, x_cont, edge_index, batch_idx)
        pred_runtime, _ = _unwrap_pred(pred, model)
        mse_sum += F.mse_loss(pred_runtime, y, reduction="sum").item()
        mae_sum += F.l1_loss(pred_runtime, y, reduction="sum").item()
        n += len(y)
    n = max(n, 1)
    return mse_sum / n, mae_sum / n


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data", default="data/plans", help="Plan JSON directory")
    parser.add_argument("--checkpoint", default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--mode", choices=["regression", "ranking"], help="Override training mode")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.get("seed", 42), config.get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    vocab = OperatorVocabulary()
    base_dataset = PlanGraphDataset(
        args.data,
        vocab=vocab,
        max_depth=config.get("graph", {}).get("max_depth", 32),
    )
    if len(base_dataset) == 0:
        logger.error("No plans found in %s. Run extract_plans.py first.", args.data)
        return

    train_cfg = config.get("training", {})
    mode = args.mode if args.mode else train_cfg.get("mode", "regression")

    if mode == "ranking":
        pair_dataset = PlanPairDataset(base_dataset)
        n_pair_val = int(len(pair_dataset) * args.val_ratio)
        n_pair_train = len(pair_dataset) - n_pair_val
        train_ds, _ = random_split(pair_dataset, [n_pair_train, n_pair_val])
        collate_fn = collate_plan_pairs
        n_val = int(len(base_dataset) * args.val_ratio)
        n_train = len(base_dataset) - n_val
        _, val_subset = random_split(base_dataset, [n_train, n_val])
    else:
        n_val = int(len(base_dataset) * args.val_ratio)
        n_train = len(base_dataset) - n_val
        train_ds, val_subset = random_split(base_dataset, [n_train, n_val])
        collate_fn = collate_plan_graphs

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_plan_graphs,
    )

    model = build_model(config, num_operators=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    patience = train_cfg.get("patience", 15)
    patience_counter = 0

    card_weight = train_cfg.get("cardinality_weight", 0.0) if config.get("model", {}).get("predict_cardinality") else 0.0
    margin = train_cfg.get("ranking_margin", 0.1)

    for epoch in range(train_cfg.get("epochs", 100)):
        if mode == "ranking":
            train_loss = train_epoch_ranking(model, train_loader, optimizer, device, margin=margin)
        else:
            train_loss = train_epoch_regression(
                model, train_loader, optimizer, device, cardinality_weight=card_weight
            )
        val_mse, val_mae = evaluate(model, val_loader, device)

        if (epoch + 1) % train_cfg.get("log_interval", 10) == 0:
            logger.info(
                "Epoch %d | train_mse=%.4f | val_mse=%.4f | val_mae=%.4f",
                epoch + 1,
                train_loss,
                val_mse,
                val_mae,
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocab_len": len(vocab),
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Best val MAE: %.4f", best_val_mae)


if __name__ == "__main__":
    main()
