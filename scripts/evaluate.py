"""
Evaluation script for the neural cost model.

Computes: MAE, RMSE, R², Spearman, Kendall, Top-1 ranking accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import PlanGraphDataset, collate_plan_graphs
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.model import build_model

logger = logging.getLogger("neural_optimizer")


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    vocab_len = ckpt.get("vocab_len", 19)
    model = build_model(config, num_operators=vocab_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def _unwrap_pred(pred):
    if isinstance(pred, tuple):
        return pred[0]
    return pred


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return predictions for all samples."""
    preds = []
    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)
        p = model(x_op, x_cont, edge_index, batch_idx)
        preds.append(_unwrap_pred(p).cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression and ranking metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(y_true) > 1:
        sp, _ = stats.spearmanr(y_true, y_pred)
        kt, _ = stats.kendalltau(y_true, y_pred)
        spearman = float(sp) if not np.isnan(sp) else 0.0
        kendall = float(kt) if not np.isnan(kt) else 0.0
    else:
        spearman = kendall = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "kendall": kendall,
    }


def compute_top1_accuracy(
    dataset: PlanGraphDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_key: str = "query_id",
) -> float:
    """
    Top-1 accuracy: for each query group, did we rank the best plan first?
    Requires plan files to have query_id in the JSON.
    """
    groups: dict[int, list[int]] = {}
    for i, path in enumerate(dataset.files):
        with open(path) as f:
            data = json.load(f)
        qid = data.get(group_key, i)
        groups.setdefault(qid, []).append(i)

    correct = 0
    total = 0
    for qid, indices in groups.items():
        if len(indices) < 2:
            continue
        times = y_true[indices]
        preds = y_pred[indices]
        best_idx = np.argmin(times)
        pred_best = np.argmin(preds)
        correct += int(best_idx == pred_best)
        total += 1
    return correct / total if total > 0 else 0.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data", default="data/plans")
    parser.add_argument("--output", help="Save metrics JSON path")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = load_model(Path(args.checkpoint), device)
    vocab = OperatorVocabulary()
    dataset = PlanGraphDataset(
        args.data,
        vocab=vocab,
        max_depth=config.get("graph", {}).get("max_depth", 32),
    )
    if len(dataset) == 0:
        logger.error("No plans in %s", args.data)
        return

    loader = DataLoader(
        dataset,
        batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_plan_graphs,
    )

    y_pred = predict_all(model, loader, device)
    y_true = np.array([dataset[i]["y"].item() for i in range(len(dataset))])

    metrics = compute_metrics(y_true, y_pred)
    try:
        top1 = compute_top1_accuracy(dataset, y_true, y_pred)
        metrics["top1_accuracy"] = top1
    except Exception:
        metrics["top1_accuracy"] = 0.0

    for k, v in metrics.items():
        logger.info("%s: %.4f", k, v)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
