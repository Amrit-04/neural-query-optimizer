"""
Error analysis for the neural cost model predictions.

Analyzes cases where the model's predictions were significantly off,
and provides insights into the types of errors made.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
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


@torch.no_grad()
def get_predictions(
    model: torch.nn.Module,
    dataset: PlanGraphDataset,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Get model predictions and ground truth, along with plan metadata."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_plan_graphs,
    )

    all_preds = []
    all_targets = []
    all_plans = []

    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)

        pred = model(x_op, x_cont, edge_index, batch_idx)
        if isinstance(pred, tuple):
            pred = pred[0]

        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(batch["y"].numpy())

        # Save plan metadata
        for i in range(len(batch["plans"])):
            plan_record = batch["plans"][i]
            plan_tree = plan_record.get("plan", {})
            plan_meta = {
                "plan_id": batch["plan_files"][i],
                "query_id": batch["query_ids"][i],
                "actual_cost": float(batch["y"][i].item()),
                "predicted_cost": float(pred[i].item()),
                "operator_types": [],
                "plan_depth": 0,
            }

            def traverse_plan(node, depth=0):
                if not isinstance(node, dict):
                    return
                plan_meta["operator_types"].append(node.get("Node Type", "unknown"))
                plan_meta["plan_depth"] = max(plan_meta["plan_depth"], depth)
                for child in node.get("Plans", []):
                    traverse_plan(child, depth + 1)

            traverse_plan(plan_tree)
            all_plans.append(plan_meta)

    return (
        np.concatenate(all_preds).flatten(),
        np.concatenate(all_targets).flatten(),
        all_plans,
    )


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plans: List[Dict],
    output_dir: Path,
) -> None:
    """Analyze and log prediction errors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = np.abs(y_pred - y_true)
    rel_errors = errors / (y_true + 1e-6)

    df = pd.DataFrame(plans)
    df["absolute_error"] = errors
    df["relative_error"] = rel_errors

    df.to_csv(output_dir / "prediction_errors.csv", index=False)

    worst_n = min(5, len(df))
    worst_predictions = df.nlargest(worst_n, "absolute_error")

    report = [
        "# Error Analysis Report",
        f"- **Total Plans Analyzed**: {len(df)}",
        f"- **Mean Absolute Error**: {errors.mean():.2f} ms",
        f"- **Median Absolute Error**: {np.median(errors):.2f} ms",
        f"- **Max Absolute Error**: {errors.max():.2f} ms",
        f"- **Mean Relative Error**: {rel_errors.mean()*100:.2f}%",
        "\n## Worst Predictions:",
    ]

    for _, row in worst_predictions.iterrows():
        report.append(
            f"- **Plan {row['plan_id']}** (Query: {row['query_id']}):\n"
            f"  - **Actual Cost**: {row['actual_cost']:.2f}\n"
            f"  - **Predicted**: {row['predicted_cost']:.2f}\n"
            f"  - **Absolute Error**: {row['absolute_error']:.2f}\n"
            f"  - **Relative Error**: {row['relative_error']*100:.2f}%\n"
            f"  - **Plan Depth**: {row['plan_depth']}\n"
            f"  - **Operators**: {', '.join(row['operator_types'])}"
        )

    with open(output_dir / "error_analysis.md", "w") as f:
        f.write("\n\n".join(report))

    logger.info("Error analysis complete. Results saved to %s", output_dir)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/plans_synthetic", help="Directory containing plan JSON files")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="results/error_analysis", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(Path(args.checkpoint), device)
    vocab = OperatorVocabulary()

    dataset = PlanGraphDataset(
        args.data,
        vocab=vocab,
        max_depth=config.get("graph", {}).get("max_depth", 32),
    )

    y_pred, y_true, plans = get_predictions(model, dataset, device)
    output_dir = Path(args.output)
    analyze_errors(y_true, y_pred, plans, output_dir)


if __name__ == "__main__":
    main()
