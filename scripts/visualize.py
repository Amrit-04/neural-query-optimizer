"""
Visualization utilities for plan embeddings and ablation.

- t-SNE / UMAP of graph embeddings (Phase 2/3)
- Ablation table formatting
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.logging_config import setup_logging

logger = logging.getLogger("neural_optimizer")


def extract_embeddings(
    model: Any,
    loader: Any,
    device: Any,
) -> tuple[list[float], list[list[float]]]:
    """
    Extract graph-level embeddings from the model (before final MLP).
    Returns (y_true, embeddings) for t-SNE visualization.
    """
    if not hasattr(model, "get_embedding"):
        raise ValueError("Model must implement get_embedding() for visualization")
    import torch
    model.eval()
    embeddings = []
    y_true = []
    with torch.no_grad():
        for batch in loader:
            x_op = batch["x_op"].to(device)
            x_cont = batch["x_cont"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch"].to(device)
            emb = model.get_embedding(x_op, x_cont, edge_index, batch_idx)
            embeddings.append(emb.cpu().numpy())
            y_true.extend(batch["y"].cpu().tolist())
    X = np.concatenate(embeddings, axis=0)
    return y_true, [X[i].tolist() for i in range(len(X))]


def plot_tsne(
    embeddings: list[list[float]] | np.ndarray,
    labels: list[str] | list[float] | None = None,
    output_path: Path | None = None,
    color_by: str = "index",
) -> None:
    """t-SNE visualization of plan embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("sklearn/matplotlib not available for t-SNE")
        return

    X = np.array(embeddings)
    if len(X) < 2:
        logger.warning("Need at least 2 samples for t-SNE")
        return

    perplexity = min(30, len(X) - 1)
    if perplexity < 5:
        perplexity = 5
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is not None and color_by == "label":
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Target (runtime)")
    else:
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=range(len(X)), cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Plan index")
    ax.set_title("Plan Embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved t-SNE plot to %s", output_path)
    plt.close()


def format_ablation_table(results: list[dict], output_path: Path | None = None) -> str:
    """
    Format ablation study results as Markdown table.
    """
    if not results:
        return ""

    keys = list(results[0].keys())
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    rows = []
    for r in results:
        row = "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |"
        rows.append(row)

    table = "\n".join([header, sep] + rows)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# Ablation Study\n\n")
            f.write(table)
        logger.info("Saved ablation table to %s", output_path)
    return table


def main() -> None:
    """CLI for ablation table and t-SNE visualization."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", help="Command")
    ablation = sub.add_parser("ablation")
    ablation.add_argument("--results", nargs="+", required=True, help="JSON files with metrics")
    ablation.add_argument("--output", default="outputs/ablation.md")
    tsne_p = sub.add_parser("tsne")
    tsne_p.add_argument("--checkpoint", default="checkpoints/best.pt")
    tsne_p.add_argument("--data", default="data/plans_synthetic")
    tsne_p.add_argument("--config", default="config/default.yaml")
    tsne_p.add_argument("--output", default="outputs/tsne_embeddings.png")
    tsne_p.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    setup_logging()

    if args.cmd == "ablation":
        results = []
        for p in args.results:
            with open(p) as f:
                results.append(json.load(f))
        if results:
            format_ablation_table(results, Path(args.output))
            print(format_ablation_table(results))
        return

    if args.cmd == "tsne":
        import torch
        from src.config import load_config
        from src.dataset import PlanGraphDataset, collate_plan_graphs
        from src.feature_encoder import OperatorVocabulary
        from src.model import build_model

        config = load_config(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(ckpt.get("config", config), num_operators=ckpt.get("vocab_len", 19))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        vocab = OperatorVocabulary()
        dataset = PlanGraphDataset(
            args.data,
            vocab=vocab,
            max_depth=config.get("graph", {}).get("max_depth", 32),
        )
        indices = list(range(min(len(dataset), args.max_samples)))
        from torch.utils.data import Subset
        subset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_plan_graphs,
        )
        y_true, embeddings = extract_embeddings(model, loader, device)
        plot_tsne(
            embeddings,
            labels=y_true,
            output_path=Path(args.output),
            color_by="label",
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
