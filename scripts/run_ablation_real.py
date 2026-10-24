"""
Ablation Study on Real PostgreSQL Plans with MLflow Tracking.

Ablation Dimensions:
1. Architecture: GCN vs GAT vs PlanSAGE
2. Pooling: Attention Pooling vs Mean Pooling
3. Features: With PostgreSQL Cost Features (cont_dim=4) vs Without Cost Features (cont_dim=2)

Evaluation & Selection:
- Train: 8 TPC-H templates
- Validation: 2 TPC-H templates
- Test: 2 TPC-H templates
- Winner selection strictly on Validation Top-1 Accuracy and Validation Regret.
- Tracked in MLflow with SQLite backend (sqlite:///mlflow.db).
- Saves results to results/ablation_real.json and results/ablation_real.md.
- Exports winning model to checkpoints/best_real.pt and models/best_real.onnx.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PlanGraphDataset, collate_plan_graphs
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.model import PlanGAT, PlanGCN, PlanSAGE

logger = logging.getLogger("neural_optimizer")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    margin: float = 0.1,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x_op = batch["x_op"].to(device)
            x_cont = batch["x_cont"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch"].to(device)
            y = batch["y"].to(device)
            qids = batch["query_ids"]

            optimizer.zero_grad()
            scores = model(x_op, x_cont, edge_index, batch_idx)
            if isinstance(scores, tuple):
                scores = scores[0]

            qid_map: dict[Any, list[int]] = {}
            for idx, qid in enumerate(qids):
                qid_map.setdefault(qid, []).append(idx)

            loss = torch.tensor(0.0, device=device)
            pairs_count = 0
            for indices in qid_map.values():
                if len(indices) < 2:
                    continue
                n_items = len(indices)
                idx_i, idx_j = torch.triu_indices(n_items, n_items, offset=1)
                s_i, s_j = scores[indices][idx_i], scores[indices][idx_j]
                y_i, y_j = y[indices][idx_i], y[indices][idx_j]
                target = torch.where(y_i < y_j, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
                valid = (y_i != y_j)
                if valid.any():
                    pair_loss = nn.functional.margin_ranking_loss(s_j[valid], s_i[valid], target[valid], margin=margin)
                    loss = loss + pair_loss
                    pairs_count += 1

            if pairs_count > 0:
                loss = loss / pairs_count
                loss.backward()
                optimizer.step()

        # Val loss
        model.eval()
        val_loss = 0.0
        val_pairs = 0
        with torch.no_grad():
            for batch in val_loader:
                x_op = batch["x_op"].to(device)
                x_cont = batch["x_cont"].to(device)
                edge_index = batch["edge_index"].to(device)
                batch_idx = batch["batch"].to(device)
                y = batch["y"].to(device)
                qids = batch["query_ids"]

                scores = model(x_op, x_cont, edge_index, batch_idx)
                if isinstance(scores, tuple):
                    scores = scores[0]

                qid_map = {}
                for idx, qid in enumerate(qids):
                    qid_map.setdefault(qid, []).append(idx)

                for indices in qid_map.values():
                    if len(indices) < 2:
                        continue
                    n_items = len(indices)
                    idx_i, idx_j = torch.triu_indices(n_items, n_items, offset=1)
                    s_i, s_j = scores[indices][idx_i], scores[indices][idx_j]
                    y_i, y_j = y[indices][idx_i], y[indices][idx_j]
                    target = torch.where(y_i < y_j, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
                    valid = (y_i != y_j)
                    if valid.any():
                        v_loss = nn.functional.margin_ranking_loss(s_j[valid], s_i[valid], target[valid], margin=margin)
                        val_loss += v_loss.item()
                        val_pairs += 1

        avg_val = val_loss / max(1, val_pairs)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss}


@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_scores, all_trues, all_qids, all_defs = [], [], [], []

    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)

        scores = model(x_op, x_cont, edge_index, batch_idx)
        if isinstance(scores, tuple):
            scores = scores[0]

        all_scores.extend(scores.cpu().squeeze(-1).tolist())
        all_trues.extend(batch["y"].squeeze(-1).tolist())
        all_qids.extend(batch["query_ids"])
        for raw in batch.get("plans", []):
            all_defs.append(bool(raw.get("is_default_plan", False)))

    y_score = np.array(all_scores, dtype=np.float32)
    y_true = np.array(all_trues, dtype=np.float32)

    unique_qids = sorted(list(set(all_qids)))
    spearmans, kendalls, top1s, regrets, speedups = [], [], [], [], []

    for qid in unique_qids:
        indices = [i for i, q in enumerate(all_qids) if q == qid]
        if len(indices) < 2:
            continue
        q_true = y_true[indices]
        q_score = y_score[indices]
        q_defs = [all_defs[i] for i in indices]

        best_true = float(np.min(q_true))
        chosen_time = float(q_true[np.argmin(q_score)])
        def_time = float(q_true[q_defs.index(True)]) if any(q_defs) else float(q_true[0])

        top1s.append(1.0 if chosen_time <= best_true + 1e-4 else 0.0)
        regrets.append(chosen_time - best_true)
        speedups.append(def_time / max(1e-4, chosen_time))

        if np.all(q_score == q_score[0]):
            spearmans.append(0.0)
            kendalls.append(0.0)
        else:
            rho, _ = stats.spearmanr(q_score, q_true)
            tau, _ = stats.kendalltau(q_score, q_true)
            if not np.isnan(rho):
                spearmans.append(rho)
            if not np.isnan(tau):
                kendalls.append(tau)

    return {
        "spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "kendall": float(np.mean(kendalls)) if kendalls else 0.0,
        "top1_accuracy": float(np.mean(top1s)) if top1s else 0.0,
        "mean_regret_ms": float(np.mean(regrets)) if regrets else 0.0,
        "mean_speedup_vs_default": float(np.mean(speedups)) if speedups else 1.0,
    }


def export_model_to_onnx(model: nn.Module, cont_dim: int, out_path: Path):
    model.eval()
    dummy_op = torch.tensor([0, 1], dtype=torch.long)
    dummy_cont = torch.randn(2, cont_dim, dtype=torch.float32)
    dummy_edge = torch.tensor([[0], [1]], dtype=torch.long)
    dummy_batch = torch.zeros(2, dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_op, dummy_cont, dummy_edge, dummy_batch),
        str(out_path),
        input_names=["x_op", "x_cont", "edge_index", "batch"],
        output_names=["predicted_cost"],
        dynamic_axes={
            "x_op": {0: "num_nodes"},
            "x_cont": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "batch": {0: "num_nodes"},
            "predicted_cost": {0: "batch_size"},
        },
        dynamo=False,
    )
    logger.info("Exported winning model to ONNX: %s", out_path)


def run_ablation(data_dir: str = "data/plans_real", epochs: int = 40, quick: bool = False):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = OperatorVocabulary()

    # Discover templates and mapping
    template_to_qids: dict[int, set[str]] = {}
    for f in Path(data_dir).glob("*.json"):
        with open(f) as fp:
            d = json.load(fp)
            t_id = int(d.get("template_id", 0))
            q_id = str(d.get("query_id", ""))
            template_to_qids.setdefault(t_id, set()).add(q_id)

    templates = sorted(list(template_to_qids.keys()))
    # Fixed deterministic split:
    # Test: [6, 19] (1 single-table + 1 multi-join)
    # Val:  [12, 14]
    # Train: [2, 3, 4, 5, 7, 8, 9, 10]
    test_templates = [6, 19]
    val_templates = [12, 14]
    train_templates = [t for t in templates if t not in test_templates and t not in val_templates]

    train_qids = set().union(*[template_to_qids[t] for t in train_templates])
    val_qids = set().union(*[template_to_qids[t] for t in val_templates])
    test_qids = set().union(*[template_to_qids[t] for t in test_templates])

    logger.info("Ablation Split: Train=%s (%d queries), Val=%s (%d queries), Test=%s (%d queries)",
                train_templates, len(train_qids), val_templates, len(val_qids), test_templates, len(test_qids))

    # Grid definition
    architectures = ["GCN", "GAT", "SAGE"] if not quick else ["GCN", "GAT"]
    poolings = ["attention", "mean"] if not quick else ["attention"]
    feature_sets = [True, False] if not quick else [True]  # True = with PG cost (dim=4), False = without (dim=2)

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("real_tpch_cost_model_ablation")

    results_grid = []

    for arch in architectures:
        for pool in poolings:
            for with_cost in feature_sets:
                config_name = f"{arch}_{pool}_{'with_cost' if with_cost else 'no_cost'}"
                cont_dim = 4 if with_cost else 2

                train_ds = PlanGraphDataset(data_dir, vocab=vocab, allowed_query_ids=train_qids, include_cost=with_cost)
                val_ds = PlanGraphDataset(data_dir, vocab=vocab, allowed_query_ids=val_qids, include_cost=with_cost)
                test_ds = PlanGraphDataset(data_dir, vocab=vocab, allowed_query_ids=test_qids, include_cost=with_cost)

                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_plan_graphs)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_plan_graphs)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_plan_graphs)

                torch.manual_seed(42)
                if arch == "GCN":
                    model = PlanGCN(len(vocab), operator_embed_dim=16, cont_dim=cont_dim, hidden_dim=48, num_layers=2, pooling=pool).to(device)
                elif arch == "GAT":
                    model = PlanGAT(len(vocab), operator_embed_dim=16, cont_dim=cont_dim, hidden_dim=48, num_layers=2, pooling=pool).to(device)
                else:
                    model = PlanSAGE(len(vocab), operator_embed_dim=16, cont_dim=cont_dim, hidden_dim=48, num_layers=2, pooling=pool).to(device)

                with mlflow.start_run(run_name=config_name):
                    mlflow.log_params({
                        "architecture": arch,
                        "pooling": pool,
                        "include_cost_features": with_cost,
                        "continuous_dim": cont_dim,
                        "epochs": epochs if not quick else 15,
                    })

                    train_info = train_model(model, train_loader, val_loader, device, epochs=epochs if not quick else 15)

                    # Evaluate on Validation split (Used for Model Selection)
                    val_metrics = evaluate_split(model, val_loader, device)

                    # Evaluate on Test split (Held-out)
                    test_metrics = evaluate_split(model, test_loader, device)

                    mlflow.log_metrics({
                        "val_loss": train_info["best_val_loss"],
                        "val_top1": val_metrics["top1_accuracy"],
                        "val_regret_ms": val_metrics["mean_regret_ms"],
                        "val_spearman": val_metrics["spearman"],
                        "val_kendall": val_metrics["kendall"],
                        "test_top1": test_metrics["top1_accuracy"],
                        "test_regret_ms": test_metrics["mean_regret_ms"],
                        "test_spearman": test_metrics["spearman"],
                        "test_kendall": test_metrics["kendall"],
                    })

                row = {
                    "config_name": config_name,
                    "architecture": arch,
                    "pooling": pool,
                    "include_cost": with_cost,
                    "val_loss": round(train_info["best_val_loss"], 4),
                    "val_top1": round(val_metrics["top1_accuracy"], 4),
                    "val_regret_ms": round(val_metrics["mean_regret_ms"], 2),
                    "val_spearman": round(val_metrics["spearman"], 4),
                    "val_kendall": round(val_metrics["kendall"], 4),
                    "val_speedup_vs_default": round(val_metrics["mean_speedup_vs_default"], 3),
                    "test_top1": round(test_metrics["top1_accuracy"], 4),
                    "test_regret_ms": round(test_metrics["mean_regret_ms"], 2),
                    "test_spearman": round(test_metrics["spearman"], 4),
                    "test_kendall": round(test_metrics["kendall"], 4),
                    "test_speedup_vs_default": round(test_metrics["mean_speedup_vs_default"], 3),
                    "model_state": copy.deepcopy(model.state_dict()),
                    "model_obj": model,
                    "cont_dim": cont_dim,
                }
                results_grid.append(row)
                logger.info("Config %s: Val Top-1=%.1f%%, Val Regret=%.2fms, Val Spearman=%.3f",
                            config_name, row["val_top1"] * 100, row["val_regret_ms"], row["val_spearman"])

    elapsed = time.time() - t0

    # SELECT WINNER STRICTLY ON VALIDATION PERFORMANCE (Val Top-1 DESC, Val Regret ASC, Val Spearman DESC)
    results_grid.sort(key=lambda r: (r["val_top1"], -r["val_regret_ms"], r["val_spearman"]), reverse=True)
    winner = results_grid[0]
    logger.info("Winning configuration (chosen strictly on Validation): %s (Val Top-1=%.1f%%, Val Regret=%.2fms)",
                winner["config_name"], winner["val_top1"] * 100, winner["val_regret_ms"])

    # Export best model
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config_name": winner["config_name"],
        "architecture": winner["architecture"],
        "pooling": winner["pooling"],
        "include_cost": winner["include_cost"],
        "cont_dim": winner["cont_dim"],
        "model_state_dict": winner["model_state"],
        "winner_val_metrics": {
            "top1": winner["val_top1"],
            "regret_ms": winner["val_regret_ms"],
            "spearman": winner["val_spearman"],
        },
    }, ckpt_dir / "best_real.pt")

    # Export to ONNX
    onnx_path = Path("models/best_real.onnx")
    export_model_to_onnx(winner["model_obj"], winner["cont_dim"], onnx_path)

    # Save summary tables
    clean_grid = []
    for r in results_grid:
        c = dict(r)
        del c["model_state"]
        del c["model_obj"]
        clean_grid.append(c)

    output = {
        "selection_rule": "Highest Validation Top-1 Accuracy, lowest Validation Regret ms (strict validation selection)",
        "elapsed_seconds": round(elapsed, 2),
        "winner": {
            "config_name": winner["config_name"],
            "architecture": winner["architecture"],
            "pooling": winner["pooling"],
            "include_cost": winner["include_cost"],
            "val_top1": winner["val_top1"],
            "val_regret_ms": winner["val_regret_ms"],
            "val_spearman": winner["val_spearman"],
            "test_top1": winner["test_top1"],
            "test_regret_ms": winner["test_regret_ms"],
            "test_spearman": winner["test_spearman"],
        },
        "grid": clean_grid,
    }

    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "ablation_real.json", "w") as fp:
        json.dump(output, fp, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Real PostgreSQL TPC-H Ablation Study Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Duration:** {elapsed:.2f}s  ",
        f"**Selection Criterion:** Winner selected strictly on Validation Top-1 and Regret (No test set leakage)  ",
        f"**Winning Configuration:** `{winner['config_name']}`  ",
        "",
        "| Architecture | Pooling | Cost Features? | Val Top-1 (%) | Val Regret (ms) | Val Spearman | Val Speedup vs Def | Test Top-1 (%) | Test Regret (ms) | Test Spearman |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in clean_grid:
        is_win = "**" if r["config_name"] == winner["config_name"] else ""
        cost_str = "Yes (dim=4)" if r["include_cost"] else "No (dim=2)"
        md_lines.append(
            f"| {is_win}{r['architecture']}{is_win} | {r['pooling']} | {cost_str} | "
            f"{r['val_top1']*100:.1f}% | {r['val_regret_ms']:.2f} ms | {r['val_spearman']:.3f} | {r['val_speedup_vs_default']:.3f}x | "
            f"{r['test_top1']*100:.1f}% | {r['test_regret_ms']:.2f} ms | {r['test_spearman']:.3f} |"
        )

    with open(res_dir / "ablation_real.md", "w") as fp:
        fp.write("\n".join(md_lines) + "\n")

    logger.info("Ablation complete! Results saved to %s", res_dir)
    return output


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run real GNN architecture & feature ablation study.")
    parser.add_argument("--data", default="data/plans_real")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--quick", action="store_true", help="Quick mode for ablation")
    args = parser.parse_args()

    run_ablation(data_dir=args.data, epochs=args.epochs, quick=args.quick)


if __name__ == "__main__":
    main()
