"""
Nested Leave-One-Template-Out Cross-Validation on Real PostgreSQL Plans.

Implements rigorous evaluation protocol:
1. Outer Cross-Validation: Leave-One-Template-Out (12 folds).
   All instances and hint variants of each TPC-H template are kept in the same fold.
2. Inner Split: Remaining 11 templates split into Inner Train (9 templates) and Inner Val (2 templates).
   Affine calibration probe (y_cal = a * y_hat + b) fit strictly on Inner Val.
   Model checkpoint early-stopped on Inner Val pairwise ranking accuracy.
3. Plain EXPLAIN features only.
4. Metric suite:
   - Within-query Spearman rank correlation
   - Within-query Kendall's tau rank correlation
   - Top-1 plan selection accuracy
   - Mean & median regret (ms)
   - Runtime of GNN plan vs PostgreSQL default plan vs true best plan (speedup & slowdown)
   - Calibrated MAE, RMSE, and R2 (affine fit on val only, preserves plan ranking)
5. Baselines evaluated on identical folds:
   - PostgreSQL Native Cost (EXPLAIN total_cost)
   - Flat GBDT (HistGradientBoosting on tree summary features)
   - Random Plan Selection
6. 3-seed evaluation (seeds 42, 43, 44) reporting mean +/- std.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PlanGraphDataset, collate_plan_graphs
from src.feature_encoder import OperatorVocabulary
from src.logging_config import setup_logging
from src.model import PlanGCN

logger = logging.getLogger("neural_optimizer")


def train_ranking_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 40,
    margin: float = 0.1,
) -> dict[str, Any]:
    """
    Train model using pairwise margin ranking loss with early stopping on validation loss.
    """
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

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

            # Pairwise ranking loss within each query
            loss = torch.tensor(0.0, device=device)
            pairs_count = 0

            # Group indices by query_id
            qid_map: dict[Any, list[int]] = {}
            for idx, qid in enumerate(qids):
                qid_map.setdefault(qid, []).append(idx)

            for indices in qid_map.values():
                if len(indices) < 2:
                    continue
                sub_scores = scores[indices]
                sub_y = y[indices]

                # Generate all pairs (i, j)
                n_items = len(indices)
                idx_i, idx_j = torch.triu_indices(n_items, n_items, offset=1)
                s_i, s_j = sub_scores[idx_i], sub_scores[idx_j]
                y_i, y_j = sub_y[idx_i], sub_y[idx_j]

                # target = 1 if y_i < y_j (plan i faster -> lower score predicted)
                # target = -1 if y_i > y_j (plan i slower -> higher score predicted)
                # MarginRankingLoss: loss(s_i, s_j, target) = max(0, -target * (s_j - s_i) + margin)
                # If target=1 (i faster): want s_i < s_j, i.e. s_j - s_i > margin
                target = torch.where(y_i < y_j, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
                valid_mask = (y_i != y_j)
                if valid_mask.any():
                    pair_loss = nn.functional.margin_ranking_loss(
                        s_j[valid_mask],
                        s_i[valid_mask],
                        target[valid_mask],
                        margin=margin,
                    )
                    loss = loss + pair_loss
                    pairs_count += 1

            if pairs_count > 0:
                loss = loss / pairs_count
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        # Validation loss
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

        avg_val_loss = val_loss / max(1, val_pairs)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss}


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_trues, all_qids, all_costs, all_flats, all_defaults = [], [], [], [], [], []

    for batch in loader:
        x_op = batch["x_op"].to(device)
        x_cont = batch["x_cont"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_idx = batch["batch"].to(device)

        scores = model(x_op, x_cont, edge_index, batch_idx)
        if isinstance(scores, tuple):
            scores = scores[0]

        all_preds.extend(scores.cpu().squeeze(-1).tolist())
        all_trues.extend(batch["y"].squeeze(-1).tolist())
        all_qids.extend(batch["query_ids"])
        all_costs.extend(batch["root_cost"].squeeze(-1).tolist())
        all_flats.append(batch["flat_features"].numpy())

        for raw in batch.get("plans", []):
            all_defaults.append(bool(raw.get("is_default_plan", False)))

    return (
        np.array(all_preds, dtype=np.float32),
        np.array(all_trues, dtype=np.float32),
        all_qids,
        np.array(all_costs, dtype=np.float32),
        np.concatenate(all_flats, axis=0) if all_flats else np.zeros((0,)),
        all_defaults,
    )


def compute_metrics_for_ranking(
    query_ids: list[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    is_default: list[bool],
    is_cost: bool = False,
) -> dict[str, Any]:
    """
    Compute within-query ranking metrics and plan selection speedups.
    For GNN and GBDT: lower score = faster plan.
    For PostgreSQL total_cost: lower cost = faster plan.
    """
    unique_qids = sorted(list(set(query_ids)))
    spearmans = []
    kendalls = []
    top1_matches = []
    regrets_ms = []
    regret_ratios = []
    speedup_vs_defaults = []

    per_query_details = {}

    for qid in unique_qids:
        indices = [i for i, q in enumerate(query_ids) if q == qid]
        if len(indices) < 2:
            continue

        q_true = y_true[indices]
        q_score = scores[indices]
        q_defaults = [is_default[i] for i in indices]

        best_idx_true = np.argmin(q_true)
        best_time_true = q_true[best_idx_true]

        # Chosen plan by predictor
        chosen_sub_idx = np.argmin(q_score)
        chosen_time = q_true[chosen_sub_idx]

        # Default plan (hint == 'default')
        if any(q_defaults):
            def_idx = q_defaults.index(True)
            default_time = q_true[def_idx]
        else:
            default_time = q_true[0]

        # Top-1 accuracy
        is_top1 = bool(chosen_time <= (best_time_true + 1e-4))
        top1_matches.append(1.0 if is_top1 else 0.0)

        # Regret
        regret = float(chosen_time - best_time_true)
        regrets_ms.append(regret)
        regret_ratio = float(chosen_time / max(1e-4, best_time_true))
        regret_ratios.append(regret_ratio)

        # Speedup vs PostgreSQL default
        speedup = float(default_time / max(1e-4, chosen_time))
        speedup_vs_defaults.append(speedup)

        # Correlations
        # Spearman: lower score should correspond to lower runtime, so correlation between rank(score) and rank(true)
        rho, _ = stats.spearmanr(q_score, q_true)
        tau, _ = stats.kendalltau(q_score, q_true)
        if not np.isnan(rho):
            spearmans.append(rho)
        if not np.isnan(tau):
            kendalls.append(tau)

        per_query_details[qid] = {
            "num_plans": len(indices),
            "true_best_ms": round(float(best_time_true), 2),
            "chosen_time_ms": round(float(chosen_time), 2),
            "default_time_ms": round(float(default_time), 2),
            "is_top1": is_top1,
            "regret_ms": round(regret, 2),
            "speedup_vs_default": round(speedup, 3),
            "spearman": round(float(rho), 3) if not np.isnan(rho) else 0.0,
            "kendall": round(float(tau), 3) if not np.isnan(tau) else 0.0,
        }

    return {
        "spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "kendall": float(np.mean(kendalls)) if kendalls else 0.0,
        "top1_accuracy": float(np.mean(top1_matches)) if top1_matches else 0.0,
        "mean_regret_ms": float(np.mean(regrets_ms)) if regrets_ms else 0.0,
        "median_regret_ms": float(np.median(regrets_ms)) if regrets_ms else 0.0,
        "mean_regret_ratio": float(np.mean(regret_ratios)) if regret_ratios else 1.0,
        "mean_speedup_vs_default": float(np.mean(speedup_vs_defaults)) if speedup_vs_defaults else 1.0,
        "per_query": per_query_details,
    }


def fit_affine_calibration(y_pred_val: np.ndarray, y_true_val: np.ndarray) -> tuple[float, float]:
    """
    Fit linear regression y = a * y_pred + b strictly on validation data.
    Preserves rank order if a > 0.
    """
    if len(y_pred_val) < 2 or np.all(y_pred_val == y_pred_val[0]):
        return 1.0, float(np.mean(y_true_val))

    var_pred = float(np.var(y_pred_val))
    if var_pred < 1e-8:
        return 1.0, float(np.mean(y_true_val) - np.mean(y_pred_val))

    cov = float(np.cov(y_pred_val, y_true_val)[0, 1])
    a = cov / var_pred
    if a <= 0:
        a = 1.0
    b = float(np.mean(y_true_val) - a * np.mean(y_pred_val))
    return a, b


def run_nested_cross_validation(
    data_dir: str = "data/plans_real",
    include_cost: bool = True,
    seeds: list[int] = [42, 43, 44],
    epochs: int = 40,
    quick: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = OperatorVocabulary()

    # Load all real plans
    ds_all = PlanGraphDataset(data_dir, vocab=vocab, include_cost=include_cost)
    logger.info("Loaded %d real plans from %s (include_cost=%s, device=%s)", len(ds_all), data_dir, include_cost, device)

    # Discover templates and mapping
    template_to_qids: dict[int, set[str]] = {}
    qid_to_template: dict[str, int] = {}
    all_files = list(Path(data_dir).glob("*.json"))
    for f in all_files:
        with open(f) as fp:
            d = json.load(fp)
            t_id = int(d.get("template_id", 0))
            q_id = str(d.get("query_id", ""))
            template_to_qids.setdefault(t_id, set()).add(q_id)
            qid_to_template[q_id] = t_id

    templates = sorted(list(template_to_qids.keys()))
    if quick:
        templates = templates[:4]
        seeds = [42]
        epochs = 15

    logger.info("Evaluating %d templates across seeds %s with Leave-One-Template-Out CV", len(templates), seeds)

    seed_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        fold_gnn_metrics = []
        fold_native_metrics = []
        fold_gbdt_metrics = []
        fold_random_metrics = []
        fold_calibrated_regr = []

        per_template_summary = {}

        for test_idx, test_template in enumerate(templates):
            # Outer Test queries
            test_qids = template_to_qids[test_template]
            outer_train_templates = [t for t in templates if t != test_template]

            # Inner Split: 2 templates for validation, remainder for train
            val_idx1 = (test_idx) % len(outer_train_templates)
            val_idx2 = (test_idx + 1) % len(outer_train_templates)
            val_templates = {outer_train_templates[val_idx1], outer_train_templates[val_idx2]}
            train_templates = [t for t in outer_train_templates if t not in val_templates]

            train_qids = set().union(*[template_to_qids[t] for t in train_templates])
            val_qids = set().union(*[template_to_qids[t] for t in val_templates])

            # Datasets
            train_ds = PlanGraphDataset(data_dir, vocab=vocab, allowed_query_ids=train_qids, include_cost=include_cost)
            val_ds = PlanGraphDataset(data_dir, vocab=vocab, allowed_query_ids=val_qids, include_cost=include_cost)
            test_ds = PlanGraphDataset(data_dir, vocab=vocab, allowed_query_ids=test_qids, include_cost=include_cost)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_plan_graphs)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_plan_graphs)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_plan_graphs)

            # Model
            cont_dim = 4 if include_cost else 2
            model = PlanGCN(
                num_operators=len(vocab),
                operator_embed_dim=16,
                cont_dim=cont_dim,
                hidden_dim=48,
                num_layers=2,
                pooling="attention",
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Train on inner train, early stop on inner val
            train_ranking_model(model, train_loader, val_loader, optimizer, device, epochs=epochs)

            # Fit affine calibration on Inner Validation
            val_preds, val_trues, _, _, _, _ = predict_dataset(model, val_loader, device)
            cal_a, cal_b = fit_affine_calibration(val_preds, val_trues)

            # Evaluate on Outer Test
            test_preds, test_trues, test_q_list, test_costs, test_flats, test_defs = predict_dataset(model, test_loader, device)

            # Calibrated regression metrics
            test_preds_cal = cal_a * test_preds + cal_b
            mae = float(np.mean(np.abs(test_preds_cal - test_trues)))
            rmse = float(np.sqrt(np.mean((test_preds_cal - test_trues) ** 2)))
            ss_tot = float(np.sum((test_trues - np.mean(test_trues)) ** 2))
            ss_res = float(np.sum((test_trues - test_preds_cal) ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 0.0

            fold_calibrated_regr.append({"mae": mae, "rmse": rmse, "r2": r2})

            # GNN Ranking Metrics
            gnn_m = compute_metrics_for_ranking(test_q_list, test_trues, test_preds, test_defs)
            fold_gnn_metrics.append(gnn_m)

            # PostgreSQL Native Cost Ranking Metrics
            native_m = compute_metrics_for_ranking(test_q_list, test_trues, test_costs, test_defs, is_cost=True)
            fold_native_metrics.append(native_m)

            # Flat GBDT Baseline (trained on inner train flat features)
            train_preds_raw, train_trues_raw, _, _, train_flats, _ = predict_dataset(model, train_loader, device)
            gbdt = HistGradientBoostingRegressor(max_iter=60, random_state=seed)
            gbdt.fit(train_flats, train_trues_raw)
            gbdt_preds = gbdt.predict(test_flats)
            gbdt_m = compute_metrics_for_ranking(test_q_list, test_trues, gbdt_preds, test_defs)
            fold_gbdt_metrics.append(gbdt_m)

            # Random selection baseline
            np.random.seed(seed + test_idx)
            rand_scores = np.random.rand(len(test_trues))
            rand_m = compute_metrics_for_ranking(test_q_list, test_trues, rand_scores, test_defs)
            fold_random_metrics.append(rand_m)

            per_template_summary[f"T{test_template}"] = {
                "top1_gnn": gnn_m["top1_accuracy"],
                "top1_native": native_m["top1_accuracy"],
                "top1_gbdt": gbdt_m["top1_accuracy"],
                "speedup_vs_default_gnn": gnn_m["mean_speedup_vs_default"],
                "speedup_vs_default_native": native_m["mean_speedup_vs_default"],
                "regret_ms_gnn": gnn_m["mean_regret_ms"],
                "regret_ms_native": native_m["mean_regret_ms"],
                "spearman_gnn": gnn_m["spearman"],
                "kendall_gnn": gnn_m["kendall"],
            }

        # Aggregate across folds for this seed
        def aggregate_folds(metric_list: list[dict[str, Any]]) -> dict[str, float]:
            return {
                "top1_accuracy": float(np.mean([m["top1_accuracy"] for m in metric_list])),
                "mean_regret_ms": float(np.mean([m["mean_regret_ms"] for m in metric_list])),
                "median_regret_ms": float(np.mean([m["median_regret_ms"] for m in metric_list])),
                "mean_speedup_vs_default": float(np.mean([m["mean_speedup_vs_default"] for m in metric_list])),
                "spearman": float(np.mean([m["spearman"] for m in metric_list])),
                "kendall": float(np.mean([m["kendall"] for m in metric_list])),
            }

        seed_results.append({
            "seed": seed,
            "gnn": aggregate_folds(fold_gnn_metrics),
            "native": aggregate_folds(fold_native_metrics),
            "gbdt": aggregate_folds(fold_gbdt_metrics),
            "random": aggregate_folds(fold_random_metrics),
            "calibrated_regr": {
                "mae": float(np.mean([c["mae"] for c in fold_calibrated_regr])),
                "rmse": float(np.mean([c["rmse"] for c in fold_calibrated_regr])),
                "r2": float(np.mean([c["r2"] for c in fold_calibrated_regr])),
            },
            "per_template": per_template_summary,
        })

    elapsed = time.time() - t0

    # Headline 3-seed Mean +/- Std
    def mean_std(key: str, subkey: str) -> dict[str, float]:
        vals = [s[key][subkey] for s in seed_results]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    headline = {
        "top1_accuracy_gnn": mean_std("gnn", "top1_accuracy"),
        "top1_accuracy_native": mean_std("native", "top1_accuracy"),
        "top1_accuracy_gbdt": mean_std("gbdt", "top1_accuracy"),
        "top1_accuracy_random": mean_std("random", "top1_accuracy"),
        "spearman_gnn": mean_std("gnn", "spearman"),
        "spearman_native": mean_std("native", "spearman"),
        "spearman_gbdt": mean_std("gbdt", "spearman"),
        "kendall_gnn": mean_std("gnn", "kendall"),
        "kendall_native": mean_std("native", "kendall"),
        "kendall_gbdt": mean_std("gbdt", "kendall"),
        "mean_regret_ms_gnn": mean_std("gnn", "mean_regret_ms"),
        "mean_regret_ms_native": mean_std("native", "mean_regret_ms"),
        "mean_regret_ms_gbdt": mean_std("gbdt", "mean_regret_ms"),
        "mean_speedup_vs_default_gnn": mean_std("gnn", "mean_speedup_vs_default"),
        "mean_speedup_vs_default_native": mean_std("native", "mean_speedup_vs_default"),
        "mean_speedup_vs_default_gbdt": mean_std("gbdt", "mean_speedup_vs_default"),
        "calibrated_mae": mean_std("calibrated_regr", "mae"),
        "calibrated_rmse": mean_std("calibrated_regr", "rmse"),
        "calibrated_r2": mean_std("calibrated_regr", "r2"),
    }

    # Relative improvements
    gnn_top1 = headline["top1_accuracy_gnn"]["mean"]
    native_top1 = headline["top1_accuracy_native"]["mean"]
    gbdt_top1 = headline["top1_accuracy_gbdt"]["mean"]
    top1_gain_vs_native = ((gnn_top1 - native_top1) / max(1e-4, native_top1)) * 100.0
    top1_gain_vs_gbdt = ((gnn_top1 - gbdt_top1) / max(1e-4, gbdt_top1)) * 100.0

    gnn_regret = headline["mean_regret_ms_gnn"]["mean"]
    native_regret = headline["mean_regret_ms_native"]["mean"]
    regret_reduction_vs_native = ((native_regret - gnn_regret) / max(1e-4, native_regret)) * 100.0

    full_output = {
        "evaluation_protocol": "Leave-One-Template-Out Nested Cross-Validation",
        "num_templates": len(templates),
        "templates_list": templates,
        "include_cost_features": include_cost,
        "seeds": seeds,
        "elapsed_seconds": round(elapsed, 2),
        "headline_3seed_mean_std": headline,
        "relative_improvements": {
            "top1_gain_pct_vs_postgres_native": round(top1_gain_vs_native, 2),
            "top1_gain_pct_vs_gbdt": round(top1_gain_vs_gbdt, 2),
            "regret_reduction_pct_vs_postgres_native": round(regret_reduction_vs_native, 2),
        },
        "per_seed_runs": seed_results,
    }

    out_file = Path("results/evaluation_real_no_cost.json") if not include_cost else Path("results/evaluation_real.json")
    with open(out_file, "w") as fp:
        json.dump(full_output, fp, indent=2)

    logger.info("Evaluation complete in %.2fs! Results saved to %s", elapsed, out_file)
    return full_output


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate GNN and baselines on real PostgreSQL TPC-H data.")
    parser.add_argument("--data", default="data/plans_real")
    parser.add_argument("--no_cost", action="store_true", help="Exclude PostgreSQL cost features (cont_dim=2)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (4 templates, 1 seed, 15 epochs)")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    results = run_nested_cross_validation(
        data_dir=args.data,
        include_cost=not args.no_cost,
        epochs=args.epochs,
        quick=args.quick,
    )

    h = results["headline_3seed_mean_std"]
    print("\n" + "=" * 60)
    print("=== REAL PostgreSQL TPC-H Evaluation Results (Nested CV) ===")
    print("=" * 60)
    print(f"Top-1 Selection Accuracy:  GNN: {h['top1_accuracy_gnn']['mean']*100:.1f}% ± {h['top1_accuracy_gnn']['std']*100:.1f}%")
    print(f"                           Postgres Native: {h['top1_accuracy_native']['mean']*100:.1f}% ± {h['top1_accuracy_native']['std']*100:.1f}%")
    print(f"                           Flat GBDT:       {h['top1_accuracy_gbdt']['mean']*100:.1f}% ± {h['top1_accuracy_gbdt']['std']*100:.1f}%")
    print(f"                           Random:          {h['top1_accuracy_random']['mean']*100:.1f}% ± {h['top1_accuracy_random']['std']*100:.1f}%")
    print("-" * 60)
    print(f"Spearman Rank Correlation: GNN: {h['spearman_gnn']['mean']:.3f} ± {h['spearman_gnn']['std']:.3f}")
    print(f"                           Postgres Native: {h['spearman_native']['mean']:.3f} ± {h['spearman_native']['std']:.3f}")
    print(f"                           Flat GBDT:       {h['spearman_gbdt']['mean']:.3f} ± {h['spearman_gbdt']['std']:.3f}")
    print("-" * 60)
    print(f"Kendall's tau:             GNN: {h['kendall_gnn']['mean']:.3f} ± {h['kendall_gnn']['std']:.3f}")
    print(f"                           Postgres Native: {h['kendall_native']['mean']:.3f} ± {h['kendall_native']['std']:.3f}")
    print(f"                           Flat GBDT:       {h['kendall_gbdt']['mean']:.3f} ± {h['kendall_gbdt']['std']:.3f}")
    print("-" * 60)
    print(f"Mean Regret (ms):          GNN: {h['mean_regret_ms_gnn']['mean']:.2f} ms ± {h['mean_regret_ms_gnn']['std']:.2f} ms")
    print(f"                           Postgres Native: {h['mean_regret_ms_native']['mean']:.2f} ms ± {h['mean_regret_ms_native']['std']:.2f} ms")
    print(f"                           Flat GBDT:       {h['mean_regret_ms_gbdt']['mean']:.2f} ms ± {h['mean_regret_ms_gbdt']['std']:.2f} ms")
    print("-" * 60)
    print(f"Speedup vs PG Default:     GNN: {h['mean_speedup_vs_default_gnn']['mean']:.3f}x")
    print(f"                           Postgres Native: {h['mean_speedup_vs_default_native']['mean']:.3f}x")
    print(f"                           Flat GBDT:       {h['mean_speedup_vs_default_gbdt']['mean']:.3f}x")
    print("-" * 60)
    print(f"Calibrated Probe (Val fit): MAE={h['calibrated_mae']['mean']:.2f} ms, RMSE={h['calibrated_rmse']['mean']:.2f} ms, R2={h['calibrated_r2']['mean']:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
