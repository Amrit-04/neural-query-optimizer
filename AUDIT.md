# Neural Query Optimizer: Repository Audit Report

**Date:** 2026-10-23  
**Repository:** [Amrit-04/neural-query-optimizer](https://github.com/Amrit-04/neural-query-optimizer)  
**Auditor:** Antigravity AI Pair Programmer  

---

## 1. Executive Summary

This audit assesses the code, experiments, artifacts, and documentation in `neural-query-optimizer` against the four key resume claims and empirical rigor requirements. 

**Key Verdict:**
- The repository contains an initial proof-of-concept GNN implementation (GCN and GAT) with tree-to-graph parsing and attention pooling.
- **Critical Flaw 1 (Data Origin & Resume Mismatch):** The resume claims that the pipeline achieved **0.659 Spearman** and **0.468 Kendall's tau** on **PostgreSQL EXPLAIN plans (TPC-H)**. In reality, these exact numbers (`outputs/metrics.json`) were obtained exclusively from **100 randomly generated synthetic plans**, not TPC-H. In fact, `README.md` explicitly lists `[ ] Evaluation on TPC-H benchmarks` as an uncompleted item under *Future Directions* (Line 349).
- **Critical Flaw 2 (34% Top-1 Improvement Claim):** The claim *"Improves top-1 join-plan selection by 34% over heuristics"* is entirely unsupported. The number `34%` does not exist in any result file, script, config, or documentation in the repository. No baseline (PostgreSQL native cost or gradient boosting) is implemented.
- **Critical Flaw 3 (MLflow & ONNX Missing):** The resume claims *"GCN vs GAT ablations tracked in MLflow; best model exported to ONNX Runtime with sub-8ms p95 CPU inference"*. MLflow is neither installed nor referenced in any script. ONNX export and ONNX Runtime latency benchmarking scripts are completely missing.
- **Critical Flaw 4 (Evaluation Methodology):** Spearman and Kendall correlations were computed **pooled across all queries** (inflating scores due to cross-query scale differences), the train/validation split was done **per-plan** rather than **per-query** (causing data leakage across splits), and ground-truth runtime was measured without repeats or medians.
- **Critical Flaw 5 (Pairwise Ranking Loss Inversion):** The pairwise ranking loss implemented in `scripts/train.py` has an inverted target sign in `F.margin_ranking_loss`, penalizing correct cost rankings. Furthermore, default training uses regression (MSE), not ranking loss.
- **Critical Flaw 6 (Input Feature Leakage):** The node feature encoder in `src/feature_encoder.py` includes `actual_rows` (ground truth from execution) as an input continuous feature to the GNN, which would not be available at query planning time.

---

## 2. Audit Checklist

| Item | Status | File and Line References | Findings & Details |
|---|---|---|---|
| **Model & Training** | | | |
| Pairwise ranking loss implemented & used | **Partial / Flawed** | `scripts/train.py:73-108`<br>`config/default.yaml:40` | Implemented in `train_epoch_ranking()`, but **inverted sign bug** in `F.margin_ranking_loss`: with `target=1`, PyTorch penalizes `pred_fast < pred_slow`, rewarding backwards rankings. Default mode in config is `regression`. Saved `best.pt` was trained on MSE regression. RankNet is not implemented. |
| GCN, GAT, and GraphSAGE variants | **Partial** | `src/model.py:37-115` (GCN)<br>`src/model.py:117-203` (GAT) | GCN (`PlanGCN`) and GAT (`PlanGAT`) are implemented using PyG. GraphSAGE (`SAGEConv`) is **completely missing**. |
| Attention pooling implemented | **Done** | `src/model.py:21-35` | `global_attention_pool` implemented using a linear gate and per-graph softmax normalization. Mean and add pooling are also implemented. |
| Operator embeddings | **Done** | `src/feature_encoder.py:17-37`<br>`config/default.yaml:24`<br>`src/model.py:52, 133` | 19 operator types (scans, joins, utilities). Embedding dimension is 16 (`operator_embedding_dim: 16`), concatenated with projected continuous features. |
| Numeric features log-scaled & standardized | **Partial / Flawed** | `src/feature_encoder.py:72-88, 113-119` | Log-scaling applied to `plan_rows` and `actual_rows`, but NOT to `startup_cost`, `total_cost`, or `plan_width`. Zero-mean unit-variance standardization is **missing**. Critical leakage: `actual_rows` is passed into input features. |
| **Data & Evaluation** | | | |
| TPC-H dataset & plan extraction | **Partial / Missing** | `scripts/extract_plans.py:147-202`<br>`config/default.yaml:9-20` | Extraction script exists with 4 hardcoded query templates (Q1, Q3, Q5, Q6) and 3 planner hint configurations. However, **no TPC-H data exists in the repo**; only 100 synthetic plans exist in `data/plans_synthetic/`. |
| Plan execution measurement rigor | **Missing** | `scripts/extract_plans.py:40-60` | Plans are executed only once. No repeats (not 3x), no median aggregation, no warm-up run, and no `statement_timeout` to guard against runaway executions. |
| Train / test split by query | **Missing / Leakage** | `scripts/train.py:170-183` | Splits randomly across **individual plans** via `random_split()`. Plans of the same query appear in both train and validation sets (data leakage). No test set exists. |
| Per-query ranking evaluation | **Missing / Flawed** | `scripts/evaluate.py:71-77` | Spearman and Kendall tau are computed **pooled across all plans/queries**, which inflates rank correlation. Within-query rank correlation is not computed. |
| Dataset size & plan counts | **Discrepancy** | `data/plans_synthetic/`<br>`README.md:58, 95` | `README.md` claims "Evaluation on 200 synthetic query plans" and "Synthetic training data (200 plans)". In reality, `data/plans_synthetic/` contains only **100 plans** (20 queries, 5 plans each). |
| Plan deduplication | **Missing** | `scripts/extract_plans.py`<br>`scripts/generate_synthetic_plans.py` | No plan tree hashing or AST comparison to eliminate identical plan trees generated under different planner flags. |
| Baselines (PostgreSQL native & Flat GBDT) | **Missing** | `scripts/evaluate.py` | No native cost baseline evaluation. No flat feature baseline (LightGBM/XGBoost/RandomForest). |
| 34% Top-1 selection claim origin | **Missing** | Entire repository | Completely unbacked. 34% appears nowhere in results or code. |
| Multiple random seeds | **Missing** | `scripts/train.py:151`<br>`config/default.yaml:5` | Only a single fixed seed (42) is used. No multi-seed runner or variance reporting. |
| **Ablations & Tracking** | | | |
| Ablation execution script | **Missing** | `scripts/visualize.py:93-118` | Only a markdown table formatter exists (`visualize.py ablation --results ...`). No automated runner executes an ablation grid across layer type, pooling, or embedding dimension. |
| MLflow experiment tracking | **Missing** | Entire repository | MLflow is not in `requirements.txt`, not imported, and not used anywhere. |
| **Deployment** | | | |
| ONNX model export | **Missing** | Entire repository | No export script exists. Attention pooling uses dynamic loops over batch masks that require specialized ONNX-friendly export handling. |
| CPU latency benchmark (p50/p95/p99) | **Missing** | Entire repository | No benchmarking script or ONNX Runtime inference harness exists. The "sub-8ms p95 CPU inference" claim has no supporting data. |
| **Documentation & Consistency** | | | |
| README vs result files | **Partial** | `README.md:60-68`<br>`outputs/metrics.json` | README numbers match `outputs/metrics.json` (MAE: 452.13, RMSE: 561.10, R2: 0.286, Spearman: 0.659, Kendall: 0.468, Top-1: 50.0%), but all originate from 100 synthetic plans evaluated on the whole dataset. |
| EXPLANATION.md numbers | **Missing / Fabricated** | `EXPLANATION.md:43-47` | Reports MAE: 438.95, RMSE: 531.47, R2: 0.36, Spearman: 0.62. These numbers match no file or artifact in the repository. |
| Code defects / crashing scripts | **Broken** | `scripts/analyze_errors.py:61, 73, 92` | Crashes immediately on execution (`batch["x_attr"]` key error, `batch["plan"]` key error, and `plan["Plan"]` casing mismatch). |

---

## 3. Resume Claims vs. Reality

| # | Resume Claim | Code / Data Evidence | Verdict |
|---|---|---|---|
| 1 | *Learned query cost estimator in PyTorch using GCN and GAT, with attention pooling and pairwise ranking loss.* | GCN, GAT, and attention pooling are implemented in `src/model.py`. Pairwise ranking loss is implemented in `scripts/train.py:100` but has an **inverted target sign bug** and is **disabled by default** (`config/default.yaml:40`). The checkpoint `best.pt` was trained with MSE regression. | **Partly Supported** (Architecture implemented, but ranking loss bugged & unused in checkpoint) |
| 2 | *Improves top-1 join-plan selection by 34% over heuristics.* | **Zero evidence.** The number `34%` does not appear anywhere in the codebase. No heuristic or PostgreSQL baseline is computed. Top-1 accuracy in `metrics.json` is 50.0% on synthetic plans without any baseline comparison. | **Not Supported** |
| 3 | *Tree-to-graph pipeline converting PostgreSQL EXPLAIN plans (TPC-H) into node/edge tensors, 0.659 Spearman and 0.468 Kendall's tau on plan ranking.* | Tree-to-graph parser exists (`src/graph_builder.py`). However, the **0.659 Spearman and 0.468 Kendall** metrics were generated on **synthetic random trees**, NOT TPC-H PostgreSQL plans. Furthermore, the correlation was computed **pooled across queries**, not per-query. TPC-H evaluation is listed as uncompleted in `README.md:349`. | **Partly Supported** (Pipeline exists; metrics are real but come from synthetic data, not TPC-H, and are pooled) |
| 4 | *GCN vs GAT ablations tracked in MLflow; best model exported to ONNX Runtime with sub-8ms p95 CPU inference per query plan.* | MLflow is not installed or integrated anywhere. No ONNX export script or ONNX model exists. No latency benchmark script exists. Sub-8ms p95 figure is unmeasured. | **Not Supported** |

---

## 4. Script Runtime Estimates & Time Budget Analysis

| Script | Current Defaults | Estimated Duration | 15-min Stage Limit | Action / Recommendation |
|---|---|---|---|---|
| `generate_synthetic_plans.py` | `--num 200` | ~1.5 seconds | **PASS** | Add structured queries, plan deduplication, and `--quick` flag. |
| `train.py` | 100 epochs, bs 32, 100 plans | ~20 seconds (CPU) | **PASS** | Add query-based split, fix ranking loss, add MLflow logging, add `--quick` (10 epochs). |
| `evaluate.py` | Evaluates single checkpoint | ~1.0 second (CPU) | **PASS** | Add per-query Spearman/Kendall, top-1 accuracy, regret, and baselines (PostgreSQL cost + GBDT). |
| `visualize.py` | t-SNE (100 samples) | ~2.5 seconds | **PASS** | Functional. |
| `analyze_errors.py` | Evaluates dataset | Crashes immediately | **FAIL (Bug)** | Fix `x_cont` attribute key, fix batch metadata collation. |
| `extract_plans.py` | Runs PostgreSQL queries | Indefinite / Fails | **N/A** | PostgreSQL / TPC-H instance is not installed/running locally. If run against a live DB, unindexed TPC-H queries without timeouts can exceed 15 mins. |
| *New: Ablation runner* | Grid: 3 layers × 2 poolings × 2 dims × 3 seeds = 36 runs | 36 × 15s = ~9.0 minutes | **PASS (<12 min budget)** | Implement with `--quick` mode (1 seed, 4 configurations = ~40s). |
| *New: ONNX & Benchmark* | 500 runs × 5 repeats | ~3.0 minutes | **PASS (<5 min budget)** | Implement ONNX export and benchmark harness with CPU thread pinning. |

---

## 5. Required Gaps to Fix in Step 2

To make the repository mathematically sound, fully reproducible, and aligned with the resume claims (or accurately re-scoped):

1. **Bug Fixes:**
   - Fix inverted target in `scripts/train.py:100` (`target = -torch.ones_like(pred_fast)` or swap fast/slow).
   - Remove `actual_rows` from `src/feature_encoder.py` input features to prevent runtime data leakage.
   - Fix crash bugs in `scripts/analyze_errors.py` (`x_cont`, plan dictionary access).
   - Add GraphSAGE layer (`SAGEConv`) support in `src/model.py`.

2. **Data & Splitting:**
   - Implement **Query-Level Splitting**: Train / Val / Test split grouped by `query_id` (e.g. 70/15/15) to prevent plan leakage.
   - Expand synthetic/benchmark dataset to support diverse alternative join plans (e.g., 200+ deduplicated plans across 25+ queries).
   - If PostgreSQL / TPC-H cannot be executed locally within the time limit or lacks local server, provide realistic simulated PostgreSQL EXPLAIN plans matching TPC-H templates with multiple planner hints, and clearly document this in the report limitations.

3. **Evaluation & Baselines:**
   - In `scripts/evaluate.py`, calculate:
     - Per-query Spearman rank correlation (averaged across queries).
     - Per-query Kendall's tau (averaged across queries).
     - Top-1 plan selection accuracy.
     - Suboptimality regret: `(runtime(selected_plan) - runtime(optimal_plan))`.
     - Regression metrics: MAE, RMSE, R², Mean Relative Error (MRE).
   - Implement baselines on the exact same test split:
     - Baseline 1: PostgreSQL native optimizer cost estimate (`Total Cost`).
     - Baseline 2: Gradient Boosting (LightGBM/HistGradientBoosting/RandomForest) on flat aggregated plan features.
   - Compute the true % improvement over baseline heuristics for Top-1 selection and regret.

4. **Ablation & Experiment Tracking:**
   - Create `scripts/run_ablation.py`:
     - Grid: GCN vs GAT vs GraphSAGE; Mean vs Attention Pooling; 16 vs 32 operator embedding dims.
     - 3 seeds per configuration, tracking mean and standard deviation.
     - Integrated MLflow logging (or local SQLite/JSON tracking compatible with MLflow) logging params, metrics, and duration.
     - Include `--quick` mode.

5. **ONNX Export & Latency Benchmarking:**
   - Create `scripts/export_onnx.py`:
     - Implement ONNX-compatible export (adapting attention pooling for ONNX tracing/scripting).
   - Create `scripts/benchmark_latency.py`:
     - CPU only, batch size 1, thread count fixed to 1.
     - 500+ iterations after 50 warm-up iterations.
     - Measure both **Model-Only** and **End-to-End** (JSON parse -> Graph construct -> Tensor convert -> ONNX inference).
     - Repeat 5 times and take the median p50, p95, and p99.
   - Verify if p95 is sub-8ms.

6. **Results Persistence:**
   - Save all evaluation outputs, baseline comparisons, ablation tables, and latency benchmarks into `results/` with timestamps and elapsed wall-clock times.
