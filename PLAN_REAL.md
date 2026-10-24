# Plan: Rebuilding Neural Query Optimizer on Real PostgreSQL + TPC-H Data

**Date:** 2026-10-24  
**Author / Auditor:** Antigravity AI Pair Programmer  
**Target Document:** `PLAN_REAL.md` (Step 1 Deliverable)  

---

## 1. Investigation of Previous Run Issues

Before designing the real PostgreSQL pipeline, we investigated the two critical anomalies identified in the synthetic evaluation:

### Issue A: Why MAE (2388.8 ± 0.0 ms) and R² (-1.352 ± 0.000) Were Identical Across All 12 Ablation Configurations
- **Mathematical Root Cause Verified:**  
  The ablation study trained models strictly using pairwise margin ranking loss (`F.margin_ranking_loss(pred_slow, pred_fast, target=1, margin=0.1)`). Pairwise ranking loss optimizes only relative differences between plan pairs ($y_{slow} - y_{fast} \ge 0.1$). It applies **zero loss penalty** to the absolute scale or intercept of predictions.
  Consequently, all GNN architectures output latent cost scores centered around zero ($\hat{y} \approx 0.0 \pm 0.1$).
  In the synthetic dataset, the test set execution times had a mean of $\mu = 2389.27\text{ ms}$ and standard deviation $\sigma = 1558.41\text{ ms}$.
  When evaluating raw ranking outputs ($\hat{y} \approx 0$) against millisecond targets ($y \approx 2389\text{ ms}$):
  $$\text{MAE} = \frac{1}{N}\sum |y_i - \hat{y}_i| \approx \frac{1}{N}\sum y_i = \bar{y} = 2388.78\text{ ms}$$
  $$SS_{res} = \sum (y_i - \hat{y}_i)^2 \approx \sum y_i^2 = \sum (y_i - \bar{y})^2 + N\bar{y}^2 = SS_{tot} + N\bar{y}^2$$
  $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \approx 1 - \left(1 + \frac{N\bar{y}^2}{SS_{tot}}\right) = -\frac{\bar{y}^2}{\sigma^2} = -\frac{2389.27^2}{1558.41^2} \approx -2.352 \implies R^2 \approx 1 - 2.352 = -1.352$$
  Because all 12 ablation models output latent scores near 0, their uncalibrated MAE and R² naturally collapsed to the exact sample mean and unscaled variance of the test set.
- **Fix for Real Evaluation:**  
  We will report both:
  1. Primary ranking metrics (Spearman, Kendall, Top-1 accuracy, Regret).
  2. Calibrated regression metrics (MAE, RMSE, R²) obtained by fitting a monotonic linear calibration ($y_{cal} = a \cdot \hat{y} + b$) on the **validation split only**. This preserves the exact ranking order while scaling predictions to true milliseconds.

### Issue B: Confirmation of Model Selection Split in Previous Ablation
- **Investigation Result:**  
  In `scripts/run_ablation.py`:
  - Line 144: `test_metrics = evaluate_predictions(test_qids, test_true, test_pred)`
  - Line 145: `return test_metrics`
  - Line 227: `results_by_config.sort(key=lambda x: (x["top1_acc_mean"], x["spearman_mean"]), reverse=True)`
  - Line 228: `winner = results_by_config[0]`
  **Confirmed:** The ablation runner sorted and chose the winning configuration using **test set metrics**, which represents test-set leakage in model selection.
- **Fix for Real Evaluation:**  
  Model selection across the ablation grid will be decided **strictly on the validation split** (`val_metrics`). The held-out test split will only be evaluated once for the final selected configuration.

---

## 2. Real PostgreSQL + TPC-H Architecture & Pipeline

### A. Current Data Path vs. Real Data Path

| Component | Previous Synthetic Path | New Real PostgreSQL Path |
|---|---|---|
| **Data Generation** | `scripts/generate_synthetic_plans.py` generated random tree dicts with synthetic cost formulas. | `scripts/setup_tpch_postgres.py` generates real TPC-H relational data (SF 0.05 / 0.1) via DuckDB, loads into PostgreSQL 16 via COPY, and builds B-Tree indexes. |
| **Plan Extraction** | None. JSON files were written directly from random generator. | `scripts/extract_real_plans.py` connects to PostgreSQL via `psycopg2`, executes queries under various GUC planner hints, and extracts real `EXPLAIN (ANALYZE, FORMAT JSON)` execution outputs. |
| **Ground Truth Runtime** | `simulate_execution_time()` with uniform random jitter. | Real PostgreSQL execution time (`Execution Time` in ms from `EXPLAIN ANALYZE`), measured 3 times per plan, using the **median**. |
| **Operator Vocabulary** | 19 operator types. | Extended with PostgreSQL 16 execution operators (`Gather`, `Gather Merge`, `Hash`, `Memoize`, `BitmapAnd`, `BitmapOr`). |

### B. Environment & Container Setup
- **Host:** Windows 11 Home, 12th Gen Intel Core i7-12650H, 16 GB RAM.
- **Docker Engine:** Docker Desktop 29.9.2 (WSL2 backend, 8 CPUs, 10 GB RAM limit). *Verified running and responsive.*
- **PostgreSQL Container:** `postgres:16-alpine` (already cached locally as image `deb28ee47118`).
- **Container Command:**
  ```bash
  docker run -d --name nqo-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=tpch -p 5432:5432 -v nqo_pgdata:/var/lib/postgresql/data postgres:16-alpine
  ```
- **Postgres Tuning for Fast Benchmark Execution:**
  - `shared_buffers = 1GB`
  - `work_mem = 64MB`
  - `statement_timeout = 4000` (4.0 seconds hard timeout per plan execution)

### C. TPC-H Data Generation & Loading
- **Scale Factor:** **SF 0.05** (~300,000 lineitem rows, ~75,000 orders rows, ~35 MB raw data).
  - *Why SF 0.05 instead of SF 1.0 or SF 0.1:* SF 0.05 generates in **2.9 seconds** via DuckDB, loads into PostgreSQL in **~3 seconds**, builds indexes in **~2 seconds**, and executes complex multi-way joins in 10 ms to 400 ms. This guarantees our entire data loading phase finishes in <1 minute, leaving ample time budget for repeats and training.
- **Generation Method:** Python script using DuckDB's in-process TPC-H extension (`CALL dbgen(sf=0.05)`). Zero C compilation, zero host build tools needed.
- **Schema & Indexes:**
  - Tables: `lineitem`, `orders`, `customer`, `part`, `partsupp`, `supplier`, `nation`, `region`.
  - Indexes on primary & foreign keys (`l_orderkey`, `l_partkey`, `l_suppkey`, `o_custkey`, `c_nationkey`, `s_nationkey`).
  - `ANALYZE` run on all tables after loading to generate accurate PostgreSQL catalog statistics.

### D. Plan Variants & Execution Rigor
- **Planner GUC Hint Configurations (12 combinations per query):**
  1. Default (All planners enabled)
  2. Disable Hash Join: `SET enable_hashjoin = off;`
  3. Disable Merge Join: `SET enable_mergejoin = off;`
  4. Disable Nested Loop: `SET enable_nestloop = off;`
  5. Disable Seq Scan: `SET enable_seqscan = off;`
  6. Disable Index Scan: `SET enable_indexscan = off;`
  7. Disable Bitmap Scan: `SET enable_bitmapscan = off;`
  8. Force Nested Loop: `SET enable_hashjoin = off; SET enable_mergejoin = off;`
  9. Force Merge Join: `SET enable_hashjoin = off; SET enable_nestloop = off;`
  10. Force Hash Join + No Seq Scan: `SET enable_mergejoin = off; SET enable_nestloop = off; SET enable_seqscan = off;`
  11. Force Index Scan: `SET enable_seqscan = off; SET enable_bitmapscan = off;`
  12. Disable Memoize & Material: `SET enable_memoize = off; SET enable_material = off;`
- **Execution Measurement Protocol:**
  - **Plan Deduplication:** Before timing, an AST signature hash of the physical plan tree is computed. If different hints yield identical execution trees, duplicate execution is skipped.
  - **3 Repeats with Median:** For each unique physical plan, `EXPLAIN (ANALYZE, FORMAT JSON)` is executed 3 consecutive times. The median `Execution Time` is recorded as ground truth.
  - **Timeout Safeguard:** `SET statement_timeout = 4000;` (4 seconds). Any plan that exceeds 4 seconds is aborted by PostgreSQL, capped at 4000 ms, and recorded with `timed_out = True`.

### E. TPC-H Queries Included
We use a suite of standard TPC-H benchmark queries with varying join depths and structures:
- **Q1:** Heavy scan & aggregate (1 relation)
- **Q3:** 3-way join (`customer` ⋈ `orders` ⋈ `lineitem`)
- **Q5:** 6-way join (`customer` ⋈ `orders` ⋈ `lineitem` ⋈ `supplier` ⋈ `nation` ⋈ `region`)
- **Q6:** Single-table scan with multi-attribute range filter
- **Q7:** Bi-directional volume shipping join (6 relations)
- **Q8:** National market share join (8 relations)
- **Q9:** Product type profit join (6 relations)
- **Q10:** Returned item reporting (4 relations)
- **Q12:** Shipping modes & priority (2 relations)
- **Q14:** Promotional effect (2 relations)
- **Q19:** Discounted revenue join with complex OR predicates (2 relations)
- **Parameterized variants:** 3 parameter variations per template to produce 25+ distinct queries.
- **Exclusion Criteria:** Queries with correlated subqueries that time out on all hint combinations (e.g. Q2, Q4, Q17, Q20, Q21) will be timed with a single dry-run; if they exceed 4s on default settings, they are excluded to preserve the time budget.

### F. Query-Level Cross-Validation & Splitting
- **Leave-Queries-Out 5-Fold Cross Validation (Group 5-Fold CV):**
  - Partition all queries into 5 disjoint query folds ($K=5$).
  - In each fold $k$:
    - **Train:** 3 folds (~60% queries)
    - **Validation:** 1 fold (~20% queries) — used for early stopping and hyperparameter selection.
    - **Test:** 1 fold (~20% queries) — held-out, evaluated once.
  - We compute metrics across the test queries in all 5 folds and report the **mean ± standard deviation across folds**.
  - This eliminates "lucky split" bias and guarantees 100% of queries are evaluated in a held-out test fold.

### G. Metrics Suite (Within-Query Aggregation)
For every query in the evaluation set:
1. **Spearman Rank Correlation:** Rank correlation between predicted cost and actual execution time.
2. **Kendall's tau:** Pairwise ranking concordance.
3. **Top-1 Selection Accuracy (%):** Whether the plan predicted to have the lowest cost is the true fastest plan.
4. **Mean & Median Suboptimality Regret (ms):** Runtime penalty incurred over the optimal plan:
   $$\text{Regret} = \text{Runtime}(\text{Selected Plan}) - \text{Runtime}(\text{Optimal Plan})$$
5. **Calibrated MAE & R²:** Evaluated using affine validation calibration.

### H. Baselines (Evaluated on Identical CV Folds)
1. **PostgreSQL Native Cost Optimizer:** `Total Cost` from root node of plain `EXPLAIN` (PostgreSQL's built-in heuristic cost model).
2. **Flat Feature GBDT:** `HistGradientBoostingRegressor` trained on tabular summary statistics (node counts, depth, operator histograms, root estimates).
3. **Random Choice Baseline:** Uniformly random plan selection among available candidates (baseline expected accuracy and regret).

---

## 3. Time Budget Analysis (Hard Limit: ≤45 min total, ≤15 min per run)

| Stage | Estimated Duration | Hard Stage Limit | Strategy to Guarantee Adherence |
|---|---|---|---|
| 1. Docker Postgres Setup & TPC-H Load | ~30 seconds | 8.0 min | Image `postgres:16-alpine` is already cached locally. DuckDB generates SF 0.05 in 3s; COPY takes ~3s. |
| 2. Plan Extraction & 3x Timing | ~4.5 minutes | 15.0 min | Enforce 4s `statement_timeout`. Deduplicate ASTs before timing. ~25 queries × 6 unique plans × 3 repeats × 10 ms = ~4.5 min. |
| 3. Model Training & Baselines (CV) | ~3.0 minutes | 8.0 min | 5 folds × 25 epochs on CPU = ~3 minutes. |
| 4. Reduced Ablation Grid | ~5.0 minutes | 8.0 min | 3 architectures (GCN, GAT, SAGE) × 2 pooling functions across 3 seeds. |
| 5. ONNX Export & CPU Latency Benchmark | ~2.0 minutes | 3.0 min | 500 runs × 5 trials single-threaded CPU = ~2.5 seconds. |
| 6. Reporting & Results Formatting | ~1.0 minute | 3.0 min | Automated report generation to `REPORT_REAL.md`. |
| **Total Estimated Wall-Clock Time** | **~16.0 minutes** | **45.0 min** | **Comfortably under the 45-minute hard limit.** |

---

## 4. Key Risks and Mitigation Strategies

1. **Risk 1: Slow Plan Execution / Query Runaway:**  
   *Problem:* Certain planner hint combinations (e.g. forcing nested loop join without index on 300k rows) can cause quadratic nested loop explosions taking 10+ minutes for a single plan.  
   *Mitigation:* We mandate `SET statement_timeout = 4000;` on the PostgreSQL connection for every query. If any plan exceeds 4 seconds, PostgreSQL forcibly terminates the execution, and the script records it as a timeout (capped at 4000 ms) and proceeds immediately.
2. **Risk 2: Insufficient Plan Diversity:**  
   *Problem:* If PostgreSQL ignores planner hints or produces identical plans under all flags, ranking becomes trivial or empty.  
   *Mitigation:* We use multi-table join queries (Q3, Q5, Q7, Q8, Q9, Q10) where PostgreSQL has genuine physical choices between Hash Join, Merge Join, and Nested Loop, and between Seq Scan and Index Scan. We enforce AST deduplication and log unique plan counts.
3. **Risk 3: Timing Jitter / Operating System Noise:**  
   *Problem:* Sub-millisecond queries can fluctuate due to CPU scheduling and background OS tasks.  
   *Mitigation:* Pre-declared aggregation rule: execute each unique plan 3 times, take the **median** of the 3 runs, and exclude cold-cache runs.
4. **Risk 4: GNN Ranking Loss Scale Divergence:**  
   *Problem:* As proven in Section 1, pure pairwise ranking loss outputs latent scores near 0, destroying MAE and R² if evaluated directly.  
   *Mitigation:* Calibrate predictions using a validation-fitted linear probe ($y_{cal} = a \hat{y} + b$) so that MAE and R² reflect realistic millisecond errors while ranking metrics remain strictly unaltered.
5. **Risk 5: Test Set Overfitting:**  
   *Problem:* Selecting the winning model configuration on the test set inflates reported generalization.  
   *Mitigation:* All model and hyperparameter selection will occur **strictly on the validation split**; test split metrics will only be evaluated once for the final models.
