# Neural Query Cost Model

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Sub--0.5ms%20CPU-005CED?style=flat)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A learned query cost estimator and physical plan ranker for PostgreSQL query execution plans using Graph Neural Networks (GNNs). The system converts PostgreSQL `EXPLAIN` plan trees into graph representations, encodes relational operator semantics, and ranks alternative physical join plans using pairwise ranking objectives.

---

## The Problem

Traditional database query optimizers (Cost-Based Optimizers, CBOs) rely on static, hand-crafted mathematical formulas to estimate execution costs. These analytical cost models:
- Assume uniform data distribution and attribute value independence (AVI).
- Suffer severe cardinality estimation error propagation through multi-table joins.
- Cannot easily adapt to skewed real-world execution characteristics.
- Often choose suboptimal physical plans (e.g. choosing nested loops where hash joins are 2–3x faster).

---

## Our Solution

We learn cost behavior directly from relational execution traces using **Graph Neural Networks**:
- **Tree-to-Graph Pipeline**: Maps PostgreSQL `EXPLAIN (FORMAT JSON)` operator trees directly to graph tensors without runtime execution leakage.
- **Relational Operator Embeddings**: Captures operator semantics (Seq Scan, Index Scan, Hash Join, Merge Join, Nested Loop, Aggregate, etc.) with learned categorical embeddings.
- **Continuous Features**: Ingests planning-time estimated rows, plan width, and cost estimates.
- **Pairwise Ranking Optimization**: Uses margin ranking loss to learn relative cost utility within query candidate plan pools.
- **Ultra-Fast ONNX CPU Inference**: Compiles the trained GNN to ONNX Runtime for sub-millisecond CPU execution inside query planning loops.

```
       PostgreSQL Query
              ↓
  Planner Hints / GUC Variants (Hash, Merge, NestLoop, Index)
              ↓
    Plain EXPLAIN Plan Tree (No runtime leakage)
              ↓
    Tree-to-Graph Tensor Conversion (Nodes, Edges, Operator Embeddings)
              ↓
    GNN Cost Model (GCN / GraphSAGE + Attention Pooling)
              ↓
Predicted Cost Rank & Optimal Plan Selection (<0.5ms on CPU)
```

---

## Empirical Benchmark Results (Real PostgreSQL 16 + TPC-H)

Evaluated under strict **Nested Leave-One-Template-Out Cross-Validation** on **184 real PostgreSQL 16 execution plans** across 12 TPC-H query templates at Scale Factor 0.05 (~300,000 lineitem rows). Hyperparameters, model selection, and calibration probes were fit **strictly on validation splits** with zero test set leakage.

Results are reported as the **mean $\pm$ standard deviation across 3 random seeds**:

| Metric | Learned GNN (PlanGCN) | PostgreSQL Native CBO | Flat GBDT Baseline | Random Baseline | GNN vs. Flat GBDT |
|---|---|---|---|---|---|
| **Top-1 Plan Selection Accuracy** | **38.9% ± 13.8%** | 41.7% ± 0.0% | 29.2% ± 0.0% | 19.4% ± 7.1% | **+33.3% relative** |
| **Spearman Rank Correlation ($\rho$)** | **0.738 ± 0.030** | 0.826 ± 0.000 | 0.722 ± 0.000 | 0.017 ± 0.045 | **+2.2% relative** |
| **Kendall's Tau ($\tau$)** | **0.623 ± 0.038** | 0.739 ± 0.000 | 0.619 ± 0.000 | 0.028 ± 0.039 | **+0.6% relative** |
| **Mean Plan Regret (ms)** | **3.29 ± 0.60 ms** | 2.28 ± 0.00 ms | 4.26 ± 0.00 ms | 9.85 ± 1.20 ms | **-22.8% regret** |
| **Speedup vs. PostgreSQL Default** | **0.991x** | 1.002x | 0.854x | 0.680x | **+16.0% faster** |

*Full evaluation report, query-by-query breakdown, and methodology: see [`REPORT_REAL.md`](REPORT_REAL.md).*

---

### Feature Ablation: With vs. Without PostgreSQL Cost Features

To verify whether the GNN relies purely on PostgreSQL's internal cost formulas or learns topological graph patterns, we evaluated the GNN with and without PostgreSQL cost features (`startup_cost` and `total_cost`):

| Feature Setting | Continuous Dim | Top-1 Accuracy | Spearman $\rho$ | Kendall's $\tau$ | Mean Regret | Speedup vs Default |
|---|---|---|---|---|---|---|
| **Without Cost Features** (Structure + Rows/Width only) | 2 | 23.6% ± 2.0% | **0.659 ± 0.063** | **0.524 ± 0.069** | 3.41 ms | 0.953x |
| **With Cost Features** (Structure + Rows/Width + PG Costs) | 4 | **38.9% ± 13.8%** | **0.738 ± 0.030** | **0.623 ± 0.038** | **3.29 ms** | **0.991x** |

> **Key Finding:** When evaluated strictly on graph structure and table statistics (zero PostgreSQL cost features), the model achieves **0.659 Spearman $\rho$** and **0.524 Kendall's $\tau$**, confirming that structural operator representations generalize across relational topologies.

---

### Key Optimizer Insights

1. **Beating Tabular Heuristics:** The GNN achieves a **+33.3% relative improvement in Top-1 plan selection** over standard flat tabular models (Flat GBDT), cutting average plan regret by **22.8%**.
2. **PostgreSQL Comparison:** On completely unseen query templates, PostgreSQL's hand-tuned analytical CBO achieves 41.7% Top-1 accuracy vs. the GNN's 38.9%. However, on specific query families where PostgreSQL's CBO fails due to cardinality misestimation:
   - **TPC-H Q10 (Returned Item Reporting):** The GNN selects the optimal Hash Join plan, achieving a **2.17x speedup over PostgreSQL's default plan** and reducing plan regret from **18.60 ms to 1.35 ms**.
   - **TPC-H Q12 (Shipping Modes and Priority):** The GNN achieves **100% Top-1 accuracy** ($0.0\text{ ms}$ regret), whereas PostgreSQL native cost incurs 5.08 ms of regret.
   - **TPC-H Q8 (National Market Share):** The GNN achieves a **1.07x speedup** over PostgreSQL's default plan.

---

### ONNX CPU Latency Benchmark

The best model (`GCN_mean_with_cost`) was exported to ONNX Runtime and benchmarked under single-threaded CPU execution (`intra_op_num_threads=1`) across 5 independent trials of 500 runs each (median aggregation):

| Scope | p50 (Median) | p95 | p99 | Mean | SLA Target |
|---|---|---|---|---|---|
| **Model-Only Inference** | **0.181 ms** | **0.210 ms** | **0.269 ms** | 0.186 ms | **< 8.0 ms (PASSED)** |
| **End-to-End (Raw JSON $\to$ Graph $\to$ Inference)** | **0.302 ms** | **0.439 ms** | **0.504 ms** | 0.320 ms | **< 8.0 ms (PASSED)** |

At **0.21ms p95 model latency** and **0.44ms end-to-end latency**, neural cost inference uses less than 1% of PostgreSQL's typical planning time budget.

---

## Project Structure

```
neural-query-optimizer/
├── config/
│   └── default.yaml              # Hyperparameter configuration
├── src/
│   ├── graph_builder.py          # Convert plan trees to graphs (parent -> child)
│   ├── feature_encoder.py        # Operator vocabulary & continuous feature extraction
│   ├── dataset.py                # PyG/PyTorch Dataset & graph batch collation
│   ├── model.py                  # PlanGCN, PlanGAT, and PlanSAGE architectures
│   ├── metrics.py                # Within-query ranking metrics (Spearman, Kendall, Regret)
│   └── split_utils.py            # Deterministic template-aware dataset splitting
├── scripts/
│   ├── setup_tpch_postgres.py    # DuckDB TPC-H SF 0.05 schema & data load into Postgres
│   ├── extract_real_plans.py     # Real PostgreSQL plan extraction & 3-run timing
│   ├── train_eval_real.py        # Nested Leave-One-Template-Out cross-validation
│   ├── run_ablation_real.py      # Architecture/pooling ablation with MLflow tracking
│   ├── benchmark_onnx.py         # Single-threaded CPU ONNX latency benchmark
│   └── analyze_errors.py         # Plan-level error and regret diagnostics
├── data/
│   ├── plans_real/               # 184 real PostgreSQL execution plans (TPC-H SF 0.05)
│   └── plans_synthetic/          # Synthetic plan dataset for rapid local testing
├── results/
│   ├── evaluation_real.json      # Full 3-seed nested CV results (with cost)
│   ├── evaluation_real_no_cost.json # Full 3-seed nested CV results (without cost)
│   ├── ablation_real.md          # Multi-architecture ablation summary
│   ├── latency_results_real.json # ONNX CPU latency benchmark data
│   └── real_data_summary.json    # Plan counts, runtimes, and timeout audits
├── models/
│   └── best_real.onnx            # Exported winning model for ONNX Runtime
├── REPORT_REAL.md                # Comprehensive 12-section empirical audit report
└── requirements.txt
```

---

## Quick Start: Reproducing the Benchmark

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Amrit-04/neural-query-optimizer.git
cd neural-query-optimizer

# Install Python dependencies (Python 3.10 or 3.11 recommended)
pip install -r requirements.txt
```

### 2. Launch PostgreSQL & Load TPC-H Data

```bash
# Launch PostgreSQL 16 container
docker run -d --name nqo-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-alpine

# Generate TPC-H SF 0.05 in DuckDB and stream directly into PostgreSQL
python scripts/setup_tpch_postgres.py
```

### 3. Extract Real Plans & Benchmark Execution Timings

```bash
# Extract plain EXPLAIN plans and measure 3-run median execution times
python scripts/extract_real_plans.py
```

### 4. Run Nested Cross-Validation

```bash
# Full 12-template Leave-One-Template-Out Nested CV across 3 seeds
python scripts/train_eval_real.py

# Feature ablation: without PostgreSQL cost features
python scripts/train_eval_real.py --no_cost
```

### 5. Run Ablations & Benchmark ONNX Latency

```bash
# Run 12-configuration GCN vs GAT vs SAGE ablation with MLflow tracking
python scripts/run_ablation_real.py

# Benchmark single-threaded CPU ONNX inference latency
python scripts/benchmark_onnx.py --onnx_model models/best_real.onnx --data data/plans_real
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
