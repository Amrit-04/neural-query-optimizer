# Benchmark & Baseline Comparison Report

**Date:** 2026-10-23 17:52:20  
**Evaluation Set:** 5 Held-Out Queries (40 Plans), Zero Query Leakage  
**Evaluation Duration:** 0.66s  

---

## 1. Primary Ranking & Selection Metrics (Within-Query)

| Metric | GNN Model (Learned) | Baseline 1: PostgreSQL Native Cost | Baseline 2: Flat GBDT | Improvement vs. Postgres Native | Improvement vs. Flat GBDT |
|---|---|---|---|---|---|
| **Top-1 Selection Accuracy** | **100.0%** | 100.0% | 80.0% | **+0.0%** | **+25.0%** |
| **Mean Regret (ms)** | **0.00 ms** | 0.00 ms | 6.46 ms | **+0.0%** | **+100.0%** |
| **Median Regret (ms)** | **0.00 ms** | 0.00 ms | 0.00 ms | - | - |
| **Spearman Rank Corr. (Per Query)** | **0.789** | 0.975 | 0.965 | -19.0% | -18.2% |
| **Kendall's tau (Per Query)** | **0.699** | 0.942 | 0.913 | -25.8% | -23.4% |

---

## 2. Global Regression & Pooled Metrics

| Metric | GNN Model | PostgreSQL Native | Flat GBDT |
|---|---|---|---|
| **MAE (ms)** | 2388.78 ms | 16237.22 ms | 49.32 ms |
| **RMSE (ms)** | 3150.72 ms | 50982.83 ms | 68.68 ms |
| **R² Score** | -1.352 | -614.784 | 0.999 |
| **Mean Relative Error (MRE)** | 99.8% | 452.1% | 18.6% |
| **Spearman (Pooled across queries)** | 0.808 | 0.994 | 0.992 |
| **Kendall (Pooled across queries)** | 0.672 | 0.963 | 0.942 |
