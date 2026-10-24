# Real PostgreSQL TPC-H Ablation Study Report

**Date:** 2026-10-24 13:25:59  
**Duration:** 89.80s  
**Selection Criterion:** Winner selected strictly on Validation Top-1 and Regret (No test set leakage)  
**Winning Configuration:** `GCN_mean_with_cost`  

| Architecture | Pooling | Cost Features? | Val Top-1 (%) | Val Regret (ms) | Val Spearman | Val Speedup vs Def | Test Top-1 (%) | Test Regret (ms) | Test Spearman |
|---|---|---|---|---|---|---|---|---|---|
| **GCN** | mean | Yes (dim=4) | 50.0% | 1.00 ms | 0.296 | 0.957x | 0.0% | 5.45 ms | -0.200 |
| SAGE | attention | Yes (dim=4) | 50.0% | 1.00 ms | 0.296 | 0.957x | 50.0% | 5.33 ms | -0.050 |
| SAGE | mean | Yes (dim=4) | 50.0% | 1.00 ms | 0.296 | 0.957x | 50.0% | 5.33 ms | -0.050 |
| GCN | attention | Yes (dim=4) | 50.0% | 1.00 ms | 0.264 | 0.957x | 50.0% | 5.33 ms | -0.050 |
| GAT | attention | Yes (dim=4) | 0.0% | 5.09 ms | 0.243 | 0.847x | 0.0% | 5.45 ms | -0.100 |
| GAT | mean | No (dim=2) | 0.0% | 5.09 ms | 0.232 | 0.847x | 50.0% | 5.33 ms | -0.150 |
| SAGE | attention | No (dim=2) | 0.0% | 5.09 ms | 0.232 | 0.847x | 50.0% | 5.33 ms | -0.150 |
| GAT | attention | No (dim=2) | 0.0% | 5.09 ms | 0.189 | 0.847x | 0.0% | 5.45 ms | -0.100 |
| GCN | attention | No (dim=2) | 0.0% | 5.09 ms | 0.146 | 0.847x | 50.0% | 5.33 ms | -0.050 |
| GCN | mean | No (dim=2) | 0.0% | 5.09 ms | 0.146 | 0.847x | 0.0% | 5.45 ms | -0.200 |
| SAGE | mean | No (dim=2) | 0.0% | 5.09 ms | 0.146 | 0.847x | 100.0% | 0.00 ms | 0.950 |
| GAT | mean | Yes (dim=4) | 0.0% | 11.98 ms | -0.025 | 0.586x | 0.0% | 61.61 ms | -0.700 |
