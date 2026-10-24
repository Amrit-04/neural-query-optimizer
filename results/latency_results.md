# ONNX CPU Inference Latency Benchmark Report

**Date:** 2026-10-24 13:34:46  
**Architecture:** GNN  
**Environment:** Windows (CPU only, single-threaded execution `intra_op_num_threads=1`)  
**Benchmarking Protocol:** 5 trials of 500 runs (median aggregation across trials)  
**Total Benchmark Time:** 1.32s  

---

## Latency Summary

| Scope | p50 (Median) | p95 | p99 | Mean | Resume Claim (<8ms p95) |
|---|---|---|---|---|---|
| **Model-Only Inference** | **0.181 ms** | **0.21 ms** | **0.269 ms** | 0.186 ms | **PASSED (Sub-8ms verified)** |
| **End-to-End (JSON -> Inference)** | **0.302 ms** | **0.439 ms** | **0.504 ms** | 0.32 ms | PASSED (Sub-8ms) |

### Trial-by-Trial p95 Stability (ms)
- **Model-Only p95 Trials:** `[0.108, 0.266, 0.216, 0.21, 0.207]`
- **End-to-End p95 Trials:** `[0.37, 0.46, 0.439, 0.446, 0.42]`
