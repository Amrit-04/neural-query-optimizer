"""
ONNX Export and Latency Benchmarking.

Protocol:
- CPU inference only, batch size 1.
- Fixed single thread (intra_op_num_threads=1).
- 50 warm-up iterations.
- 500 benchmark iterations per trial.
- 5 repeated trials; reports median across trials for p50, p95, and p99.
- Measures both Model-Only and End-to-End (Raw JSON -> Graph -> Tensors -> ONNX).
- Outputs to results/latency_results.json and results/latency_results.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch

from src.dataset import PlanGraphDataset
from src.feature_encoder import OperatorVocabulary, node_features_to_tensor
from src.graph_builder import build_graph
from src.logging_config import setup_logging
from src.model import build_model

logger = logging.getLogger("neural_optimizer")


def export_model_to_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    sample_plan_nodes: list[dict],
    vocab: OperatorVocabulary,
) -> tuple[dict, str]:
    """Export PyTorch checkpoint to ONNX format."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    vocab_len = ckpt.get("vocab_len", len(vocab))

    model = build_model(config, num_operators=vocab_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create sample inputs with batch size 1
    tensors = node_features_to_tensor(sample_plan_nodes, vocab)
    x_op = tensors["operator_ids"]
    x_cont = tensors["continuous"]
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    batch = torch.zeros(len(sample_plan_nodes), dtype=torch.long)

    # Export using PyTorch legacy TorchScript engine for PyG graph compatibility
    torch.onnx.export(
        model,
        (x_op, x_cont, edge_index, batch),
        str(onnx_path),
        input_names=["x_op", "x_cont", "edge_index", "batch"],
        output_names=["predicted_cost"],
        dynamic_axes={
            "x_op": {0: "num_nodes"},
            "x_cont": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "batch": {0: "num_nodes"},
        },
        dynamo=False,
    )
    logger.info("Exported ONNX model to %s", onnx_path)
    return config, config.get("model", {}).get("type", "gnn")


def run_latency_benchmark(
    onnx_path: Path,
    sample_raw_json: dict,
    vocab: OperatorVocabulary,
    num_runs: int = 500,
    warmup_runs: int = 50,
    num_trials: int = 5,
) -> dict:
    """Execute latency benchmark over 5 trials with median aggregation."""
    # Set single thread in ONNX Runtime for reproducible CPU latency
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.log_severity_level = 3  # Errors only
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    # Prepare pre-encoded inputs for Model-Only benchmark
    graph = build_graph(sample_raw_json["plan"])
    nodes = graph["nodes"]
    edge_index = graph["edge_index"]
    tensors = node_features_to_tensor(nodes, vocab)

    model_inputs = {
        "x_op": tensors["operator_ids"].numpy(),
        "x_cont": tensors["continuous"].numpy(),
        "edge_index": edge_index,
        "batch": np.zeros(len(nodes), dtype=np.int64),
    }

    # 1. Warm-up
    for _ in range(warmup_runs):
        session.run(None, model_inputs)

    # 2. Benchmark Trials
    model_only_trials = []
    e2e_trials = []

    for t_idx in range(num_trials):
        # A. Model-Only
        m_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run(None, model_inputs)
            t1 = time.perf_counter()
            m_times.append((t1 - t0) * 1000.0)  # ms

        # B. End-to-End (Raw Plan JSON -> Graph -> Features -> Inference)
        e2e_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            g = build_graph(sample_raw_json["plan"])
            nds = g["nodes"]
            ei = g["edge_index"]
            ts = node_features_to_tensor(nds, vocab)
            inp = {
                "x_op": ts["operator_ids"].numpy(),
                "x_cont": ts["continuous"].numpy(),
                "edge_index": ei,
                "batch": np.zeros(len(nds), dtype=np.int64),
            }
            session.run(None, inp)
            t1 = time.perf_counter()
            e2e_times.append((t1 - t0) * 1000.0)  # ms

        model_only_trials.append({
            "p50": float(np.percentile(m_times, 50)),
            "p95": float(np.percentile(m_times, 95)),
            "p99": float(np.percentile(m_times, 99)),
            "mean": float(np.mean(m_times)),
        })
        e2e_trials.append({
            "p50": float(np.percentile(e2e_times, 50)),
            "p95": float(np.percentile(e2e_times, 95)),
            "p99": float(np.percentile(e2e_times, 99)),
            "mean": float(np.mean(e2e_times)),
        })

    # Compute Median of the 5 Trials
    def median_metric(trials, metric):
        return float(np.median([tr[metric] for tr in trials]))

    summary = {
        "num_trials": num_trials,
        "runs_per_trial": num_runs,
        "warmup_runs": warmup_runs,
        "threads": 1,
        "batch_size": 1,
        "plan_num_nodes": len(nodes),
        "model_only": {
            "p50_ms": round(median_metric(model_only_trials, "p50"), 3),
            "p95_ms": round(median_metric(model_only_trials, "p95"), 3),
            "p99_ms": round(median_metric(model_only_trials, "p99"), 3),
            "mean_ms": round(median_metric(model_only_trials, "mean"), 3),
            "trials_p95": [round(tr["p95"], 3) for tr in model_only_trials],
        },
        "end_to_end": {
            "p50_ms": round(median_metric(e2e_trials, "p50"), 3),
            "p95_ms": round(median_metric(e2e_trials, "p95"), 3),
            "p99_ms": round(median_metric(e2e_trials, "p99"), 3),
            "mean_ms": round(median_metric(e2e_trials, "mean"), 3),
            "trials_p95": [round(tr["p95"], 3) for tr in e2e_trials],
        },
    }
    return summary


def main() -> None:
    start_time = time.time()
    setup_logging()

    parser = argparse.ArgumentParser(description="Export to ONNX and benchmark CPU latency")
    parser.add_argument("--onnx_model", default=None, help="Existing ONNX model to benchmark directly")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--onnx_out", default="checkpoints/model_best.onnx")
    parser.add_argument("--data", default="data/plans_synthetic")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Quick mode (50 runs, 2 trials)")
    args = parser.parse_args()

    num_runs = 50 if args.quick else args.runs
    num_trials = 2 if args.quick else args.trials

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = OperatorVocabulary()
    ds = PlanGraphDataset(args.data, vocab=vocab)
    if len(ds) == 0:
        logger.error("No plans found in %s", args.data)
        sys.exit(1)

    # Pick representative plan
    sample_path = ds.files[0]
    with open(sample_path) as f:
        sample_raw = json.load(f)

    graph = build_graph(sample_raw["plan"])

    # 1. Export (if onnx_model not provided)
    if args.onnx_model and Path(args.onnx_model).exists():
        onnx_file_path = Path(args.onnx_model)
        model_type = "GNN"
        logger.info("Using existing ONNX model: %s", onnx_file_path)
    else:
        onnx_file_path = Path(args.onnx_out)
        config, model_type = export_model_to_onnx(
            Path(args.checkpoint),
            onnx_file_path,
            graph["nodes"],
            vocab,
        )

    # 2. Benchmark
    logger.info(
        "Benchmarking ONNX inference (%d runs × %d trials, warmup=%d, CPU thread=1)...",
        num_runs,
        num_trials,
        15 if args.quick else 50,
    )
    bench_results = run_latency_benchmark(
        onnx_file_path,
        sample_raw,
        vocab,
        num_runs=num_runs,
        warmup_runs=15 if args.quick else 50,
        num_trials=num_trials,
    )

    elapsed = time.time() - start_time
    bench_results["benchmark_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    bench_results["total_duration_seconds"] = round(elapsed, 2)
    bench_results["onnx_file"] = str(onnx_file_path)
    bench_results["model_type"] = model_type

    # Save JSON
    with open(out_dir / "latency_results.json", "w") as f:
        json.dump(bench_results, f, indent=2)

    # Format Markdown Report
    p95_model = bench_results["model_only"]["p95_ms"]
    p95_e2e = bench_results["end_to_end"]["p95_ms"]
    sub_8ms_verdict = "PASSED (Sub-8ms verified)" if p95_model < 8.0 else "FAILED (Exceeds 8ms)"

    md_content = f"""# ONNX CPU Inference Latency Benchmark Report

**Date:** {bench_results['benchmark_timestamp']}  
**Architecture:** {model_type.upper()}  
**Environment:** Windows (CPU only, single-threaded execution `intra_op_num_threads=1`)  
**Benchmarking Protocol:** {bench_results['num_trials']} trials of {bench_results['runs_per_trial']} runs (median aggregation across trials)  
**Total Benchmark Time:** {elapsed:.2f}s  

---

## Latency Summary

| Scope | p50 (Median) | p95 | p99 | Mean | Resume Claim (<8ms p95) |
|---|---|---|---|---|---|
| **Model-Only Inference** | **{bench_results['model_only']['p50_ms']} ms** | **{bench_results['model_only']['p95_ms']} ms** | **{bench_results['model_only']['p99_ms']} ms** | {bench_results['model_only']['mean_ms']} ms | **{sub_8ms_verdict}** |
| **End-to-End (JSON -> Inference)** | **{bench_results['end_to_end']['p50_ms']} ms** | **{bench_results['end_to_end']['p95_ms']} ms** | **{bench_results['end_to_end']['p99_ms']} ms** | {bench_results['end_to_end']['mean_ms']} ms | {"PASSED (Sub-8ms)" if p95_e2e < 8.0 else "Exceeds 8ms"} |

### Trial-by-Trial p95 Stability (ms)
- **Model-Only p95 Trials:** `{bench_results['model_only']['trials_p95']}`
- **End-to-End p95 Trials:** `{bench_results['end_to_end']['trials_p95']}`
"""

    with open(out_dir / "latency_results.md", "w") as f:
        f.write(md_content)

    print(md_content)


if __name__ == "__main__":
    main()
