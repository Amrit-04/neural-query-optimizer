# Neural Query Cost Model

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


A machine learning-based cost model that learns to predict database query execution times using Graph Neural Networks (GNNs). This model can augment or replace traditional handcrafted heuristics in query optimizers, enabling better plan selection through learned cost functions.

## The Problem

Traditional database query optimizers use static, heuristic cost formulas that:
- Assume uniform data distributions
- Fail to adapt to actual workload characteristics
- Propagate cardinality estimation errors through the query plan
- Result in suboptimal query plan selection

## Our Solution

We learn cost behavior directly from query execution data using **Graph Neural Networks**:
- **Input**: Query execution plan (tree structure of operators)
- **Representation**: Graph with nodes=operators, edges=data flow
- **Features**: Operator type embeddings, estimated/actual row counts, costs, tree depth
- **Output**: Predicted execution time + ability to rank alternative plans
- **Training**: Supervised learning from real execution traces

## System Architecture

```
Database Execution Trace
        ↓
   Extract Plans
        ↓
  Convert to Graphs
        ↓
  GNN Model (GCN/GAT)
        ↓
Cost Prediction & Plan Ranking
```

### Key Components

**Graph Representation**
- Nodes: Query operators (Scan, Hash Join, Sort, etc.)
- Edges: Data flow between operators (parent → child)
- Node Features: Operator embeddings, row estimates, execution stats

**Neural Architecture**
- Graph Convolutional Networks (GCN) or Graph Attention Networks (GAT)
- Learnable operator type embeddings
- Global graph pooling (mean, sum, or attention)
- Regression or pairwise ranking objectives

## Performance

Evaluation on 200 synthetic query plans:

| Metric | Value |
|--------|-------|
| **MAE** (Mean Absolute Error) | 452.13 ms |
| **RMSE** (Root Mean Squared Error) | 561.10 ms |
| **R² Score** | 0.286 |
| **Spearman Rank Correlation** | 0.659 |
| **Kendall's τ** | 0.468 |
| **Top-1 Plan Ranking Accuracy** | 50.0% |

The model learns meaningful cost patterns and can rank alternative plans with ~66% correlation to actual execution times.

## Project Structure

```
Neural_optimizer/
├── config/
│   └── default.yaml              # Training & model configuration
├── src/
│   ├── config.py                 # Config loading & hyperparameters
│   ├── logging_config.py         # Structured logging setup
│   ├── graph_builder.py          # Convert plan trees to graphs
│   ├── feature_encoder.py        # Operator vocabulary & node features
│   ├── dataset.py                # PyTorch Dataset for plan data
│   ├── model.py                  # GCN/GAT neural network models
│   └── __init__.py
├── scripts/
│   ├── generate_synthetic_plans.py    # Generate synthetic query plans
│   ├── extract_plans.py               # PostgreSQL plan extraction
│   ├── train.py                       # Training loop & checkpoint saving
│   ├── evaluate.py                    # Evaluation metrics & analysis
│   ├── visualize.py                   # t-SNE visualization & ablations
│   └── analyze_errors.py              # Error analysis tools
├── checkpoints/
│   └── best.pt                   # Trained model weights
├── data/
│   └── plans_synthetic/          # Synthetic training data (200 plans)
├── outputs/
│   └── metrics.json              # Evaluation results
├── requirements.txt
├── README.md
└── EXPLANATION.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU training)
- PostgreSQL (optional, only for real data extraction)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/neural-query-optimizer.git
cd neural-query-optimizer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set Python path** (required for imports):
```bash
# Windows (PowerShell)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

$env:PYTHONPATH="."

# Windows (cmd)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

set PYTHONPATH=.

# Linux/macOS

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

export PYTHONPATH=.
```

### Quick Start with Synthetic Data

No database required! Train and evaluate in minutes:

```bash
# 1. Generate 200 synthetic query plans

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/generate_synthetic_plans.py --output data/plans_synthetic --num 200

# 2. Train the model

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/train.py --data data/plans_synthetic --checkpoint checkpoints

# 3. Evaluate performance

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/evaluate.py --data data/plans_synthetic --checkpoint checkpoints/best.pt

# 4. Visualize embeddings (t-SNE)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/visualize.py tsne --data data/plans_synthetic --output outputs/tsne.png
```

### Using Real Database Data (PostgreSQL + TPC-H)

For production evaluation:

```bash
# 1. Load TPC-H dataset into PostgreSQL

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# (Follow TPC-H installation guide)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


# 2. Extract real execution plans

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/extract_plans.py \
  --host localhost \
  --dbname tpch \
  --user postgres \
  --output data/plans_real

# 3. Train on real data

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/train.py --data data/plans_real --checkpoint checkpoints

# 4. Evaluate

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179?style=flat)](https://pyg.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Query%20Plans-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

python scripts/evaluate.py --data data/plans_real --checkpoint checkpoints/best.pt
```

## Configuration

All settings are controlled via `config/default.yaml`. Key parameters:

```yaml
model:
  type: gcn                      # Model: 'gcn' or 'gat'
  operator_embedding_dim: 16     # Size of learned operator embeddings
  pooling: attention             # Graph pooling: 'mean', 'add', or 'attention'
  predict_cardinality: false     # Multi-task: also predict cardinality

training:
  mode: regression               # 'regression' or 'ranking' (ranking requires multiple plans/query)
  epochs: 100
  batch_size: 32
  lr: 0.001
  patience: 10                   # Early stopping patience
  ranking_margin: 1.0            # Margin for pairwise ranking loss

data:
  force_alternative_plans: false # Generate multiple plans per query using hints
  max_plans_per_query: 5
```

### Training Modes

- **Regression**: Predict execution time directly (MSE loss)
- **Ranking**: Learn to rank plans by cost relative to each other (pairwise margin loss)

### Generating Alternative Plans for Ranking

To train a ranking model, you need multiple plans per query. Use PostgreSQL planner hints:

```sql
SET enable_hashjoin = off;      -- Force NestLoop instead
SET enable_mergejoin = off;
EXPLAIN (FORMAT JSON) SELECT ...
```

The config defines hint combinations that generate multiple plans per query for training ranking models.

## Evaluation Metrics

- **Regression Mode**:
  - **MAE** (Mean Absolute Error): Average prediction error in milliseconds
  - **RMSE** (Root Mean Squared Error): Penalizes larger errors
  - **R² Score**: Proportion of variance explained (0-1)

- **Ranking Mode**:
  - **Spearman Correlation**: How well the model ranks plans by actual cost
  - **Kendall's τ**: Rank correlation robust to ties
  - **Top-1 Accuracy**: Does the model pick the fastest plan?

## Advanced Usage

### Error Analysis

Identify problematic query patterns:
```bash
python scripts/analyze_errors.py --data data/plans_synthetic --checkpoint checkpoints/best.pt
```

### Visualization

Generate t-SNE plots of learned operator embeddings:
```bash
python scripts/visualize.py tsne --data data/plans_synthetic --output outputs/tsne.png
```

### Ablation Studies

Systematically remove model components to measure their contribution:
```bash
python scripts/visualize.py ablation --data data/plans_synthetic --checkpoint checkpoints/best.pt
```

## Implementation Highlights

- **Graph Representation**: Query plans as directed acyclic graphs (DAGs)
- **Scalability**: Batched processing, supports variable-sized graphs
- **Flexibility**: Regression or ranking objectives, single/multi-task learning
- **Reproducibility**: Fixed random seeds, configuration-driven experiments
- **Production-Ready**: Model checkpointing, early stopping, comprehensive logging

## Future Directions

- [ ] Cardinality estimation as auxiliary task
- [ ] Attention visualization for interpretability
- [ ] Integration with real query optimizers (PostgreSQL extension)
- [ ] Evaluation on TPC-H benchmarks
- [ ] Comparison with hand-tuned cost models

## Citation

If you use this project in your research, please cite:

```bibtex
@software{neural_query_optimizer_2026,
  title={Neural Query Cost Model},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/neural-query-optimizer}
}
```

## License

MIT
