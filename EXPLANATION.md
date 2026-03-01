# Neural Query Cost Model: Explanation

## What is this project about?

This project implements a **learned cost model** for database query optimization. Traditional database systems use handcrafted heuristics to estimate query execution costs, which often leads to suboptimal query plans. This project replaces those heuristics with a machine learning model that can learn the true cost behavior from actual execution data.

## How It Works

### 1. Core Concept
- **Input**: A query execution plan (as a tree/graph of operations)
- **Output**: Predicted execution time and resource usage
- **Key Idea**: Use Graph Neural Networks (GNNs) to process the query plan structure and operator features

### 2. Data Flow
```
Query Plan → Graph Representation → Neural Network → Cost Prediction
```

### 3. Key Components

#### a. Plan as a Graph
- **Nodes**: Represent database operations (scans, joins, sorts, etc.)
- **Edges**: Represent data flow between operations
- **Node Features**:
  - Operator type (encoded as embedding)
  - Estimated rows
  - Actual rows
  - Operation-specific metrics

#### b. Model Architecture
- **Graph Neural Network (GNN)**: Processes the query plan graph
- **Operator Embeddings**: Learn similarities between different types of operations
- **Global Pooling**: Aggregates node features to make a final prediction

#### c. Training
- **Objective**: Minimize the difference between predicted and actual execution times
- **Loss Function**: Mean Squared Error (MSE) for regression
- **Training Data**: Can use both real execution plans from PostgreSQL or synthetic data

## Results on Synthetic Data

### Model Performance
- **Mean Absolute Error (MAE)**: 438.95
- **Root Mean Square Error (RMSE)**: 531.47
- **R² Score**: 0.36
- **Spearman's Rank Correlation**: 0.62

### Interpretation
- The model has learned meaningful patterns from the synthetic data
- The positive Spearman correlation indicates it can correctly rank plans by cost
- The R² score suggests there's room for improvement, which is expected with synthetic data

## How to Use This Project

### 1. Generate Synthetic Data
```bash
python scripts/generate_synthetic_plans.py --output data/plans_synthetic --num 1000
```

### 2. Train the Model
```bash
python scripts/train.py --data data/plans_synthetic --checkpoint checkpoints
```

### 3. Evaluate Performance
```bash
python scripts/evaluate.py --data data/plans_synthetic --checkpoint checkpoints/best.pt
```

### 4. Visualize Results
```bash
python scripts/visualize.py tsne --data data/plans_synthetic --output outputs/tsne.png
```

## Real-World Integration

To use this with a real database:

1. **Set up PostgreSQL** with your database
2. **Extract plans**:
   ```bash
   python scripts/extract_plans.py --host localhost --dbname your_db --output data/real_plans
   ```
3. **Train** on real execution data
4. **Integrate** the trained model into your database's query optimizer

## Next Steps for Improvement

1. **More Training Data**: Use real query plans from production workloads
2. **Feature Engineering**: Add more detailed operator features
3. **Model Architecture**: Experiment with different GNN architectures
4. **Multi-Task Learning**: Predict both execution time and memory usage
5. **Transfer Learning**: Pre-train on synthetic data, fine-tune on real data

## Why This Matters

- **Better Performance**: More accurate cost models lead to faster query execution
- **Reduced Tuning**: Less need for manual query optimization
- **Adaptability**: The model can learn the characteristics of your specific workload
- **Future-Proof**: Can adapt as hardware and data distributions change
