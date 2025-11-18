# 10-Node Architecture Documentation

## Overview

This document describes the **implemented 10-node flexible architecture** used in the PPO-driven AutoML pipeline for materials science formation energy prediction.

## Node Definitions

| Node | Name             | Type               | Available Methods                | Position        | Hyperparams |
|------|------------------|--------------------|----------------------------------|-----------------|-------------|
| **N0** | DataFetch       | Data               | `api`                            | **Fixed (start)** | No |
| **N1** | Impute          | DataProcessing     | `mean`, `median`, `knn`          | Flexible        | Yes |
| **N2** | FeatureMatrix   | FeatureEngineering | `default`                        | **Fixed (2nd)** | No |
| **N3** | Cleaning        | DataProcessing     | `outlier`, `noise`, `none`       | Flexible        | Yes |
| **N4** | GNN             | FeatureEngineering | `gcn`, `gat`, `sage`             | Flexible        | No |
| **N5** | KnowledgeGraph  | FeatureEngineering | `entity`, `relation`, `none`     | Flexible        | No |
| **N6** | FeatureSelection| FeatureEngineering | `variance`, `univariate`, `pca`  | Flexible        | Yes |
| **N7** | Scaling         | Preprocessing      | `std`, `robust`, `minmax`        | Flexible        | Yes |
| **N8** | ModelTraining   | Training           | `rf`, `gbr`, `xgb`, `cat`        | **Fixed (pre-end)** | Yes |
| **N9** | End             | Control            | `terminate`                      | **Fixed (end)** | No |

## Node Categories

### Fixed Nodes (Mandatory Execution Order)

- **N0**: Always executes first - fetches data from Materials Project API
- **N2**: Always executes second - constructs feature matrix from crystal structures
- **N8**: Always executes before termination - trains the final ML model
- **N9**: Always executes last - terminates pipeline and computes reward

### Flexible Middle Nodes (PPO Controlled)

- **N1, N3, N4, N5, N6, N7**: PPO agent decides:
  - Which nodes to execute
  - In what order
  - With which methods
  - Can be skipped if deemed unnecessary

### Parameter Nodes

The following nodes accept hyperparameters (controlled by PPO):

- **N1** (Impute): `param` ∈ [0.0, 1.0]
- **N3** (Cleaning): `param` ∈ [0.0, 1.0]
- **N6** (FeatureSelection): `param` ∈ [0.0, 1.0]
- **N7** (Scaling): `param` ∈ [0.0, 1.0]
- **N8** (ModelTraining): `param` ∈ [0.0, 1.0]

## Architecture Constraints

### Sequencing Rules

1. **Step 0**: Must execute N0 (data fetch)
2. **Step 1**: Must execute N2 (feature matrix construction)
3. **Middle Steps**: Can execute any unvisited flexible node (N1, N3, N4, N5, N6, N7) OR jump directly to N8
4. **After N8**: Must execute N9 (termination)

### Action Masking Logic

The environment enforces legal node transitions through action masks:

```python
# Step 0: Only N0 allowed
if current_step == 0:
    mask[N0] = 1.0

# Step 1: Only N2 allowed  
elif current_step == 1:
    mask[N2] = 1.0

# Middle steps: Flexible nodes or jump to N8
else:
    if N8_visited and not N9_visited:
        mask[N9] = 1.0  # After training, only termination
    else:
        # Allow unvisited flexible nodes
        for node in [N1, N3, N4, N5, N6, N7]:
            if not visited[node]:
                mask[node] = 1.0
        # Allow jump to training
        if not N8_visited:
            mask[N8] = 1.0
```

### Method Masking

Each node has a method mask to prevent invalid method selection:

```python
max_methods = 4  # Maximum methods across all nodes

# Method mask shape: [num_nodes, max_methods]
# Example for N4 (GNN) with 3 methods [gcn, gat, sage]:
method_mask[N4] = [1.0, 1.0, 1.0, 0.0]  # 4th method invalid
```

## Decision Space

### Total Combinations

The architecture provides enormous flexibility:

- **Flexible node ordering**: 6! = 720 permutations of [N1, N3, N4, N5, N6, N7]
- **Node selection**: 2^6 = 64 ways to choose which flexible nodes to execute
- **Method selection**: 3×3×3×3×3×3 = 729 method combinations
- **Hyperparameter tuning**: Continuous space [0.0, 1.0]^5 for param nodes

**Practical combinations**: Millions of possible pipelines

## Example Valid Sequences

### Minimal Pipeline

```
N0 → N2 → N8 → N9
```

- Skips all flexible preprocessing
- Uses default features
- Direct model training

### Standard Pipeline

```
N0 → N2 → N1 → N6 → N7 → N8 → N9
```

- Imputation → Feature selection → Scaling → Model
- Classic ML preprocessing sequence

### Advanced Pipeline with GNN

```
N0 → N2 → N3 → N4 → N1 → N5 → N6 → N7 → N8 → N9
```

- Cleaning → GNN → Imputation → KG → Selection → Scaling → Model
- Leverages graph neural networks for crystal structures
- Incorporates knowledge graph enrichment

### GNN-Focused Pipeline

```
N0 → N2 → N4 → N5 → N7 → N8 → N9
```

- GNN → Knowledge Graph → Scaling → Model
- Skips traditional imputation and feature selection
- Relies on graph representations

### Invalid Sequences (Blocked by Masking)

❌ `N2 → N0` - Cannot execute N0 after N2  
❌ `N0 → N8 → N9` - Cannot skip N2 (feature matrix required)  
❌ `N0 → N2 → N1 → N1` - Cannot execute same node twice  
❌ `N0 → N2 → N9` - Cannot terminate without training (N8)

## Implementation Details

### Environment Configuration

From `env/pipeline_env.py`:

```python
pipeline_nodes = ['N0', 'N2', 'N1', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']

methods_for_node = {
    'N0': ['api'],
    'N1': ['mean', 'median', 'knn'],
    'N2': ['default'],
    'N3': ['outlier', 'noise', 'none'],
    'N4': ['gcn', 'gat', 'sage'],
    'N5': ['entity', 'relation', 'none'],
    'N6': ['variance', 'univariate', 'pca'],
    'N7': ['std', 'robust', 'minmax'],
    'N8': ['rf', 'gbr', 'xgb', 'cat'],
    'N9': ['terminate']
}

param_nodes = {'N1', 'N3', 'N6', 'N7', 'N8'}
```

### Node Classes

From `nodes.py`:

- **DataFetchNode** (N0): Fetches material data from Materials Project API
- **ImputeNode** (N1): Handles missing values with multiple strategies
- **FeatureMatrixNode** (N2): Constructs feature matrix from crystal structures
- **CleaningNode** (N3): Outlier detection and noise filtering
- **GNNNode** (N4): Graph neural network processing (placeholder)
- **KGNode** (N5): Knowledge graph enrichment (placeholder)
- **SelectionNode** (N6): Feature selection via variance/univariate/PCA
- **ScalingNodeB** (N7): Feature scaling with different strategies
- **ModelTrainingNodeB** (N8): Trains regression models (RF, GBR, XGB, CatBoost)
- **EndNode** (N9): Pipeline termination and reward computation

## PPO Integration

### Action Space

```python
action = {
    'node': int,      # Node index [0-9]
    'method': int,    # Method index [0-3]
    'params': list    # Hyperparameters [0.0-1.0]
}
```

### Observation Space

```python
observation = {
    'fingerprint': np.array([mae, r2, n_features], dtype=float32),
    'node_visited': np.array([0/1] * 10, dtype=float32),
    'action_mask': np.array([0/1] * 10, dtype=float32),
    'method_count': np.array([num_methods_per_node], dtype=int32),
    'method_mask': np.array([[0/1] * 4] * 10, dtype=float32)
}
```

### Reward Computation

Triggered at N9 termination:

```python
reward = r2_score - mae - complexity_penalty - repetition_penalty
```

- **r2_score**: Model R² on test set (higher is better)
- **mae**: Mean absolute error on test set (lower is better)
- **complexity_penalty**: Penalizes complex methods (0.1-0.3)
- **repetition_penalty**: -0.5 for repeated method calls

## Detailed Node Descriptions

### N0: DataFetch

**Purpose**: Fetch crystalline material data from Materials Project API

**Methods**:

- `api`: Query Materials Project for formation energy data

**Implementation**: `methods.data_methods.fetch_and_featurize()`

**Outputs**: Raw material structures and target properties

### N1: Impute

**Purpose**: Handle missing values in feature matrix

**Methods**:

- `mean`: Replace missing with column mean
- `median`: Replace missing with column median
- `knn`: K-nearest neighbors imputation

**Hyperparameter**: Controls imputation strategy specifics

**Implementation**: `methods.data_methods.impute_data()`

### N2: FeatureMatrix

**Purpose**: Construct numerical feature matrix from crystal structures

**Methods**:

- `default`: Standard featurization (composition, structure, properties)

**Implementation**: `methods.data_methods.feature_matrix()`

**Outputs**: X_train, X_val, y_train, y_val, feature_names

### N3: Cleaning

**Purpose**: Data quality improvement through outlier/noise handling

**Methods**:

- `outlier`: Detect and remove statistical outliers
- `noise`: Apply noise filtering techniques
- `none`: Skip cleaning

**Hyperparameter**: Controls cleaning aggressiveness

**Implementation**: `methods.data.preprocessing.clean_data()`

### N4: GNN (Graph Neural Network)

**Purpose**: Process crystal structures as graphs for enhanced representations

**Methods**:

- `gcn`: Graph Convolutional Network
- `gat`: Graph Attention Network
- `sage`: GraphSAGE

**Status**: Placeholder implementation

**Implementation**: `methods.data.preprocessing.gnn_process()`

### N5: KnowledgeGraph

**Purpose**: Enrich features with materials science domain knowledge

**Methods**:

- `entity`: Entity-based knowledge integration
- `relation`: Relation-based knowledge integration
- `none`: Skip knowledge graph enrichment

**Status**: Placeholder implementation

**Implementation**: `methods.data.preprocessing.kg_process()`

### N6: FeatureSelection

**Purpose**: Reduce dimensionality by selecting most informative features

**Methods**:

- `variance`: Remove low-variance features
- `univariate`: Univariate statistical tests
- `pca`: Principal Component Analysis

**Hyperparameter**: Controls selection threshold/components

**Implementation**: `methods.data_methods.feature_selection()`

### N7: Scaling

**Purpose**: Normalize feature ranges for model training

**Methods**:

- `std`: StandardScaler (zero mean, unit variance)
- `robust`: RobustScaler (median/IQR, resistant to outliers)
- `minmax`: MinMaxScaler (scale to [0, 1])

**Hyperparameter**: Controls scaling parameters

**Implementation**: `methods.data_methods.scale_features()`

### N8: ModelTraining

**Purpose**: Train regression model for formation energy prediction

**Methods**:

- `rf`: Random Forest Regressor
- `gbr`: Gradient Boosting Regressor
- `xgb`: XGBoost Regressor
- `cat`: CatBoost Regressor

**Hyperparameter**: Controls model complexity

**Implementation**: `methods.model_methods.train_*()` functions

**Outputs**: Trained model, metrics (MAE, R², RMSE)

### N9: End

**Purpose**: Terminate pipeline and compute reward

**Methods**:

- `terminate`: Pipeline completion

**Implementation**: `methods.data.preprocessing.terminate()`

**Triggers**: Full pipeline evaluation and reward computation

## Design Philosophy

1. **Flexibility**: PPO explores vast search space of pipelines
2. **Efficiency**: Action masking prevents illegal/wasteful actions
3. **Modularity**: Each node is independent and composable
4. **Extensibility**: New nodes can be added without breaking existing logic
5. **Interpretability**: Clear node sequencing reveals discovered strategies

## Performance Considerations

### Computational Complexity

- **Minimal pipeline** (N0→N2→N8→N9): ~5-10 seconds
- **Standard pipeline** (+N1, N6, N7): ~10-20 seconds
- **Advanced pipeline** (all nodes): ~20-40 seconds

### Memory Requirements

- **Feature matrix**: ~50-500 MB depending on dataset size
- **GNN processing**: Additional ~100-500 MB (when implemented)
- **Model training**: ~100-200 MB for ensemble models

## Future Extensions

1. **GNN Implementation**: Replace N4 placeholder with actual graph convolution
   - Use PyTorch Geometric or DGL
   - Process crystal structures as graphs (atoms as nodes, bonds as edges)

2. **Knowledge Graph**: Implement N5 with materials science ontology
   - Integrate Materials Project knowledge base
   - Add periodic table relationships
   - Include crystal system hierarchies

3. **Multi-objective Optimization**: Extend reward function
   - Balance accuracy, speed, and interpretability
   - Add Pareto frontier exploration

4. **Transfer Learning**: Pre-train on related tasks
   - Band gap prediction
   - Bulk modulus estimation
   - Multi-property learning

5. **Explainability**: Add node for interpretability
   - SHAP values
   - Feature importance visualization
   - Decision path analysis

## References

- **Code**: `env/pipeline_env.py`, `nodes.py`
- **Methods**: `methods/data_methods.py`, `methods/model_methods.py`
- **PPO**: `ppo/trainer.py`, `ppo/policy.py`
- **Tests**: `tests/test_pipeline.py`, `tests/test_ppo.py`
