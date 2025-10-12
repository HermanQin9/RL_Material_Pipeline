# GitHub Copilot Instructions for Materials Science ML Pipeline

## 🎯 Project Overview

This is an advanced **PPO Reinforcement Learning-driven AutoML pipeline** for materials science formation energy prediction. The system uses intelligent agents to automatically construct optimal machine learning pipelines through node-based architecture and Materials Project API integration.

### 🔬 Domain Context
- **Materials Science**: Formation energy prediction for crystalline materials
- **Data Source**: Materials Project API with 4K+ material datasets  
- **Target Property**: `formation_energy_per_atom` prediction
- **ML Focus**: Automated feature engineering and model selection

## 🏗️ Architecture Understanding

### Node-Based Pipeline System
The core architecture implements a **10-node flexible execution system** with PPO-controlled sequencing:

```
N0 (DataFetch) → N2 (FeatureMatrix) → [Flexible Middle Nodes] → N8 (ModelTraining) → N9 (End)
```

#### Current 10-Node Implementation
| Node | Purpose | Methods Available | Position |
|------|---------|------------------|----------|
| **N0** | Data Fetch | `api` | Fixed (start) |
| **N1** | Imputation | `mean`, `median`, `knn` | Flexible |
| **N2** | Feature Matrix | `default` | Fixed (2nd) |
| **N3** | Cleaning | `outlier`, `noise`, `none` | Flexible |
| **N4** | GNN | `gcn`, `gat`, `sage` | Flexible |
| **N5** | Knowledge Graph | `entity`, `relation`, `none` | Flexible |
| **N6** | Feature Selection | `variance`, `univariate`, `pca` | Flexible |
| **N7** | Scaling | `std`, `robust`, `minmax` | Flexible |
| **N8** | Model Training | `rf`, `gbr`, `xgb`, `cat` | Fixed (pre-end) |
| **N9** | End | `terminate` | Fixed (end) |

#### Architecture Features
- **Fixed Nodes**: N0 (start), N2 (second), N8 (pre-end), N9 (end)
- **Flexible Nodes**: N1, N3, N4, N5, N6, N7 - PPO decides order and selection
- **Decision Space**: Millions of combinations (node ordering × method selection × hyperparameters)
- **New Components**: GNN processing (N4), Knowledge Graph enrichment (N5)

### PPO Reinforcement Learning Core
- **Agent Role**: Automatically selects optimal node sequences and methods
- **Action Space**: `{'node': int, 'method': int, 'params': list}`
- **Reward System**: Based on validation performance and pipeline efficiency
- **Learning Rate**: Adaptive optimization for gradient-based policy updates

## 📁 Codebase Structure

### Core Modules
```
env/
├── pipeline_env.py          # PipelineEnv class - RL environment
└── utils.py                 # Environment utilities

nodes.py                     # Node base class and implementations
pipeline.py                  # Main pipeline execution logic
config.py                    # Global configuration and paths

methods/
├── data_methods.py          # Data processing functions
└── model_methods.py         # ML model implementations

ppo/
├── trainer.py               # PPO training logic
├── policy.py                # Neural network policies
├── buffer.py                # Experience replay buffer
└── utils.py                 # PPO utilities
```

### Key Files to Understand

#### `env/pipeline_env.py`
- **PipelineEnv class**: Core RL environment with 10-node architecture
- **Action validation**: Ensures valid node sequences (N0→N2 fixed, flexible middle, N8→N9 end)
- **State management**: Tracks node visits and pipeline configuration
- **Reward calculation**: Performance-based learning signals (triggered at N9)

#### `nodes.py`
- **Node base class**: Common interface for all pipeline steps
- **Specialized nodes**: DataFetchNode, FeatureMatrixNode, ImputeNode, etc.
- **Method execution**: Each node has multiple available methods

#### `methods/data_methods.py`
- **Feature engineering**: `feature_matrix()`, `feature_selection()`
- **Data preprocessing**: `impute_data()`, `scale_features()`
- **State management**: `prepare_node_input()`, `update_state()`

## 🤖 PPO Implementation Details

### Action Space Structure
```python
action = {
    'node': int,      # Index into pipeline_nodes [0-9]
    'method': int,    # Index into node's available methods [0-3]
    'params': list    # Hyperparameters [0.0-1.0]
}
```

### Training Flow
1. **Reset Environment**: Initialize empty pipeline state
2. **Select Node**: PPO agent chooses next node and method
3. **Execute Action**: Run selected node with chosen method
4. **Compute Reward**: Evaluate pipeline performance
5. **Update Policy**: Gradient-based learning with clip ratio

### Key PPO Components
- **Policy Network**: Neural network for action selection
- **Value Network**: State value estimation for advantage calculation
- **Experience Buffer**: Trajectory storage for batch training
- **Clip Ratio**: 0.2 for stable policy updates

## 🧪 Materials Science Integration

### Materials Project API
- **Data Source**: Crystalline structure and formation energy data
- **Caching**: Local storage at `data/processed/mp_data_cache_*.pkl`
- **Batch Processing**: Configurable batch sizes (20 test, 100 production)

### Feature Engineering Pipeline
- **Crystal Features**: Lattice parameters, atomic properties, composition
- **Missing Value Handling**: Multiple imputation strategies
- **Feature Selection**: Variance, univariate, and PCA methods
- **Scaling**: Standard, robust, and min-max normalization

### Model Training
- **Algorithms**: Random Forest, Gradient Boosting, XGBoost, CatBoost
- **Validation**: Train/validation splits with performance metrics
- **Hyperparameter Optimization**: PPO-driven parameter tuning

## 💡 Development Guidelines

### When Working with Nodes
```python
# Always use the node execution pattern
node_input = prepare_node_input(node_key, state, verbose=True)
node_output = node.execute(method, params, node_input)
update_state(node_key, node_output, state, verbose=True)
```

### PPO Development Patterns
```python
# Action validation is critical
if not env.select_node(action):
    return False  # Invalid action sequence

# Always check step constraints
if env.current_step == 0 and node_name != 'N2':
    return False  # First step must be N2
```

### State Management
- **Input Validation**: Use `validate_state_keys()` for required data
- **Data Consistency**: Ensure X_train, y_train, X_val, y_val alignment
- **Memory Management**: Clean up intermediate states between nodes

## 🚧 Current Development Phase

### Completed Features
- ✅ 10-node flexible architecture fully implemented
- ✅ Action masking and method-level masking
- ✅ 4K dataset training validation  
- ✅ Comprehensive testing and debugging
- ✅ Complete documentation (10-NODE_ARCHITECTURE.md, NODE_ARCHITECTURE_SUMMARY.md)
- ✅ Function reorganization and code deduplication

### Active Development
- 🔄 GNN implementation (currently placeholder)
- 🔄 Knowledge Graph integration (currently placeholder)
- 🔄 PPO policy optimization for 10-node space

### Historical Milestones
- v1.0: Initial 5-node pipeline with basic PPO (85% success rate)
- v2.0: 10-node flexible architecture with millions of combinations

### Code Quality Standards
- **Error Handling**: Robust exception management for missing data
- **Logging**: Comprehensive logging with Chinese/English bilingual comments
- **Testing**: Extensive validation in `tests/` directory
- **Documentation**: Bilingual documentation throughout codebase

## 🎮 Common Patterns

### Node Execution Pattern
```python
# Standard node execution flow
try:
    input_data = prepare_node_input('N3', state)
    output = feature_selection_node.execute('select', params, input_data)
    update_state('N3', output, state)
except Exception as e:
    logger.error(f"Node N3 execution failed: {e}")
    return None
```

### PPO Training Pattern
```python
# Typical PPO episode structure
obs = env.reset()
for step in range(max_steps):
    action, log_probs = trainer.select_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

### Configuration Management
```python
# Use config.py for all settings
from config import PROC_DIR, MODEL_DIR, TARGET_PROP, TEST_MODE
```

## 🔧 Development Tools

### Essential Commands
```bash
# Run PPO training
python ppo/trainer.py --episodes 100

# Test pipeline components  
python tests/test_pipeline.py

# Validate all models
python tests/test_all_models.py
```

### Debugging Utilities
- **PPO_Testing_and_Debugging.ipynb**: Comprehensive testing notebook
- **debug_pipeline.py**: Pipeline debugging utilities
- **test_components.py**: Individual component validation

## 📊 Performance Expectations

### Training Metrics
- **Success Rate**: Target 85%+ for valid pipeline completion
- **Processing Speed**: 695K samples/sec on test datasets
- **Memory Usage**: Efficient state management for 4K+ materials
- **Convergence**: PPO policy convergence within 100-500 episodes

### Materials Science Accuracy
- **Formation Energy MAE**: Target <0.1 eV/atom
- **R² Score**: Target >0.85 for validation set
- **Feature Importance**: Interpretable materials physics relationships

## 🌟 Best Practices

### Code Style
- **Bilingual Comments**: Chinese/English for international collaboration
- **Type Hints**: Use typing module for all function signatures
- **Error Messages**: Descriptive error handling with context
- **Logging**: Use logging module instead of print statements

### Testing Strategy
- **Unit Tests**: Individual node and method validation
- **Integration Tests**: Full pipeline execution tests
- **Performance Tests**: Memory and speed benchmarking
- **Edge Cases**: Missing data and invalid action handling

### Git Workflow
- **Branch Naming**: `feature/10-node-upgrade`, `fix/ppo-convergence`
- **Commit Messages**: Clear, descriptive commits with scope
- **Documentation**: Update docs/ for significant changes

Remember: This is a sophisticated AI system combining reinforcement learning, materials science, and automated machine learning. Always consider the PPO agent's learning process when making changes to the pipeline structure or reward mechanisms.
