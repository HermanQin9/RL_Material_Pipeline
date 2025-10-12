# Node Architecture Summary

> **⚠️ This file is deprecated. Please refer to the comprehensive documentation:**
> 
> **📘 [10-NODE_ARCHITECTURE.md](docs/10-NODE_ARCHITECTURE.md)**

## Quick Reference

The project implements a **10-node flexible architecture** for PPO-driven AutoML:

### Node List (Current Implementation)

| Node | Name             | Methods                          | Position        |
|------|------------------|----------------------------------|-----------------|
| N0   | DataFetch        | `api`                            | Fixed (start)   |
| N1   | Impute           | `mean`, `median`, `knn`          | Flexible        |
| N2   | FeatureMatrix    | `default`                        | Fixed (2nd)     |
| N3   | Cleaning         | `outlier`, `noise`, `none`       | Flexible        |
| N4   | GNN              | `gcn`, `gat`, `sage`             | Flexible        |
| N5   | KnowledgeGraph   | `entity`, `relation`, `none`     | Flexible        |
| N6   | FeatureSelection | `variance`, `univariate`, `pca`  | Flexible        |
| N7   | Scaling          | `std`, `robust`, `minmax`        | Flexible        |
| N8   | ModelTraining    | `rf`, `gbr`, `xgb`, `cat`        | Fixed (pre-end) |
| N9   | End              | `terminate`                      | Fixed (end)     |

### Execution Flow

```
N0 (start) → N2 (fixed) → [N1,N3,N4,N5,N6,N7 - flexible order] → N8 (fixed) → N9 (end)
```

### Key Features

- **Fixed nodes**: N0, N2, N8, N9 (mandatory execution order)
- **Flexible nodes**: N1, N3, N4, N5, N6, N7 (PPO controls order and selection)
- **Action masking**: Enforces legal node sequences
- **Method masking**: Prevents invalid method selection
- **Millions of combinations**: Flexible ordering × method selection × hyperparameters

### Example Pipeline

```
N0 → N2 → N1 → N6 → N7 → N8 → N9
```

### For Detailed Information

See **[docs/10-NODE_ARCHITECTURE.md](docs/10-NODE_ARCHITECTURE.md)** for:

- Complete node descriptions
- Action masking logic
- Method-level masking
- Example valid/invalid sequences
- PPO integration details
- Implementation references
- Future extensions
