# Node Architecture Summary

## Planned 10-Node Upgrade

### Extended Node List
| Node | Purpose | Methods | Position | Status |
|------|---------|---------|----------|--------|
| **N0** | Data Fetch | `api` | Fixed (start) | Existing |
| **N1** | **[Imputation A]** | `mean`, `median`, `knn` | Flexible | Modified |
| **N2** | Feature Matrix | `default` | Fixed (after N0) | Existing |
| **N3** | **[Cleaning A]** | `outlier`, `noise`, `none` | Flexible | **New** |
| **N4** | **[GNN Processing]** | `gcn`, `gat`, `sage` | Flexible | **New** |
| **N5** | **[Knowledge Graph]** | `entity`, `relation`, `none` | Flexible | **New** |
| **N6** | **[Feature Selection]** | `variance`, `univariate`, `pca` | Flexible | Modified |
| **N7** | **[Scaling B]** | `std`, `robust`, `minmax` | Flexible | Modified |
| **N8** | Model Training | `rf`, `gbr`, `xgb`, `cat` | Fixed (pre-end) | Existing |
| **N9** | Pipeline End | `terminate` | Fixed (end) | **New** |

### New Nodes Added (10-Node System)
- **N3**: **[Cleaning A]** - outlier detection, noise filtering
- **N4**: **[GNN Processing]** - graph neural networks for crystal structures  
- **N5**: **[Knowledge Graph]** - materials science knowledge integration

### Upgrade Architecture Options

#### Option 1: Fixed Group Order (Current)
```
N0 → N2 → [Group A: N1,N3] → [Group B: N4,N5] → [Group C: N6,N7] → N8 → N9
```
- **Group ordering**: Fixed A→B→C
- **Within groups**: Flexible ordering
- **Total combinations**: 2×2×2×729 = **5,832 combinations**

#### Option 2: Flexible Group Order  
```
N0 → N2 → [Any Group Order: A,B,C] → N8 → N9
```
- **Group ordering**: 3! = 6 ways (A→B→C, A→C→B, B→A→C, etc.)
- **Within groups**: Flexible ordering  
- **Total combinations**: 6×8×729 = **34,992 combinations**

#### Option 3: Complete Flexibility
```
N0 → N2 → [N1,N3,N4,N5,N6,N7 in any order] → N8 → N9
```
- **Node ordering**: 6! = 720 ways
- **Method selection**: 729 ways
- **Total combinations**: 720×729 = **524,880 combinations**

### Node Constraints
- **Fixed positions**: N0 (start), N2 (second), N8 (pre-end), N9 (end)
- **PPO controlled**: **[N1]**, **[N3]**, **[N4]**, **[N5]**, **[N6]**, **[N7]**
- **Decision space**: Node ordering + method selection

### Recommended Approach
**Option 2 (Flexible Group Order)** is recommended because:
- Maintains logical grouping for related functions
- Provides significant flexibility (34,992 combinations)
- Manageable complexity for PPO optimization
- Allows creative processing sequences while preserving constraints

### Example Sequences (Option 2)
1. **A→B→C**: `N0 → N2 → [N1→N3] → [N4→N5] → [N6→N7] → N8 → N9`
2. **B→A→C**: `N0 → N2 → [N4→N5] → [N1→N3] → [N6→N7] → N8 → N9`  
3. **C→B→A**: `N0 → N2 → [N6→N7] → [N4→N5] → [N1→N3] → N8 → N9`
