# Environment Setup Guide

This project provides four environment configuration files for different use cases:

### 1. `environment.yml` - Full Development Environment
**Purpose**: Local development with all features
**Python**: 3.11
**Features**: 
- Complete dependency set (366 packages)
- Includes PyTorch Geometric (full GNN support)
- All visualization and analysis tools
- Jupyter Lab support

**Installation**:
```bash
conda env create -f environment.yml
conda activate summer_project_2025
```

### 2. `environment-ci.yml` - CI Testing Environment
**Purpose**: GitHub Actions CI/CD
**Python**: 3.8, 3.9, 3.10 (matrix)
**Features**:
- Minimal dependencies
- Multi-Python version compatibility
- Fast installation
- GNN statistical fallback

**Installation**:
```bash
conda env create -f environment-ci.yml
conda activate matformppo-ci
```

### 3. `requirements.txt` - Core Pip Dependencies
**Purpose**: Production deployment via pip
**Features**:
- Core dependency packages
- Includes torch-geometric (optional)
- Suitable for containerization

**Installation**:
```bash
pip install -r requirements.txt
```

### 4. `requirements-full.txt` - Complete Pip Freeze
**Purpose**: Exact version reproduction
**Features**:
- Precise versions of 476 packages
- pip freeze output
- For exact environment replication

**Installation**:
```bash
pip install -r requirements-full.txt
```

## Python Version Compatibility

| Python Version | Support Status | Notes |
|---------------|----------------|-------|
| 3.11 | ✅ Recommended | Full functionality, best performance |
| 3.10 | ✅ Supported | CI tested |
| 3.9 | ✅ Supported | CI tested |
| 3.8 | ✅ Supported | Minimum version, CI tested |

## GNN Functionality

### When PyTorch Geometric is Available
- Full GNN implementation (GCN, GAT, GraphSAGE)
- k-NN graph construction
- Graph convolutional feature extraction
- Optimal performance

### When PyTorch Geometric is Unavailable
- Automatic fallback to statistical features
- 11 graph statistical features
- No additional dependencies required
- Pipeline runs normally

## Troubleshooting

### 1. PyTorch Geometric Installation Failed
```bash
# This is normal, system will use fallback
# If you need full GNN functionality, install manually:
pip install torch-geometric
# or
conda install pytorch-geometric -c pyg
```

### 2. Materials Project API Key
```bash
# Set environment variable
export MP_API_KEY="your_api_key_here"
# or configure in config.py
```

### 3. Windows Path Issues
```bash
# In PowerShell:
$env:PYTHONPATH = "$PWD"
```

### 4. Test Failures
```bash
# Run quick validation
pytest tests/quick_test.py -v

# Skip GNN tests (if PyG not installed)
pytest tests/ -v -k "not gnn"
```

## Environment Verification

Verify your environment is correctly configured:

```bash
# 检查Python版本
python --version

# 检查核心包
python -c "import numpy, pandas, sklearn, torch; print('Core packages OK')"

# 检查材料科学包
python -c "import pymatgen, matminer; print('Materials packages OK')"

# 检查GNN支持
python -c "try: import torch_geometric; print('GNN: Full support'); except: print('GNN: Statistical fallback')"

# 运行快速测试
pytest tests/quick_test.py -v
```

## CI/CD Environment

GitHub Actions uses the following environment:
- Python 3.8, 3.9, 3.10 (matrix)
- Ubuntu Latest & Windows Latest
- CPU-only PyTorch
- GNN statistical fallback (if PyG installation fails)

All tests should pass without PyTorch Geometric.

## Updating Environments

### Update conda environment
```bash
conda env update -f environment.yml --prune
```

### Update pip packages
```bash
pip install --upgrade -r requirements.txt
```

### Export new environment
```bash
# Conda environment
conda env export > environment.yml

# Pip packages
pip freeze > requirements-full.txt
```
