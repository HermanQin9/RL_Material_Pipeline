#!/usr/bin/env python3
"""Test script for Clear_Version components"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

print("=== Testing Clear_Version Components ===")

# Test 1: Config import
print("\n1. Testing config import...")
try:
    import config
    print("✓ Config loaded successfully")
    print(f"  - API_KEY: {config.API_KEY[:10]}..." if config.API_KEY else "  - API_KEY: None")
    print(f"  - TEST_MODE: {config.TEST_MODE}")
    print(f"  - BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"  - N_TOTAL: {config.N_TOTAL}")
except Exception as e:
    print(f"✗ Config import failed: {e}")

# Test 2: Methods import
print("\n2. Testing methods import...")
try:
    from methods.data_methods import fetch_and_featurize, impute_data, feature_matrix, feature_selection, scale_features
    print("✓ Data methods imported successfully")
    
    from methods.model_methods import train_rf, train_gbr, train_lgbm, train_xgb, train_cat
    print("✓ Model methods imported successfully")
except Exception as e:
    print(f"✗ Methods import failed: {e}")

# Test 3: Nodes import
print("\n3. Testing nodes import...")
try:
    from nodes import DataFetchNode, ImputeNode, FeatureMatrixNode, FeatureSelectionNode, ScalingNode, ModelTrainingNode
    print("✓ All nodes imported successfully")
except Exception as e:
    print(f"✗ Nodes import failed: {e}")

# Test 4: Pipeline import
print("\n4. Testing pipeline import...")
try:
    from pipeline import run_pipeline
    print("✓ Pipeline imported successfully")
except Exception as e:
    print(f"✗ Pipeline import failed: {e}")

# Test 5: Data fetching (small test)
print("\n5. Testing data fetch...")
try:
    from methods.data_methods import fetch_and_featurize
    result = fetch_and_featurize(cache=True)
    print("✓ Data fetch successful!")
    print(f"  - Result keys: {list(result.keys())}")
    if 'train_data' in result and hasattr(result['train_data'], 'shape'):
        print(f"  - Train data shape: {result['train_data'].shape}")
    if 'test_data' in result and hasattr(result['test_data'], 'shape'):
        print(f"  - Test data shape: {result['test_data'].shape}")
except Exception as e:
    print(f"✗ Data fetch failed: {e}")

print("\n=== Test completed ===")
