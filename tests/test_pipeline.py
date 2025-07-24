#!/usr/bin/env python3
"""Test a complete pipeline run"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

print("=== Testing Complete Pipeline ===")

try:
    from pipeline import run_pipeline
    
    print("\nRunning complete pipeline with default parameters...")
    result = run_pipeline(
        cache=True,                    # Use cached data
        impute_strategy='mean',        # Mean imputation
        nan_thresh=0.5,                # Feature selection threshold
        train_val_ratio=0.8,           # 80% train, 20% validation
        selection_strategy='none',     # No feature selection for speed
        scaling_strategy='standard',   # Standard scaling
        model_strategy='rf',           # Random Forest
        model_params={'n_estimators': 10}  # Small number for speed
    )
    
    print("✓ Pipeline completed successfully!")
    print(f"  - Result keys: {list(result.keys())}")
    
    # Check model performance
    if 'model' in result:
        print(f"  - Model type: {type(result['model']).__name__}")
    
    if 'y_val_pred' in result and result['y_val_pred'] is not None:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(result['y_val'], result['y_val_pred'])
        print(f"  - Validation MAE: {mae:.4f}")
    
    if 'params' in result:
        print(f"  - Model parameters: {result['params']}")
        
    print(f"  - Data shapes:")
    print(f"    - X_train: {result['X_train'].shape}")
    print(f"    - X_val: {result['X_val'].shape}")
    print(f"    - y_train: {result['y_train'].shape}")
    print(f"    - y_val: {result['y_val'].shape}")
    
except Exception as e:
    import traceback
    print(f"✗ Pipeline failed: {e}")
    print("Full traceback:")
    traceback.print_exc()

print("\n=== Pipeline test completed ===")
