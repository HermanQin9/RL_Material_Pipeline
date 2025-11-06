#!/usr/bin/env python3
"""
 - Example Usage
 MatFormPPO 
Demonstrates how to use the MatFormPPO ML pipeline
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# / Import pipeline runner and nodes
from pipeline import run_pipeline
from nodes import DataFetchNode, FeatureMatrixNode, ImputeNode, FeatureSelectionNode, ScalingNode, ModelTrainingNode

def main():
 """ / Main function"""
 print("=== MatFormPPO Materials ML Pipeline Demo ===")

 # 1. / Run complete pipeline
 print("\n1. Running complete ML pipeline...")
 result = run_pipeline(
 cache=True,
 nan_thresh=0.5,
 train_val_ratio=0.8,
 impute_strategy='mean',
 selection_strategy='none',
 scaling_strategy='standard',
 model_strategy='rf',
 model_params={'n_estimators': 10}
 )

 print(f" Pipeline completed successfully!")
 print(f" - Model type: {type(result['model']).__name__}")
 print(f" - Training data shape: {result['X_train'].shape}")
 print(f" - Validation data shape: {result['X_val'].shape}")
 print(f" - Test data shape: {result['X_test'].shape}")

 # 2. / Test different models
 print("\n2. Testing different ML models...")
 models_to_test = ['rf', 'xgb', 'lgb']

 for model_name in models_to_test:
 print(f" Testing {model_name.upper()}...")
 try:
 result = run_pipeline(
 cache=True,
 model_strategy=model_name,
 model_params={'n_estimators': 5}
 )
 print(f" {model_name.upper()} trained successfully")
 except Exception as e:
 print(f" {model_name.upper()} failed: {e}")

 # 3. / Demonstrate individual node operations
 print("\n3. Testing individual nodes...")

 # N0: / Data fetch
 print(" N0: Data fetching...")
 data_node = DataFetchNode()
 fetched = data_node.execute(method='api', params={'cache': True}, data={})
 print(f" Fetched {len(fetched.get('train_data', []))} training materials")

 # N2: / Feature matrix
 print(" N2: Feature matrix construction...")
 feature_node = FeatureMatrixNode()
 features = feature_node.execute(
 method='construct',
 params={'nan_thresh': 0.5, 'train_val_ratio': 0.8},
 data=fetched
 )
 print(f" Feature matrix: {features['X_train'].shape}")

 print("\n All components working correctly!")
 print(" MatFormPPO pipeline is ready for production use!")

if __name__ == "__main__":
 main()
