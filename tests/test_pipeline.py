#!/usr/bin/env python3
"""
Pipeline / Pipeline testing script

pipeline
Tests basic pipeline execution, including node execution, state management, and data flow.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging
from typing import Dict, Any

from pipeline import run_pipeline, run_pipeline_config
from nodes import (
 DataFetchNode,
 FeatureMatrixNode,
 ImputeNode,
 FeatureSelectionNode,
 ScalingNode,
 ModelTrainingNode
)
from env.pipeline_env import PipelineEnv

# / Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_pipeline():
 """
 pipeline / Test basic pipeline execution
 pipeline
 """
 print("\n" + "="*60)
 print("1: Pipeline / Test 1: Basic Pipeline Execution")
 print("="*60)

 try:
 result = run_pipeline(
 cache=True,
 impute_strategy='mean',
 nan_thresh=0.5,
 train_val_ratio=0.8,
 selection_strategy='none',
 scaling_strategy='standard',
 model_strategy='rf',
 model_params={'n_estimators': 10}
 )

 # / Verify results
 assert result is not None, "PipelineNone"
 assert 'model' in result, "model"
 assert 'y_val_pred' in result, "y_val_pred"

 # Check prediction accuracy
 mae = result.get('mae')
 r2 = result.get('r2')

 print(f"\n✓ Pipeline execution successful")
 print(f" MAE: {mae:.4f}" if mae else " MAE: N/A")
 print(f" R²: {r2:.4f}" if r2 else " R²: N/A")

 return True

 except Exception as e:
 print(f"\n✗ ERROR in pipeline execution: {e}")
 import traceback
 traceback.print_exc()
 return False


def test_node_execution():
 """
 / Test individual node execution

 """
 print("\n" + "="*60)
 print("2: / Test 2: Individual Node Execution")
 print("="*60)

 try:
 # DataFetchNode
 print("\n DataFetchNode...")
 n0 = DataFetchNode()
 data = n0.execute('api', {'cache': True}, {})
 assert 'X_train' in data, "DataFetchNodeX_train"
 assert 'y_train' in data, "DataFetchNodey_train"
 print(f" DataFetchNode: train={data['X_train'].shape[0]} samples")

 # FeatureMatrixNode
 print("\n FeatureMatrixNode...")
 n2 = FeatureMatrixNode()
 features = n2.execute('construct', {
 'nan_thresh': 0.5,
 'train_val_ratio': 0.8,
 'verbose': False
 }, data)
 assert 'X_train' in features, "FeatureMatrixNodeX_train"
 assert 'feature_names' in features, "FeatureMatrixNodefeature_names"
 print(f" FeatureMatrixNode: {features['X_train'].shape[1]} features")

 # ImputeNode
 print("\n🩹 ImputeNode...")
 n1 = ImputeNode()
 imputed = n1.execute('impute', {'strategy': 'mean'}, features)
 assert 'X_train' in imputed, "ImputeNodeX_train"
 assert not np.isnan(imputed['X_train']).any(), "ImputeNodeNaN"
 print(f" ImputeNode: NaN")

 # ScalingNode
 print("\n ScalingNode...")
 n4 = ScalingNode()
 scaled = n4.execute('scale', {'strategy': 'standard'}, imputed)
 assert 'X_train' in scaled, "ScalingNodeX_train"
 assert 'scaler' in scaled, "ScalingNodescaler"
 print(f" ScalingNode: mean{np.mean(scaled['X_train']):.2f}, std{np.std(scaled['X_train']):.2f}")

 # ModelTrainingNode
 print("\n ModelTrainingNode...")
 n5 = ModelTrainingNode()
 trained = n5.execute('train_rf', {'n_estimators': 10}, scaled)
 assert 'model' in trained, "ModelTrainingNodemodel"
 assert 'y_val_pred' in trained, "ModelTrainingNodey_val_pred"
 print(f" ModelTrainingNode: ")

 print(f"\n✓ All nodes executed successfully")
 return True

 except Exception as e:
 print(f"\n✗ ERROR in node execution: {e}")
 import traceback
 traceback.print_exc()
 return False


def test_pipeline_config():
 """
 10pipeline / Test 10-node flexible pipeline configuration
 run_pipeline_config
 """
 print("\n" + "="*60)
 print("3: Pipeline / Test 3: Flexible Pipeline Configuration")
 print("="*60)

 try:
 # : N0 N2 N1 N7 N8 N9
 config = {
 'sequence': ['N0', 'N2', 'N1', 'N7', 'N8', 'N9'],
 'N1_method': 'impute',
 'N1_params': {'strategy': 'mean'},
 'N7_method': 'scale',
 'N7_params': {'strategy': 'standard'},
 'N8_method': 'train_rf',
 'N8_params': {'n_estimators': 10},
 'cache': True,
 'nan_thresh': 0.5,
 'train_val_ratio': 0.8
 }

 print(f"\n : {' '.join(config['sequence'])}")
 result = run_pipeline_config(**config)

 # 
 assert result is not None, "PipelineNone"
 assert 'model' in result, "model"

 print(f"\n✓ Flexible pipeline configuration successful")
 print(f" Execution time: {result.get('total_time', 'N/A'):.2f}s" if 'total_time' in result else "")

 return True

 except Exception as e:
 print(f"\n✗ ERROR in pipeline configuration: {e}")
 import traceback
 traceback.print_exc()
 return False


def test_different_strategies():
 """
 / Test different strategy combinations
 pipeline
 """
 print("\n" + "="*60)
 print("4: / Test 4: Different Strategy Combinations")
 print("="*60)

 strategies = [
 {
 'name': 'Mean + Standard + RF',
 'impute': 'mean',
 'scaling': 'standard',
 'model': 'rf'
 },
 {
 'name': 'Median + Robust + GBR',
 'impute': 'median',
 'scaling': 'robust',
 'model': 'gbr'
 },
 {
 'name': 'KNN + MinMax + XGB',
 'impute': 'knn',
 'scaling': 'minmax',
 'model': 'xgb'
 }
 ]

 results = []

 for strategy in strategies:
 print(f"\n : {strategy['name']}")
 try:
 result = run_pipeline(
 cache=True,
 impute_strategy=strategy['impute'],
 impute_params={'n_neighbors': 5} if strategy['impute'] == 'knn' else None,
 scaling_strategy=strategy['scaling'],
 model_strategy=strategy['model'],
 model_params={'n_estimators': 10}
 )

 mae = result.get('mae', float('inf'))
 r2 = result.get('r2', 0.0)

 print(f" MAE: {mae:.4f}, R²: {r2:.4f}")
 results.append({'strategy': strategy['name'], 'mae': mae, 'r2': r2, 'success': True})

 except Exception as e:
 print(f" : {e}")
 results.append({'strategy': strategy['name'], 'success': False, 'error': str(e)})

 # Summary
 success_count = sum(1 for r in results if r.get('success', False))
 print(f"\n✓ Strategies passed: {success_count}/{len(strategies)}")

 return success_count == len(strategies)


def test_pipeline_with_ppo_env():
 """
 PipelinePPO / Test pipeline integration with PPO environment
 step
 """
 print("\n" + "="*60)
 print("5: PPO / Test 5: PPO Environment Integration")
 print("="*60)

 try:
 print("\n PPO...")
 env = PipelineEnv()

 print("\n ...")
 obs = env.reset()

 # 
 assert 'fingerprint' in obs, "fingerprint"
 assert 'node_visited' in obs, "node_visited"
 assert 'action_mask' in obs, "action_mask"

 print(f" : fingerprint={len(obs['fingerprint'])}, node_visited={len(obs['node_visited'])}")
 print(f" : {np.sum(obs['action_mask'])}")

 # 
 print("\n ...")
 valid_actions = np.where(obs['action_mask'])[0]
 if len(valid_actions) > 0:
 action_idx = np.random.choice(valid_actions)

 # action dict
 node_idx = action_idx % 10 # 10
 method_idx = 0
 params = [0.5, 0.5, 0.5]

 action = {
 'node': node_idx,
 'method': method_idx,
 'params': params
 }

 obs, reward, done, truncated, info = env.step(action)
 print(f" Step result: reward={reward:.3f}, done={done}")

 print(f"\n✓ PPO environment integration successful")
 return True

 except Exception as e:
 print(f"\n✗ ERROR in PPO integration: {e}")
 import traceback
 traceback.print_exc()
 return False


def test_error_handling():
 """
 / Test error handling
 pipeline
 """
 print("\n" + "="*60)
 print("6: / Test 6: Error Handling")
 print("="*60)

 test_cases = [
 {
 'name': 'impute',
 'params': {'impute_strategy': 'invalid_strategy'},
 'should_fail': True
 },
 {
 'name': 'model',
 'params': {'model_strategy': 'invalid_model'},
 'should_fail': True
 },
 {
 'name': 'nan_thresh',
 'params': {'nan_thresh': -0.5},
 'should_fail': False # 
 }
 ]

 passed = 0

 for case in test_cases:
 print(f"\n[TEST] Case: {case['name']}")
 try:
 params = {
 'cache': True,
 'model_params': {'n_estimators': 10}
 }
 params.update(case['params'])

 result = run_pipeline(**params)

 if case['should_fail']:
 print(f" ⚠ WARNING: Expected failure but succeeded")
 else:
 print(f" ✓ Passed")
 passed += 1

 except Exception as e:
 if case['should_fail']:
 print(f" ✓ Expected failure: {type(e).__name__}")
 passed += 1
 else:
 print(f" ✗ Unexpected failure: {e}")

 print(f"\n✓ Tests passed: {passed}/{len(test_cases)}")
 return passed >= len(test_cases) * 0.5 # 50% pass threshold


def run_all_tests():
 """
 Run all pipeline tests
 """
 print("\n" + "="*70)
 print("[TEST] Pipeline - Starting Pipeline Test Suite")
 print("="*70)

 tests = [
 ("Pipeline", test_basic_pipeline),
 ("", test_node_execution),
 ("Pipeline", test_pipeline_config),
 ("", test_different_strategies),
 ("PPO", test_pipeline_with_ppo_env),
 ("", test_error_handling)
 ]

 results = []

 for test_name, test_func in tests:
 try:
 success = test_func()
 results.append((test_name, success))
 except Exception as e:
 print(f"\nERROR {test_name} : {e}")
 import traceback
 traceback.print_exc()
 results.append((test_name, False))

 # 
 print("\n" + "="*70)
 print(" / Test Results Summary")
 print("="*70)

 for test_name, success in results:
 status = "SUCCESS " if success else "ERROR "
 print(f"{status} - {test_name}")

 passed = sum(1 for _, success in results if success)
 total = len(results)

 print("\n" + "="*70)
 print(f": {passed}/{total} ({passed/total*100:.1f}%)")
 print("="*70)

 return passed == total


if __name__ == "__main__":
 import warnings
 warnings.filterwarnings('ignore')

 success = run_all_tests()

 if success:
 print("\n ")
 sys.exit(0)
 else:
 print("\nWARNING ")
 sys.exit(1)
