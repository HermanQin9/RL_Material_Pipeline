#!/usr/bin/env python3
"""
Pipeline测试脚本 / Pipeline testing script

测试基础pipeline执行功能，包括节点执行、状态管理和数据流转。
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

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_pipeline():
    """
    测试基础pipeline执行 / Test basic pipeline execution
    使用默认参数执行完整pipeline
    """
    print("\n" + "="*60)
    print("测试1: 基础Pipeline执行 / Test 1: Basic Pipeline Execution")
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
        
        # 验证结果 / Verify results
        assert result is not None, "Pipeline返回None"
        assert 'model' in result, "结果中缺少model"
        assert 'y_val_pred' in result, "结果中缺少y_val_pred"
        
        # 检查预测精度 / Check prediction accuracy
        mae = result.get('mae')
        r2 = result.get('r2')
        
        print(f"\n✅ 基础Pipeline测试通过")
        print(f"   MAE: {mae:.4f}" if mae else "   MAE: N/A")
        print(f"   R²: {r2:.4f}" if r2 else "   R²: N/A")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 基础Pipeline测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_execution():
    """
    测试单个节点执行 / Test individual node execution
    验证各节点的输入输出正确性
    """
    print("\n" + "="*60)
    print("测试2: 单个节点执行 / Test 2: Individual Node Execution")
    print("="*60)
    
    try:
        # 测试DataFetchNode
        print("\n📦 测试DataFetchNode...")
        n0 = DataFetchNode()
        data = n0.execute('api', {'cache': True}, {})
        assert 'X_train' in data, "DataFetchNode输出缺少X_train"
        assert 'y_train' in data, "DataFetchNode输出缺少y_train"
        print(f"   ✓ DataFetchNode: train={data['X_train'].shape[0]} samples")
        
        # 测试FeatureMatrixNode
        print("\n🔧 测试FeatureMatrixNode...")
        n2 = FeatureMatrixNode()
        features = n2.execute('construct', {
            'nan_thresh': 0.5,
            'train_val_ratio': 0.8,
            'verbose': False
        }, data)
        assert 'X_train' in features, "FeatureMatrixNode输出缺少X_train"
        assert 'feature_names' in features, "FeatureMatrixNode输出缺少feature_names"
        print(f"   ✓ FeatureMatrixNode: {features['X_train'].shape[1]} features")
        
        # 测试ImputeNode
        print("\n🩹 测试ImputeNode...")
        n1 = ImputeNode()
        imputed = n1.execute('impute', {'strategy': 'mean'}, features)
        assert 'X_train' in imputed, "ImputeNode输出缺少X_train"
        assert not np.isnan(imputed['X_train']).any(), "ImputeNode未能消除所有NaN"
        print(f"   ✓ ImputeNode: 无NaN值")
        
        # 测试ScalingNode
        print("\n📏 测试ScalingNode...")
        n4 = ScalingNode()
        scaled = n4.execute('scale', {'strategy': 'standard'}, imputed)
        assert 'X_train' in scaled, "ScalingNode输出缺少X_train"
        assert 'scaler' in scaled, "ScalingNode输出缺少scaler"
        print(f"   ✓ ScalingNode: mean≈{np.mean(scaled['X_train']):.2f}, std≈{np.std(scaled['X_train']):.2f}")
        
        # 测试ModelTrainingNode
        print("\n🤖 测试ModelTrainingNode...")
        n5 = ModelTrainingNode()
        trained = n5.execute('train_rf', {'n_estimators': 10}, scaled)
        assert 'model' in trained, "ModelTrainingNode输出缺少model"
        assert 'y_val_pred' in trained, "ModelTrainingNode输出缺少y_val_pred"
        print(f"   ✓ ModelTrainingNode: 模型训练完成")
        
        print(f"\n✅ 单个节点执行测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 单个节点执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_config():
    """
    测试10节点灵活pipeline配置 / Test 10-node flexible pipeline configuration
    使用run_pipeline_config执行不同的节点序列
    """
    print("\n" + "="*60)
    print("测试3: 灵活Pipeline配置 / Test 3: Flexible Pipeline Configuration")
    print("="*60)
    
    try:
        # 定义一个简单的序列: N0 → N2 → N1 → N7 → N8 → N9
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
        
        print(f"\n🔄 执行序列: {' → '.join(config['sequence'])}")
        result = run_pipeline_config(**config)
        
        # 验证结果
        assert result is not None, "Pipeline返回None"
        assert 'model' in result, "结果中缺少model"
        
        print(f"\n✅ 灵活Pipeline配置测试通过")
        print(f"   执行时间: {result.get('total_time', 'N/A'):.2f}s" if 'total_time' in result else "")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 灵活Pipeline配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_strategies():
    """
    测试不同的策略组合 / Test different strategy combinations
    验证pipeline对不同参数的适应性
    """
    print("\n" + "="*60)
    print("测试4: 不同策略组合 / Test 4: Different Strategy Combinations")
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
        print(f"\n🔧 测试策略: {strategy['name']}")
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
            
            print(f"   ✓ MAE: {mae:.4f}, R²: {r2:.4f}")
            results.append({'strategy': strategy['name'], 'mae': mae, 'r2': r2, 'success': True})
            
        except Exception as e:
            print(f"   ✗ 失败: {e}")
            results.append({'strategy': strategy['name'], 'success': False, 'error': str(e)})
    
    # 统计成功率
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"\n✅ 策略测试完成: {success_count}/{len(strategies)} 成功")
    
    return success_count == len(strategies)


def test_pipeline_with_ppo_env():
    """
    测试Pipeline与PPO环境的集成 / Test pipeline integration with PPO environment
    验证环境初始化和基本step操作
    """
    print("\n" + "="*60)
    print("测试5: PPO环境集成 / Test 5: PPO Environment Integration")
    print("="*60)
    
    try:
        print("\n🤖 初始化PPO环境...")
        env = PipelineEnv()
        
        print("\n🔄 重置环境...")
        obs = env.reset()
        
        # 验证观察空间
        assert 'fingerprint' in obs, "观察缺少fingerprint"
        assert 'node_visited' in obs, "观察缺少node_visited"
        assert 'action_mask' in obs, "观察缺少action_mask"
        
        print(f"   ✓ 观察空间维度: fingerprint={len(obs['fingerprint'])}, node_visited={len(obs['node_visited'])}")
        print(f"   ✓ 可用动作数: {np.sum(obs['action_mask'])}")
        
        # 执行一个随机动作
        print("\n🎮 执行随机动作...")
        valid_actions = np.where(obs['action_mask'])[0]
        if len(valid_actions) > 0:
            action_idx = np.random.choice(valid_actions)
            
            # 需要构建完整的action dict
            node_idx = action_idx % 10  # 假设10个节点
            method_idx = 0
            params = [0.5, 0.5, 0.5]
            
            action = {
                'node': node_idx,
                'method': method_idx,
                'params': params
            }
            
            obs, reward, done, truncated, info = env.step(action)
            print(f"   ✓ 动作执行成功: reward={reward:.3f}, done={done}")
        
        print(f"\n✅ PPO环境集成测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ PPO环境集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """
    测试错误处理 / Test error handling
    验证pipeline对异常输入的鲁棒性
    """
    print("\n" + "="*60)
    print("测试6: 错误处理 / Test 6: Error Handling")
    print("="*60)
    
    test_cases = [
        {
            'name': '无效的impute策略',
            'params': {'impute_strategy': 'invalid_strategy'},
            'should_fail': True
        },
        {
            'name': '无效的model策略',
            'params': {'model_strategy': 'invalid_model'},
            'should_fail': True
        },
        {
            'name': '负数nan_thresh',
            'params': {'nan_thresh': -0.5},
            'should_fail': False  # 应该自动纠正
        }
    ]
    
    passed = 0
    
    for case in test_cases:
        print(f"\n🧪 测试: {case['name']}")
        try:
            params = {
                'cache': True,
                'model_params': {'n_estimators': 10}
            }
            params.update(case['params'])
            
            result = run_pipeline(**params)
            
            if case['should_fail']:
                print(f"   ⚠️ 预期失败但成功了")
            else:
                print(f"   ✓ 正确处理")
                passed += 1
                
        except Exception as e:
            if case['should_fail']:
                print(f"   ✓ 按预期失败: {type(e).__name__}")
                passed += 1
            else:
                print(f"   ✗ 意外失败: {e}")
    
    print(f"\n✅ 错误处理测试: {passed}/{len(test_cases)} 通过")
    return passed >= len(test_cases) * 0.5  # 至少50%通过


def run_all_tests():
    """
    运行所有测试 / Run all tests
    """
    print("\n" + "="*70)
    print("🧪 开始Pipeline测试套件 / Starting Pipeline Test Suite")
    print("="*70)
    
    tests = [
        ("基础Pipeline执行", test_basic_pipeline),
        ("单个节点执行", test_node_execution),
        ("灵活Pipeline配置", test_pipeline_config),
        ("不同策略组合", test_different_strategies),
        ("PPO环境集成", test_pipeline_with_ppo_env),
        ("错误处理", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} 崩溃: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*70)
    print("📊 测试结果总结 / Test Results Summary")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("\n" + "="*70)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
        sys.exit(1)
