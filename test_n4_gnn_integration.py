#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 GNN节点集成测试脚本 / N4 GNN Node Integration Test Script

测试以下功能:
1. GNN模块成功导入到data_methods.py
2. 环境检查 (PyTorch, PyTorch Geometric, pymatgen)
3. GNN函数可用性
4. 备用方案验证 (当GNN库不可用时)

Tests:
1. GNN module successfully imported into data_methods.py
2. Environment checks (PyTorch, PyTorch Geometric, pymatgen)
3. GNN function availability
4. Fallback mechanism verification (when GNN libraries unavailable)
"""

import logging
import numpy as np
from pathlib import Path

# 配置日志 / Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """测试N4 GNN模块导入 / Test N4 GNN module imports"""
    logger.info("=" * 60)
    logger.info("Test 1: GNN Module Imports")
    logger.info("=" * 60)
    
    try:
        from methods.data_methods import (
            gnn_process,
            structure_to_graph,
            extract_gnn_features,
            SimpleGCN,
            SimpleGAT,
            SimpleGraphSAGE
        )
        logger.info("[OK] All GNN functions imported successfully")
        return True
    except ImportError as e:
        logger.error(f"[FAIL] Import failed: {e}")
        return False

def test_environment():
    """测试环境依赖 / Test environment dependencies"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Environment Dependencies")
    logger.info("=" * 60)
    
    # PyTorch
    try:
        import torch
        logger.info(f"[OK] PyTorch: {torch.__version__}")
        logger.info(f"     CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        logger.warning("[WARN] PyTorch not available - will use statistical fallback")
    
    # PyTorch Geometric
    try:
        import torch_geometric
        logger.info(f"[OK] PyTorch Geometric available")
    except ImportError:
        logger.warning("[WARN] PyTorch Geometric not available - will use statistical fallback")
    
    # pymatgen
    try:
        import pymatgen
        logger.info("[OK] pymatgen available")
    except ImportError:
        logger.warning("[WARN] pymatgen not available - will use fallback graph construction")
    
    return True

def test_gnn_functions():
    """测试GNN函数可用性 / Test GNN function availability"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: GNN Function Availability")
    logger.info("=" * 60)
    
    from methods.data_methods import (
        gnn_process,
        structure_to_graph,
        extract_gnn_features,
        _statistical_fallback_features,
        _fallback_graph_construction
    )
    
    logger.info("[OK] gnn_process function available")
    logger.info("[OK] structure_to_graph function available")
    logger.info("[OK] extract_gnn_features function available")
    logger.info("[OK] _statistical_fallback_features function available")
    logger.info("[OK] _fallback_graph_construction function available")
    
    return True

def test_gnn_processing():
    """测试GNN处理流程 / Test GNN processing pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: GNN Processing Pipeline (with dummy data)")
    logger.info("=" * 60)
    
    from methods.data_methods import gnn_process
    
    # 创建虚拟数据 / Create dummy data
    n_train, n_features = 10, 5
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    X_val = np.random.randn(5, n_features).astype(np.float32)
    X_test = np.random.randn(3, n_features).astype(np.float32)
    
    y_train = np.random.randn(n_train)
    y_val = np.random.randn(5)
    y_test = np.random.randn(3)
    
    feature_names = [f'feat_{i}' for i in range(n_features)]
    
    # 创建状态字典 / Create state dictionary
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'structures_train': [],  # 空结构列表将触发备用方案 / Empty structures will trigger fallback
        'structures_val': [],
        'structures_test': []
    }
    
    try:
        # 调用GNN处理函数 / Call GNN processing function
        result = gnn_process(data, strategy='gcn', param=0.5)
        
        # 检查输出 / Check output
        if 'X_train' in result and result['X_train'] is not None:
            new_shape = result['X_train'].shape
            logger.info(f"[OK] GNN processing complete")
            logger.info(f"     Input shape: {X_train.shape}")
            logger.info(f"     Output shape: {new_shape}")
            logger.info(f"     New features added: {new_shape[1] - n_features}")
            return True
        else:
            logger.error("[FAIL] GNN processing did not return X_train")
            return False
    
    except Exception as e:
        logger.error(f"[FAIL] GNN processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_mapping():
    """测试参数映射 / Test parameter mapping"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Parameter Mapping (param -> output_dim)")
    logger.info("=" * 60)
    
    from methods.data_methods import gnn_process
    
    # 创建虚拟数据 / Create dummy data
    X_train = np.random.randn(5, 3).astype(np.float32)
    X_val = np.random.randn(3, 3).astype(np.float32)
    X_test = np.random.randn(2, 3).astype(np.float32)
    
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': np.random.randn(5),
        'y_val': np.random.randn(3),
        'y_test': np.random.randn(2),
        'feature_names': ['f1', 'f2', 'f3'],
        'structures_train': [],
        'structures_val': [],
        'structures_test': []
    }
    
    test_cases = [
        (0.0, 8),   # 小维度 / Small dimension
        (0.5, 16),  # 中等维度 / Medium dimension
        (1.0, 32),  # 大维度 / Large dimension
    ]
    
    all_passed = True
    for param, expected_dim in test_cases:
        try:
            result = gnn_process(data, strategy='gcn', param=param)
            added_features = result['X_train'].shape[1] - 3
            
            if added_features == expected_dim:
                logger.info(f"[OK] param={param} -> {added_features} features (expected {expected_dim})")
            else:
                logger.error(f"[FAIL] param={param} -> {added_features} features (expected {expected_dim})")
                all_passed = False
        except Exception as e:
            logger.error(f"[FAIL] param={param} processing failed: {e}")
            all_passed = False
    
    return all_passed

def test_gnn_strategies():
    """测试不同的GNN策略 / Test different GNN strategies"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: GNN Strategies")
    logger.info("=" * 60)
    
    from methods.data_methods import gnn_process
    
    # 创建虚拟数据 / Create dummy data
    X_train = np.random.randn(5, 3).astype(np.float32)
    X_val = np.random.randn(3, 3).astype(np.float32)
    
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': None,
        'y_train': np.random.randn(5),
        'y_val': np.random.randn(3),
        'y_test': None,
        'feature_names': ['f1', 'f2', 'f3'],
        'structures_train': [],
        'structures_val': [],
    }
    
    strategies = ['gcn', 'gat', 'sage']
    all_passed = True
    
    for strategy in strategies:
        try:
            result = gnn_process(data, strategy=strategy, param=0.5)
            if 'gnn_info' in result:
                logger.info(f"[OK] Strategy '{strategy}' processed successfully")
                logger.info(f"     GNN method: {result['gnn_info']['method']}")
                logger.info(f"     Output dim: {result['gnn_info']['output_dim']}")
            else:
                logger.error(f"[FAIL] Strategy '{strategy}' missing gnn_info")
                all_passed = False
        except Exception as e:
            logger.error(f"[FAIL] Strategy '{strategy}' failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """主测试函数 / Main test function"""
    logger.info("\n" + "=" * 60)
    logger.info("N4 GNN Integration Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Environment", test_environment),
        ("Function Availability", test_gnn_functions),
        ("GNN Processing", test_gnn_processing),
        ("Parameter Mapping", test_parameter_mapping),
        ("GNN Strategies", test_gnn_strategies),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结 / Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        logger.info(f"{status} {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    return passed == total

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
