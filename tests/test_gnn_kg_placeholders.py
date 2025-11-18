#!/usr/bin/env python3
"""
GNN和知识图谱占位符功能测试 / GNN and Knowledge Graph Placeholder Function Tests

测试GNN处理和知识图谱处理的占位符实现
Tests placeholder implementations of GNN processing and knowledge graph processing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from methods.data.preprocessing import gnn_process, kg_process


def _fake_data(n=10, d=4):
    """
    生成模拟测试数据 / Generate fake test data
    
    Args:
        n: 样本数量 / Number of samples
        d: 特征维度 / Feature dimensions
    
    Returns:
        包含训练/验证/测试数据的字典 / Dict containing train/val/test data
    """
    X = np.random.randn(n, d)
    return {
        'X_train': X,
        'X_val': X.copy(),
        'X_test': X.copy(),
        'y_train': np.random.randn(n),
        'y_val': np.random.randn(n),
        'y_test': np.random.randn(n),
        'feature_names': [f"f{i}" for i in range(d)]
    }


def test_gnn_process_appends_stats():
    """
    测试GNN处理是否添加统计特征 / Test if GNN process appends statistical features
    
    验证GNN处理会添加4个额外的统计特征
    Verifies that GNN processing adds 4 additional statistical features
    """
    print("🧪 测试GNN处理功能 / Testing GNN process...")
    data = _fake_data()
    out = gnn_process(data)
    assert out['X_train'].shape[1] == data['X_train'].shape[1] + 4, \
        "GNN应添加4个特征 / GNN should add 4 features"
    assert len(out['feature_names']) == len(data['feature_names']) + 4, \
        "特征名称列表应增加4个 / Feature names list should increase by 4"
    print("✅ GNN处理测试通过 / GNN process test passed")


def test_kg_process_noop():
    """
    测试知识图谱处理（当前为占位符） / Test knowledge graph process (currently placeholder)
    
    验证知识图谱处理不改变数据形状
    Verifies that knowledge graph processing does not change data shape
    """
    print("🧪 测试知识图谱处理功能 / Testing KG process...")
    data = _fake_data()
    out = kg_process(data)
    assert out['X_train'].shape == data['X_train'].shape, \
        "知识图谱处理当前应保持数据不变 / KG process should currently keep data unchanged"
    print("✅ 知识图谱处理测试通过 / KG process test passed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 开始GNN和知识图谱测试 / Starting GNN and KG Tests")
    print("="*70 + "\n")
    
    test_gnn_process_appends_stats()
    test_kg_process_noop()
    
    print("\n" + "="*70)
    print("🎉 所有测试通过！ / All tests passed!")
    print("="*70)