#!/usr/bin/env python3
"""
流水线调试工具 / Pipeline Debug Tool

This script provides detailed debugging and logging for pipeline data flow.
此脚本为流水线数据流提供详细的调试和日志记录。
"""

import sys
from pathlib import Path

# 添加父目录到路径以便导入 / Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

print("=== 调试流水线数据流 / Debugging Pipeline Data Flow ===")

try:
    from pipeline import run_pipeline
    from nodes import DataFetchNode, ImputeNode, FeatureMatrixNode, FeatureSelectionNode, ScalingNode, ModelTrainingNode
    
    # 逐步测试每个节点 / Test each step individually
    print("\n1. 测试 N0 - 数据获取 / Testing N0 - Data Fetch")
    data_node = DataFetchNode()
    fetched = data_node.execute(method='api', params={'cache': True}, data={})
    print(f"   获取的键 / Fetched keys: {list(fetched.keys())}")
    print(f"   训练标签 / y_train: {fetched.get('y_train')}")
    print(f"   验证标签 / y_val: {fetched.get('y_val')}")
    print(f"   测试标签 / y_test: {fetched.get('y_test')}")
    
    print("\n2. 测试 N2 - 特征矩阵 / Testing N2 - Feature Matrix")
    feature_node = FeatureMatrixNode()
    features = feature_node.execute(
        method='construct',
        params={'nan_thresh': 0.5, 'train_val_ratio': 0.8, 'verbose': False},
        data=fetched
    )
    print(f"   特征键 / Features keys: {list(features.keys())}")
    print(f"   训练标签 / y_train: {type(features.get('y_train'))} {features.get('y_train')[:5] if features.get('y_train') is not None else None}")
    print(f"   验证标签 / y_val: {type(features.get('y_val'))} {features.get('y_val')[:5] if features.get('y_val') is not None else None}")
    print(f"   测试标签 / y_test: {type(features.get('y_test'))} {features.get('y_test')[:5] if features.get('y_test') is not None else None}")
    
    print("\n3. 测试 N1 - 缺失值填充 / Testing N1 - Imputation")
    impute_node = ImputeNode()
    imputed = impute_node.execute(
        method='impute', 
        params={'strategy': 'mean', 'params': {}}, 
        data=features
    )
    print(f"   填充后的键 / Imputed keys: {list(imputed.keys())}")
    print(f"   训练标签 / y_train: {type(imputed.get('y_train'))} {imputed.get('y_train')[:5] if imputed.get('y_train') is not None else None}")
    print(f"   验证标签 / y_val: {type(imputed.get('y_val'))} {imputed.get('y_val')[:5] if imputed.get('y_val') is not None else None}")
    print(f"   训练数据形状 / X_train shape: {imputed.get('X_train').shape if imputed.get('X_train') is not None else None}")
    
    print("\n4. 测试 N3 - 特征选择 / Testing N3 - Feature Selection")
    select_node = FeatureSelectionNode()
    selected = select_node.execute(
        method='select',
        params={'strategy': 'none', 'params': {}},
        data=imputed
    )
    print(f"   选择后的键 / Selected keys: {list(selected.keys())}")
    print(f"   训练标签 / y_train: {type(selected.get('y_train'))} {selected.get('y_train')[:5] if selected.get('y_train') is not None else None}")
    
    print("\n5. 测试 N4 - 数据缩放 / Testing N4 - Scaling")
    scaling_node = ScalingNode()
    scaled = scaling_node.execute(
        method='scale',
        params={'strategy': 'standard', 'params': {}},
        data=selected
    )
    print(f"   缩放后的键 / Scaled keys: {list(scaled.keys())}")
    print(f"   训练标签 / y_train: {type(scaled.get('y_train'))} {scaled.get('y_train')[:5] if scaled.get('y_train') is not None else None}")
    
except Exception as e:
    import traceback
    print(f"✗ 调试失败 / Debug failed: {e}")
    print("完整错误追踪 / Full traceback:")
    traceback.print_exc()

print("\n=== 调试完成 / Debug completed ===")
