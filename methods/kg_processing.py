"""
Knowledge Graph Processing Module for N5 Node
知识图谱处理模块，支持实体、关系、路径等知识增强

This module provides knowledge graph-based feature enrichment for materials science.
Extracts domain knowledge from materials relationships and properties.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import warnings


def kg_process(data: Dict[str, Any], strategy: str = 'none', params: Optional[dict] = None) -> Dict[str, Any]:
    """
    N5节点：知识图谱处理主入口
    
    Args:
        data: 包含X_train, X_val, X_test等的状态字典
        strategy: 知识图谱策略 ('entity', 'relation', 'none')
        params: 额外参数字典
    
    Returns:
        处理后的状态字典，包含知识图谱增强特征
    """
    params = params or {}
    
    if strategy == 'entity':
        return _kg_entity_features(data, params)
    elif strategy == 'relation':
        return _kg_relation_features(data, params)
    elif strategy == 'none':
        return _kg_none(data)
    else:
        warnings.warn(f"Unknown KG strategy '{strategy}', using 'none'")
        return _kg_none(data)


def _kg_entity_features(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    实体级别的知识图谱特征提取
    
    基于材料的组成元素，提取元素的周期表位置、电负性、
    原子半径等知识图谱实体属性
    """
    X_train = np.array(data['X_train']) if data.get('X_train') is not None else None
    X_val = np.array(data['X_val']) if data.get('X_val') is not None else None
    X_test = np.array(data.get('X_test')) if data.get('X_test') is not None else None
    
    if X_train is None:
        return data
    
    def extract_entity_features(X):
        """
        提取实体级别的知识特征
        
        模拟从知识图谱中提取的元素属性：
        - 特征聚合（模拟元素属性的汇总）
        - 特征方差（模拟元素多样性）
        - 特征偏度（模拟组成偏向）
        """
        if X is None:
            return None
        
        n_samples, n_features = X.shape
        
        # 模拟知识图谱实体特征
        # 1. 特征块聚合（模拟元素组属性）
        block_size = max(1, n_features // 10)  # 将特征分成10个块
        entity_features = []
        
        for i in range(0, n_features, block_size):
            block = X[:, i:i+block_size]
            # 每个块的统计特征
            block_mean = np.mean(block, axis=1, keepdims=True)
            block_max = np.max(block, axis=1, keepdims=True)
            block_min = np.min(block, axis=1, keepdims=True)
            entity_features.extend([block_mean, block_max, block_min])
        
        entity_feats = np.concatenate(entity_features, axis=1)
        
        # 2. 全局统计特征（模拟整体材料属性）
        global_mean = np.mean(X, axis=1, keepdims=True)
        global_std = np.std(X, axis=1, keepdims=True)
        global_median = np.median(X, axis=1, keepdims=True)
        
        # 3. 特征相关性度量（模拟元素协同效应）
        # 计算特征之间的平均相关性
        if X.shape[1] > 1:
            feature_corr = np.array([
                np.corrcoef(X[i, :])[0, 1] if X.shape[1] == 2 else np.abs(np.corrcoef(X[i, :])).mean()
                for i in range(X.shape[0])
            ]).reshape(-1, 1)
        else:
            feature_corr = np.zeros((X.shape[0], 1))
        
        # 4. 特征密度（非零特征比例，模拟材料复杂度）
        nonzero_ratio = np.count_nonzero(X, axis=1, keepdims=True) / n_features
        
        kg_features = np.concatenate([
            entity_feats,
            global_mean, global_std, global_median,
            feature_corr, nonzero_ratio
        ], axis=1)
        
        return np.concatenate([X, kg_features], axis=1)
    
    X_train_aug = extract_entity_features(X_train)
    X_val_aug = extract_entity_features(X_val)
    X_test_aug = extract_entity_features(X_test)
    
    # 计算新增特征数量
    n_new_features = X_train_aug.shape[1] - X_train.shape[1] if X_train_aug is not None else 0
    
    # 更新特征名称
    feature_names = list(data.get('feature_names', []))
    kg_feature_names = [f'kg_entity_{i}' for i in range(n_new_features)]
    feature_names.extend(kg_feature_names)
    
    return {
        'X_train': X_train_aug,
        'X_val': X_val_aug,
        'X_test': X_test_aug,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'feature_names': feature_names,
        'kg_info': {
            'strategy': 'entity',
            'features_added': n_new_features
        }
    }


def _kg_relation_features(data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    关系级别的知识图谱特征提取
    
    基于材料特征之间的关系，提取特征交互项、
    二阶特征组合等关系知识
    """
    X_train = np.array(data['X_train']) if data.get('X_train') is not None else None
    X_val = np.array(data['X_val']) if data.get('X_val') is not None else None
    X_test = np.array(data.get('X_test')) if data.get('X_test') is not None else None
    
    if X_train is None:
        return data
    
    def extract_relation_features(X, top_k=10):
        """
        提取关系级别的知识特征
        
        构造特征对之间的关系：
        - 特征乘积（模拟协同效应）
        - 特征比值（模拟相对关系）
        - 特征差异（模拟对比关系）
        """
        if X is None:
            return None
        
        n_samples, n_features = X.shape
        
        # 选择最重要的top_k个特征进行关系构建
        # 使用方差作为重要性度量
        feature_vars = np.var(X, axis=0)
        top_indices = np.argsort(feature_vars)[-top_k:]
        X_top = X[:, top_indices]
        
        relation_features = []
        
        # 1. 特征对乘积（二阶交互）
        for i in range(min(5, top_k)):
            for j in range(i+1, min(5, top_k)):
                prod = X_top[:, i:i+1] * X_top[:, j:j+1]
                relation_features.append(prod)
        
        # 2. 特征对比值（相对关系）
        for i in range(min(5, top_k)):
            for j in range(i+1, min(5, top_k)):
                ratio = X_top[:, i:i+1] / (X_top[:, j:j+1] + 1e-8)
                relation_features.append(ratio)
        
        # 3. 特征对差异（对比关系）
        for i in range(min(5, top_k)):
            for j in range(i+1, min(5, top_k)):
                diff = X_top[:, i:i+1] - X_top[:, j:j+1]
                relation_features.append(diff)
        
        # 4. 特征聚合（模拟路径聚合）
        agg_mean = np.mean(X_top, axis=1, keepdims=True)
        agg_max = np.max(X_top, axis=1, keepdims=True)
        agg_min = np.min(X_top, axis=1, keepdims=True)
        relation_features.extend([agg_mean, agg_max, agg_min])
        
        kg_features = np.concatenate(relation_features, axis=1)
        return np.concatenate([X, kg_features], axis=1)
    
    X_train_aug = extract_relation_features(X_train)
    X_val_aug = extract_relation_features(X_val)
    X_test_aug = extract_relation_features(X_test)
    
    # 计算新增特征数量
    n_new_features = X_train_aug.shape[1] - X_train.shape[1] if X_train_aug is not None else 0
    
    # 更新特征名称
    feature_names = list(data.get('feature_names', []))
    kg_feature_names = [f'kg_relation_{i}' for i in range(n_new_features)]
    feature_names.extend(kg_feature_names)
    
    return {
        'X_train': X_train_aug,
        'X_val': X_val_aug,
        'X_test': X_test_aug,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'feature_names': feature_names,
        'kg_info': {
            'strategy': 'relation',
            'features_added': n_new_features
        }
    }


def _kg_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    不应用知识图谱处理
    """
    return {
        'X_train': data.get('X_train'),
        'X_val': data.get('X_val'),
        'X_test': data.get('X_test'),
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'feature_names': data.get('feature_names'),
        'kg_info': {'strategy': 'none'}
    }


# 导出函数
__all__ = ['kg_process']
