"""
Environment Utility Functions / 环境工具函数模块

This module contains utility functions for the RL environment including
node_action validation, select_node operations, and feature_selection support.
本模块包含强化学习环境的工具函数，包括节点动作验证、节点选择操作和特征选择支持。
"""

import numpy as np
from typing import Dict, Any, List, Tuple

def validate_action(action: Dict[str, Any], methods_for_node: Dict[str, List[str]]) -> bool:
    """
    验证动作的有效性 / Validate Action Validity
    
    Check if node_action is valid for select_node and feature_selection operations.
    检查节点动作对于节点选择和特征选择操作是否有效。
    
    Args:
        action: 动作字典 / Action dictionary
        methods_for_node: 节点方法映射 / Node methods mapping
    
    Returns:
        bool: 动作是否有效 / Whether the action is valid
    """
    if not isinstance(action, dict):
        return False
    
    node_idx = action.get('node')
    method_idx = action.get('method')
    
    if node_idx is None or method_idx is None:
        return False
    
    if not isinstance(node_idx, int) or not isinstance(method_idx, int):
        return False
    
    # Check node index range for select_node
    if node_idx < 0 or node_idx >= 5:  # We have 5 nodes
        return False
    
    # Get node name for select_node operation
    node_names = ['N2', 'N1', 'N3', 'N4', 'N5']
    node_name = node_names[node_idx]
    
    if node_name not in methods_for_node:
        return False
    
    available_methods = methods_for_node[node_name]
    if method_idx < 0 or method_idx >= len(available_methods):
        return False
    
    return True

def select_node_by_name(node_name: str, pipeline_nodes: List[str]) -> int:
    """
    通过节点名称选择节点 / Select Node by Name
    
    Helper function for select_node operations to get node index.
    节点选择操作的辅助函数，用于获取节点索引。
    
    Args:
        node_name: 节点名称 / Node name
        pipeline_nodes: 流水线节点列表 / Pipeline nodes list
    
    Returns:
        int: 节点索引，-1表示未找到 / Node index, -1 if not found
    """
    try:
        return pipeline_nodes.index(node_name)
    except ValueError:
        return -1

def get_feature_selection_methods() -> List[str]:
    """
    获取特征选择方法列表 / Get Feature Selection Methods
    
    Returns available feature_selection algorithms for N3 node.
    返回N3节点可用的特征选择算法。
    
    Returns:
        List[str]: 特征选择方法列表 / Feature selection methods list
    """
    return ['none', 'variance', 'univariate', 'pca']

def is_feature_selection_node(node_name: str) -> bool:
    """
    检查是否为特征选择节点 / Check if Feature Selection Node
    
    Determines if the given node performs feature_selection operations.
    判断给定节点是否执行特征选择操作。
    
    Args:
        node_name: 节点名称 / Node name
    
    Returns:
        bool: 是否为特征选择节点 / Whether it's a feature selection node
    """
    return node_name == 'N3'

def create_node_action(node_idx: int, method_idx: int, param: float = 0.5) -> Dict[str, Any]:
    """
    创建标准的节点动作 / Create Standard Node Action
    
    Helper to create select_node action dictionary.
    创建节点选择动作字典的辅助函数。
    
    Args:
        node_idx: 节点索引 / Node index
        method_idx: 方法索引 / Method index
        param: 参数值 / Parameter value
    
    Returns:
        Dict[str, Any]: 节点动作字典 / Node action dictionary
    """
    return {
        'node': node_idx,
        'method': method_idx,
        'params': [param]
    }

def compute_action_mask(node_visited: List[bool], current_step: int, num_nodes: int) -> Dict[str, np.ndarray]:
    """
    计算动作掩码，防止选择无效动作
    Compute action mask to prevent invalid actions
    """
    mask = {
        'node': np.ones(num_nodes, dtype=bool),
        'method': {},
        'params': True
    }
    
    # 根据当前步骤和访问状态计算节点掩码
    if current_step == 0:
        # 第一步只能选择N2
        mask['node'][:] = False
        mask['node'][0] = True  # N2的索引
    elif current_step == num_nodes - 1:
        # 最后一步只能选择N5
        mask['node'][:] = False
        mask['node'][-1] = True  # N5的索引
    else:
        # 中间步骤不能选择已访问的节点
        for i, visited in enumerate(node_visited):
            if visited:
                mask['node'][i] = False
    
    return mask

def compute_reward(mae: float, r2: float, num_features: int, complexity_penalty: float = 0.01) -> float:
    """
    计算奖励函数
    Compute reward based on model performance and complexity
    
    Args:
        mae: Mean Absolute Error
        r2: R-squared score
        num_features: Number of features
        complexity_penalty: Penalty for model complexity
    
    Returns:
        reward: Computed reward value
    """
    # 基础性能奖励 (R2越高越好，MAE越低越好)
    performance_reward = r2 - mae / 10.0
    
    # 复杂度惩罚 (特征数越多惩罚越大)
    complexity_cost = complexity_penalty * num_features
    
    # 总奖励
    total_reward = performance_reward - complexity_cost
    
    return float(total_reward)

def get_observation_vector(
    fingerprint: np.ndarray,
    node_visited: List[bool], 
    method_calls: Dict[str, int],
    current_step: int
) -> np.ndarray:
    """
    构建观察向量
    Build observation vector for RL agent
    """
    obs_parts = []
    
    # 1. 性能指纹 [MAE, R2, num_features]
    obs_parts.append(fingerprint)
    
    # 2. 节点访问状态
    obs_parts.append(np.array(node_visited, dtype=np.float32))
    
    # 3. 方法使用计数 (归一化)
    method_counts = np.array(list(method_calls.values()), dtype=np.float32)
    if method_counts.sum() > 0:
        method_counts = method_counts / method_counts.sum()
    obs_parts.append(method_counts)
    
    # 4. 当前步骤 (归一化)
    step_normalized = np.array([current_step / 5.0], dtype=np.float32)
    obs_parts.append(step_normalized)
    
    # 拼接所有部分
    observation = np.concatenate(obs_parts)
    return observation
