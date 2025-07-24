"""
PPO Utility Functions / PPO工具函数模块

This module contains utility functions for PPO training including GAE computation,
loss functions, and action validation for node selection operations.
本模块包含PPO训练的工具函数，包括GAE计算、损失函数和节点选择操作验证。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def compute_gae(rewards: torch.Tensor, 
                values: torch.Tensor, 
                dones: torch.Tensor, 
                next_value: float, 
                gamma: float = 0.99, 
                gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算广义优势估计 / Compute Generalized Advantage Estimation (GAE)
    
    Computes GAE for advantage estimation in PPO training.
    为PPO训练计算广义优势估计。
    
    Args:
        rewards: 奖励序列 / Reward sequence
        values: 价值函数估计 / Value function estimates
        dones: 回合结束标志 / Episode termination flags
        next_value: 下一状态的价值 / Next state value
        gamma: 折扣因子 / Discount factor
        gae_lambda: GAE参数 / GAE lambda parameter
    
    Returns:
        advantages: 优势函数 / Advantage function
        returns: 累积回报 / Cumulative returns
    """
    batch_size = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_non_terminal = 1.0 - dones[t]
            next_return = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_return = values[t + 1]
        
        delta = rewards[t] + gamma * next_return * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    
    return advantages, returns

def ppo_loss(new_log_probs: torch.Tensor,
             old_log_probs: torch.Tensor,
             advantages: torch.Tensor,
             clip_param: float = 0.2) -> torch.Tensor:
    """
    计算PPO策略损失 / Compute PPO Policy Loss
    
    Computes the clipped surrogate objective for PPO training.
    计算PPO训练的截断代理目标函数。
    
    Args:
        new_log_probs: 新策略的对数概率 / New policy log probabilities
        old_log_probs: 旧策略的对数概率 / Old policy log probabilities
        advantages: 优势函数 / Advantage values
        clip_param: 裁剪参数 / Clipping parameter
    
    Returns:
        torch.Tensor: PPO损失 / PPO loss
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 未裁剪的目标函数 / Unclipped objective
    surr1 = ratio * advantages
    
    # 裁剪的目标函数 / Clipped objective
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    
    # 取最小值作为损失 / Take minimum as loss
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss

def value_loss(predicted_values: torch.Tensor,
               target_returns: torch.Tensor,
               clip_param: float = 0.2) -> torch.Tensor:
    """
    计算价值函数损失 / Compute Value Function Loss
    
    Computes MSE loss for value function training.
    计算价值函数训练的均方误差损失。
    
    Args:
        predicted_values: 预测的价值 / Predicted values
        target_returns: 目标回报 / Target returns
        clip_param: 裁剪参数 / Clipping parameter (currently unused)
    
    Returns:
        torch.Tensor: 价值函数损失 / Value function loss
    """
    value_loss_unclipped = F.mse_loss(predicted_values, target_returns)
    
    # 可选：价值函数裁剪 / Optional: value function clipping
    # values_clipped = old_values + torch.clamp(
    #     predicted_values - old_values, -clip_param, clip_param)
    # value_loss_clipped = F.mse_loss(values_clipped, target_returns)
    # value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    
    return value_loss_unclipped

def entropy_loss(log_probs: torch.Tensor) -> torch.Tensor:
    """
    计算熵损失 / Compute Entropy Loss
    
    Computes entropy loss to encourage exploration in PPO training.
    计算熵损失以在PPO训练中鼓励探索。
    
    Args:
        log_probs: 对数概率 / Log probabilities
    
    Returns:
        torch.Tensor: 熵损失 / Entropy loss
    """
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    计算解释方差 / Compute Explained Variance
    
    Computes the explained variance for model performance evaluation.
    计算模型性能评估的解释方差。
    
    Args:
        y_pred: 预测值 / Predicted values
        y_true: 真实值 / True values
    
    Returns:
        float: 解释方差值 / Explained variance value
    """
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    var_y = np.var(y_true_np)
    return float(1 - np.var(y_true_np - y_pred_np) / (var_y + 1e-8))

def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    归一化优势函数 / Normalize Advantages
    
    Normalizes advantage values for stable PPO training.
    为稳定的PPO训练归一化优势值。
    
    Args:
        advantages: 优势函数值 / Advantage values
    
    Returns:
        torch.Tensor: 归一化后的优势函数 / Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

def get_action_dim(action_space: dict) -> int:
    """
    获取动作空间维度 / Get Action Space Dimension
    
    Calculate action space dimension for node_action and select_node operations.
    计算节点动作和节点选择操作的动作空间维度。
    
    Args:
        action_space: 动作空间字典 / Action space dictionary
    
    Returns:
        int: 动作空间总维度 / Total action space dimension
    """
    # 节点选择 + 方法选择 + 参数 / node_action: node selection + method + params
    node_dim = len(action_space.get('node', []))  # select_node dimension
    max_method_dim = max(len(methods) for methods in action_space.get('method', {}).values())
    param_dim = 1  # 假设单一超参数 / single hyperparameter
    
    return node_dim + max_method_dim + param_dim

def validate_node_action(action: dict, action_space: dict) -> bool:
    """
    验证节点动作有效性 / Validate Node Action
    
    Check if select_node action is valid for the given action space.
    检查选择节点的动作对给定动作空间是否有效。
    
    Args:
        action: 动作字典 / Action dictionary
        action_space: 动作空间字典 / Action space dictionary
    
    Returns:
        bool: 动作是否有效 / Whether the action is valid
    """
    if 'node' not in action or 'method' not in action:
        return False
    
    node_idx = action['node']
    method_idx = action['method']
    
    # 检查节点索引 / Check select_node index
    if node_idx < 0 or node_idx >= len(action_space.get('node', [])):
        return False
    
    # 检查方法索引 / Check method index for node_action
    node_name = f"N{node_idx + 1}"  # Convert to node name
    if node_name not in action_space.get('method', {}):
        return False
    
    available_methods = action_space['method'][node_name]
    if method_idx < 0 or method_idx >= len(available_methods):
        return False
    
    return True
