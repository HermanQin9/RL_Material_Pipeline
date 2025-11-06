"""
PPO Utility Functions / PPO

This module contains utility functions for PPO training including GAE computation,
loss functions, and action validation for node selection operations.
PPOGAE
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
 / Compute Generalized Advantage Estimation (GAE)

 Computes GAE for advantage estimation in PPO training.
 PPO

 Args:
 rewards: / Reward sequence
 values: / Value function estimates
 dones: / Episode termination flags
 next_value: / Next state value
 gamma: / Discount factor
 gae_lambda: GAE / GAE lambda parameter

 Returns:
 advantages: / Advantage function
 returns: / Cumulative returns
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
 PPO / Compute PPO Policy Loss

 Computes the clipped surrogate objective for PPO training.
 PPO

 Args:
 new_log_probs: / New policy log probabilities
 old_log_probs: / Old policy log probabilities
 advantages: / Advantage values
 clip_param: / Clipping parameter

 Returns:
 torch.Tensor: PPO / PPO loss
 """
 ratio = torch.exp(new_log_probs - old_log_probs)

 # / Unclipped objective
 surr1 = ratio * advantages

 # / Clipped objective
 surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

 # / Take minimum as loss
 policy_loss = -torch.min(surr1, surr2).mean()

 return policy_loss

def value_loss(predicted_values: torch.Tensor,
 target_returns: torch.Tensor,
 clip_param: float = 0.2) -> torch.Tensor:
 """
 / Compute Value Function Loss

 Computes MSE loss for value function training.

 
 Args:
 predicted_values: / Predicted values
 target_returns: / Target returns
 clip_param: / Clipping parameter (currently unused)

 Returns:
 torch.Tensor: / Value function loss
 """
 value_loss_unclipped = F.mse_loss(predicted_values, target_returns)

 # / Optional: value function clipping
 # values_clipped = old_values + torch.clamp(
 # predicted_values - old_values, -clip_param, clip_param)
 # value_loss_clipped = F.mse_loss(values_clipped, target_returns)
 # value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

 return value_loss_unclipped

def entropy_loss(log_probs: torch.Tensor) -> torch.Tensor:
 """
 / Compute Entropy Loss

 Computes entropy loss to encourage exploration in PPO training.
 PPO

 Args:
 log_probs: / Log probabilities

 Returns:
 torch.Tensor: / Entropy loss
 """
 probs = torch.exp(log_probs)
 entropy = -(probs * log_probs).sum(dim=-1).mean()
 return entropy

def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
 """
 / Compute Explained Variance

 Computes the explained variance for model performance evaluation.

 
 Args:
 y_pred: / Predicted values
 y_true: / True values

 Returns:
 float: / Explained variance value
 """
 y_pred_np = y_pred.detach().cpu().numpy()
 y_true_np = y_true.detach().cpu().numpy()

 var_y = np.var(y_true_np)
 return float(1 - np.var(y_true_np - y_pred_np) / (var_y + 1e-8))

def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
 """
 / Normalize Advantages

 Normalizes advantage values for stable PPO training.
 PPO

 Args:
 advantages: / Advantage values

 Returns:
 torch.Tensor: / Normalized advantages
 """
 return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

def get_action_dim(action_space: dict) -> int:
 """
 / Get Action Space Dimension

 Calculate action space dimension for node_action and select_node operations.

 
 Args:
 action_space: / Action space dictionary

 Returns:
 int: / Total action space dimension
 """
 # + + / node_action: node selection + method + params
 node_dim = len(action_space.get('node', [])) # select_node dimension
 max_method_dim = max(len(methods) for methods in action_space.get('method', {}).values())
 param_dim = 1 # / single hyperparameter

 return node_dim + max_method_dim + param_dim

def validate_node_action(action: dict, action_space: dict) -> bool:
 """
 / Validate Node Action

 Check if select_node action is valid for the given action space.

 
 Args:
 action: / Action dictionary
 action_space: / Action space dictionary

 Returns:
 bool: / Whether the action is valid
 """
 if 'node' not in action or 'method' not in action:
 return False

 node_idx = action['node']
 method_idx = action['method']

 # / Check select_node index
 if node_idx < 0 or node_idx >= len(action_space.get('node', [])):
 return False

 # / Check method index for node_action
 node_name = f"N{node_idx + 1}" # Convert to node name
 if node_name not in action_space.get('method', {}):
 return False

 available_methods = action_space['method'][node_name]
 if method_idx < 0 or method_idx >= len(available_methods):
 return False

 return True
