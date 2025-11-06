"""
Rollout Buffer for PPO
PPO
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple

class RolloutBuffer:
 """
 PPO
 Rollout buffer for storing trajectory data during PPO training
 """

 def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
 """

 
 Args:
 buffer_size: 
 obs_dim: 
 action_dim: 
 """
 self.buffer_size = buffer_size
 self.obs_dim = obs_dim
 self.action_dim = action_dim

 # 
 self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
 self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
 self.rewards = np.zeros(buffer_size, dtype=np.float32)
 self.values = np.zeros(buffer_size, dtype=np.float32)
 self.log_probs = np.zeros(buffer_size, dtype=np.float32)
 self.dones = np.zeros(buffer_size, dtype=np.bool_)

 # GAE
 self.advantages = np.zeros(buffer_size, dtype=np.float32)
 self.returns = np.zeros(buffer_size, dtype=np.float32)

 # 
 self.ptr = 0
 self.size = 0

 def store(self, 
 obs: np.ndarray, 
 action: np.ndarray, 
 reward: float, 
 value: float, 
 log_prob: float, 
 done: bool):
 """

 Store a single transition
 """
 assert self.ptr < self.buffer_size

 self.observations[self.ptr] = obs
 self.actions[self.ptr] = action
 self.rewards[self.ptr] = reward
 self.values[self.ptr] = value
 self.log_probs[self.ptr] = log_prob
 self.dones[self.ptr] = done

 self.ptr += 1
 self.size = min(self.size + 1, self.buffer_size)

 def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
 """
 (GAE)
 Compute returns and advantages using GAE
 """
 if self.size == 0:
 return

 # 
 advantages = np.zeros_like(self.rewards[:self.size])
 last_gae_lambda = 0

 for step in reversed(range(self.size)):
 if step == self.size - 1:
 next_non_terminal = 1.0 - self.dones[step]
 next_value = last_value
 else:
 next_non_terminal = 1.0 - self.dones[step]
 next_value = self.values[step + 1]

 delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
 advantages[step] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda

 # 
 returns = advantages + self.values[:self.size]

 # 
 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

 self.advantages[:self.size] = advantages
 self.returns[:self.size] = returns

 def get_batch(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
 """

 Get a batch of data
 """
 if batch_size is None:
 batch_size = self.size

 assert self.size >= batch_size

 # 
 indices = np.random.choice(self.size, batch_size, replace=False)

 batch = {
 'observations': torch.FloatTensor(self.observations[indices]),
 'actions': torch.FloatTensor(self.actions[indices]),
 'returns': torch.FloatTensor(self.returns[indices]),
 'advantages': torch.FloatTensor(self.advantages[indices]),
 'log_probs': torch.FloatTensor(self.log_probs[indices]),
 'values': torch.FloatTensor(self.values[indices])
 }

 return batch # type: ignore

 def clear(self):
 """"""
 self.ptr = 0
 self.size = 0

 def __len__(self):
 return self.size

 @property
 def is_full(self):
 return self.size == self.buffer_size
