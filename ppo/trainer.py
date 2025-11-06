"""
PPO Training Module for Pipeline Optimization
PPOnode_actionselect_node
Includes GAE, minibatching, and KL early stopping
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from env.pipeline_env import PipelineEnv
from .policy import PPOPolicy
from .buffer import RolloutBuffer

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOTrainer:
 def __init__(
 self,
 env: PipelineEnv,
 policy_type: str = 'mlp',
 hidden_size: int = 64,
 learning_rate: float = 3e-4,
 clip_ratio: float = 0.2,
 value_coef: float = 0.5,
 entropy_coef: float = 0.01,
 max_steps_per_episode: int = 20,
 ) -> None:
 self.env = env

 # Dimensions
 sample_obs = env.reset()
 obs_dim = self._calculate_obs_dim(sample_obs)
 action_dim = self._calculate_action_dim(env)

 # Policy & optimizer
 self.policy = PPOPolicy(obs_dim, action_dim, hidden_size)
 self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

 # PPO hyperparameters
 self.clip_ratio = clip_ratio
 self.value_coef = value_coef
 self.entropy_coef = entropy_coef
 self.max_steps_per_episode = max_steps_per_episode
 self.gamma = 0.99
 self.gae_lambda = 0.95
 self.update_epochs = 4
 self.minibatch_size = 64
 self.target_kl = 0.01

 # Buffer & stats
 self.buffer = None # type: RolloutBuffer | None
 self.episode_rewards = [] # type: List[float]
 self.episode_lengths = [] # type: List[int]

 def _calculate_obs_dim(self, obs: Dict[str, Any]) -> int:
 total_dim = 0
 for _, value in obs.items():
 if isinstance(value, np.ndarray):
 total_dim += int(np.prod(value.shape))
 elif isinstance(value, list):
 total_dim += len(value)
 else:
 total_dim += 1
 return total_dim

 def _calculate_action_dim(self, env: PipelineEnv) -> int:
 node_dim = len(env.pipeline_nodes)
 max_method_dim = max(len(m) for m in env.methods_for_node.values())
 param_dim = env.hyperparam_dim
 return node_dim + max_method_dim + param_dim

 def select_action(self, obs: Dict[str, Any]):
 with torch.no_grad():
 action_mask = torch.tensor(obs['action_mask'], dtype=torch.float32)
 node_logits, method_logits, params, value = self.policy(obs, action_mask=action_mask)

 # Node selection
 node_dist = torch.distributions.Categorical(logits=node_logits)
 node_action = node_dist.sample()
 node_log_prob = node_dist.log_prob(node_action)

 # Method selection
 node_idx = int(node_action.item())
 node_name = self.env.pipeline_nodes[node_idx]
 num_methods = len(self.env.methods_for_node[node_name])
 # Mask methods by available count AND per-node method_mask (if provided)
 method_logits_masked = method_logits[..., :num_methods]
 mm = obs.get('method_mask', None)
 if mm is not None:
 node_row = torch.tensor(mm[node_idx, :num_methods], dtype=torch.float32)
 invalid = (node_row <= 0).to(method_logits_masked.dtype)
 method_logits_masked = method_logits_masked + (invalid * -1e9)
 method_dist = torch.distributions.Categorical(logits=method_logits_masked)
 method_action = method_dist.sample()
 method_log_prob = method_dist.log_prob(method_action)

 # Param
 param_dist = torch.distributions.Normal(params, 0.1)
 param_action = torch.clamp(param_dist.sample(), 0.0, 1.0)
 param_log_prob = param_dist.log_prob(param_action).sum()

 action = {
 'node': int(node_action.item()),
 'method': int(method_action.item()),
 'params': param_action.numpy().tolist(),
 }

 log_probs = {
 'node': node_log_prob,
 'method': method_log_prob,
 'param': param_log_prob,
 'value': value,
 }

 return action, log_probs

 def train_episode(self) -> Tuple[float, int]:
 obs = self.env.reset()
 episode_reward = 0.0
 episode_length = 0

 # Init buffer
 sample_obs = self.env._get_obs()
 obs_vec = self._flatten_obs(sample_obs)
 if self.buffer is None:
 # action vector: [node, method, param0]
 self.buffer = RolloutBuffer(self.max_steps_per_episode, len(obs_vec), 3)

 for _ in range(self.max_steps_per_episode):
 action, log_prob_dict = self.select_action(obs)
 next_obs, reward, done, _, _ = self.env.step(action)

 obs_flat = self._flatten_obs(obs)
 node_logp = float(log_prob_dict['node'])
 value_v = float(log_prob_dict['value'])
 act_vec = np.array([
 action['node'],
 action['method'],
 action['params'][0] if action['params'] else 0.0,
 ], dtype=np.float32)
 self.buffer.store(obs_flat, act_vec, float(reward), value_v, node_logp, bool(done))

 episode_reward += float(reward)
 episode_length += 1
 obs = self.env.reset() if done else next_obs

 # GAE
 last_val = 0.0
 self.buffer.compute_returns_and_advantages(last_val, gamma=self.gamma, gae_lambda=self.gae_lambda)
 # Update
 self._update_policy_with_buffer()
 self.buffer.clear()
 return episode_reward, episode_length

 def _update_policy_with_buffer(self) -> None:
 if self.buffer is None or len(self.buffer) == 0:
 return
 for _ in range(self.update_epochs):
 batch_size = min(self.minibatch_size, len(self.buffer))
 batch = self.buffer.get_batch(batch_size)
 obs_batch = batch['observations']
 adv_batch = batch['advantages']
 ret_batch = batch['returns']
 actions_batch = batch['actions'] if 'actions' in batch else None
 node_idx_tensor = actions_batch[:, 0].long() if actions_batch is not None else torch.zeros(len(obs_batch), dtype=torch.long)

 node_logits, _, _, values = self.policy(obs_batch, action_mask=None)
 node_dist = torch.distributions.Categorical(logits=node_logits)
 new_logp = node_dist.log_prob(node_idx_tensor)
 entropy = node_dist.entropy().mean()

 old_logp = batch['log_probs']
 ratio = torch.exp(new_logp - old_logp)
 surr1 = ratio * adv_batch
 surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_batch
 policy_loss = -torch.min(surr1, surr2).mean()
 value_loss = F.mse_loss(values.squeeze(), ret_batch)
 loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

 approx_kl = (old_logp - new_logp).mean().item()
 if approx_kl > 1.5 * self.target_kl:
 break

 self.optimizer.zero_grad()
 loss.backward()
 nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
 self.optimizer.step()

 def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
 fp = np.array(obs.get('fingerprint', []), dtype=np.float32)
 visited = np.array(obs.get('node_visited', []), dtype=np.float32)
 mask = np.array(obs.get('action_mask', []), dtype=np.float32)
 mcount = np.array(obs.get('method_count', []), dtype=np.float32)
 mmask = obs.get('method_mask')
 mmask_flat = mmask.astype(np.float32).ravel() if isinstance(mmask, np.ndarray) else np.array([], dtype=np.float32)
 return np.concatenate([fp, visited, mask, mcount, mmask_flat], axis=0)

 def train(self, num_episodes: int = 100, log_interval: int = 10) -> None:
 logger.info(f"PPO{num_episodes}")
 for episode in range(num_episodes):
 episode_reward, episode_length = self.train_episode()
 self.episode_rewards.append(episode_reward)
 self.episode_lengths.append(episode_length)
 if episode % log_interval == 0:
 avg_reward = float(np.mean(self.episode_rewards[-10:]))
 avg_length = float(np.mean(self.episode_lengths[-10:]))
 logger.info(f" {episode}: ={avg_reward:.3f}, ={avg_length:.1f}")

 def save_model(self, path: str) -> None:
 torch.save({
 'policy_state_dict': self.policy.state_dict(),
 'optimizer_state_dict': self.optimizer.state_dict(),
 'episode_rewards': self.episode_rewards,
 'episode_lengths': self.episode_lengths,
 }, path)
 logger.info(f": {path}")

 def load_model(self, path: str) -> None:
 checkpoint = torch.load(path)
 self.policy.load_state_dict(checkpoint['policy_state_dict'])
 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 self.episode_rewards = checkpoint.get('episode_rewards', [])
 self.episode_lengths = checkpoint.get('episode_lengths', [])
 logger.info(f" {path} ")


def main() -> None:
 parser = argparse.ArgumentParser(description='PPO Pipeline Optimization')
 parser.add_argument('--episodes', type=int, default=100, help='')
 parser.add_argument('--lr', type=float, default=3e-4, help='')
 parser.add_argument('--hidden_size', type=int, default=64, help='')
 parser.add_argument('--steps_per_ep', type=int, default=20, help='episode')
 parser.add_argument('--save_path', type=str, default='models/ppo_pipeline.pth', help='')
 args = parser.parse_args()

 env = PipelineEnv()
 trainer = PPOTrainer(env, learning_rate=args.lr, hidden_size=args.hidden_size, max_steps_per_episode=args.steps_per_ep)
 trainer.train(num_episodes=args.episodes)
 os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
 trainer.save_model(args.save_path)


if __name__ == '__main__':
 main()
