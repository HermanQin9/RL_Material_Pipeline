"""Evaluation helpers for PPO policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from env.pipeline_env import PipelineEnv
from ppo.policy import PPOPolicy


def evaluate_policy(policy_path: str, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
 """Evaluate a trained PPO policy stored at ``policy_path``."""
 env = PipelineEnv()
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 env.reset()
 obs_dim = len(env._get_obs())
 action_dim = 3

 policy = PPOPolicy(obs_dim, action_dim).to(device)
 checkpoint = torch.load(policy_path, map_location=device)
 state_dict = checkpoint.get("policy_state_dict", checkpoint)
 policy.load_state_dict(state_dict)
 policy.eval()

 episode_rewards: list[float] = []
 episode_lengths: list[int] = []
 success_count = 0

 print(f"\n {num_episodes} …")

 for episode in range(num_episodes):
 env.reset()
 total_reward = 0.0
 steps = 0
 done = False

 if render:
 print(f"\n--- {episode + 1} ---")

 while not done and steps < 100:
 with torch.no_grad():
 obs_tensor = torch.FloatTensor(env._get_obs()).unsqueeze(0).to(device)
 action_probs = policy.actor(obs_tensor)

 node_idx_raw = torch.argmax(action_probs[:, : env.num_nodes]).item()
 node_idx = int(node_idx_raw)
 method_start = env.num_nodes
 node_name = env.pipeline_nodes[node_idx]
 num_methods = len(env.methods_for_node[node_name])
 method_idx_raw = torch.argmax(action_probs[:, method_start : method_start + num_methods]).item()
 method_idx = int(method_idx_raw)
 param_idx = env.num_nodes + max(len(m) for m in env.methods_for_node.values())
 param_value = torch.sigmoid(action_probs[:, param_idx]).item()

 action = {"node": node_idx, "method": method_idx, "params": [param_value]}

 if render:
 method_name = env.methods_for_node[node_name][method_idx]
 print(f" {steps}: {node_name}.{method_name}(param={param_value:.3f})")

 _, reward, terminated, truncated, _ = env.step(action)
 done = terminated or truncated
 total_reward += reward
 steps += 1

 episode_rewards.append(total_reward)
 episode_lengths.append(steps)
 if total_reward > 0:
 success_count += 1
 if render:
 print(f" : {total_reward:.3f}, : {steps}")

 avg_reward = float(np.mean(episode_rewards))
 std_reward = float(np.std(episode_rewards))
 avg_length = float(np.mean(episode_lengths))
 success_rate = success_count / num_episodes

 print("\n :")
 print(f" : {avg_reward:.3f} ± {std_reward:.3f}")
 print(f" : {avg_length:.1f}")
 print(f" : {success_rate:.1%}")
 print(f" : {max(episode_rewards):.3f}")
 print(f" : {min(episode_rewards):.3f}")

 return {
 "avg_reward": avg_reward,
 "std_reward": std_reward,
 "avg_length": avg_length,
 "success_rate": success_rate,
 "episode_rewards": episode_rewards,
 }


def compare_policies(policy_paths: Iterable[str], num_episodes: int = 10) -> Dict[str, Dict[str, float]]:
 """Evaluate multiple policies and print a comparison table."""
 results: Dict[str, Dict[str, float]] = {}
 for policy_path in policy_paths:
 print("\n" + "=" * 50)
 print(f": {policy_path}")
 print("=" * 50)
 results[policy_path] = evaluate_policy(policy_path, num_episodes, render=False)

 print("\n" + "=" * 60)
 print(" ")
 print("=" * 60)
 print(f"{'':<30} {'':<12} {'':<10} {'':<10}")
 print("-" * 60)
 for policy_path, result in results.items():
 policy_name = Path(policy_path).stem
 print(
 f"{policy_name:<30} {result['avg_reward']:<12.3f} {result['success_rate']:<10.1%} {result['avg_length']:<10.1f}"
 )
 return results
