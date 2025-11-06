#!/usr/bin/env python3
"""
PPO / PPO Enhancement Tests

PPOepisode
Tests PPO trainer enhancements including method count and training episodes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def test_env_observation_has_method_count():
 """
 / Test if environment observation has method count

 
 Verifies observation space contains available method count for each node
 """
 print("🧪 / Testing environment method count...")
 env = PipelineEnv()
 obs = env.reset()

 assert 'method_count' in obs, "method_count / Observation should contain method_count"
 assert len(obs['method_count']) == len(env.pipeline_nodes), \
 f" / Method count length should equal number of nodes"

 print(f" / Method count test passed")
 print(f" / Number of nodes: {len(env.pipeline_nodes)}")
 print(f" / Method counts: {obs['method_count']}")


def test_trainer_one_episode_runs():
 """
 episode / Test trainer one episode run

 PPOepisode
 Verifies PPO trainer can successfully run a complete training episode
 """
 print("\n🧪 episode / Testing trainer episode run...")
 env = PipelineEnv()
 trainer = PPOTrainer(env, max_steps_per_episode=5)

 reward, length = trainer.train_episode()

 assert isinstance(reward, float), " / Reward should be float"
 assert length == 5, f"Episode5 / Episode length should be 5"

 print(f" Episode / Episode run test passed")
 print(f" / Reward: {reward:.3f}")
 print(f" / Length: {length}")


if __name__ == "__main__":
 print("\n" + "="*70)
 print("START PPO / Starting PPO Enhancement Tests")
 print("="*70 + "\n")

 test_env_observation_has_method_count()
 test_trainer_one_episode_runs()

 print("\n" + "="*70)
 print(" / All tests passed!")
 print("="*70)
