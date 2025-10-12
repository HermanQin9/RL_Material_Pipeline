#!/usr/bin/env python3
"""
PPO增强功能测试 / PPO Enhancement Tests

测试PPO训练器的增强功能，包括方法计数和训练episode
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
    测试环境观察是否包含方法计数 / Test if environment observation has method count
    
    验证观察空间中包含每个节点的可用方法数量信息
    Verifies observation space contains available method count for each node
    """
    print("🧪 测试环境方法计数 / Testing environment method count...")
    env = PipelineEnv()
    obs = env.reset()
    
    assert 'method_count' in obs, "观察空间应包含method_count / Observation should contain method_count"
    assert len(obs['method_count']) == len(env.pipeline_nodes), \
        f"方法计数长度应等于节点数 / Method count length should equal number of nodes"
    
    print(f"   ✓ 方法计数测试通过 / Method count test passed")
    print(f"   节点数 / Number of nodes: {len(env.pipeline_nodes)}")
    print(f"   方法计数 / Method counts: {obs['method_count']}")


def test_trainer_one_episode_runs():
    """
    测试训练器单个episode运行 / Test trainer one episode run
    
    验证PPO训练器可以成功运行一个完整的训练episode
    Verifies PPO trainer can successfully run a complete training episode
    """
    print("\n🧪 测试训练器episode运行 / Testing trainer episode run...")
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=5)
    
    reward, length = trainer.train_episode()
    
    assert isinstance(reward, float), "奖励应为浮点数 / Reward should be float"
    assert length == 5, f"Episode长度应为5 / Episode length should be 5"
    
    print(f"   ✓ Episode运行测试通过 / Episode run test passed")
    print(f"   奖励 / Reward: {reward:.3f}")
    print(f"   长度 / Length: {length}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 开始PPO增强功能测试 / Starting PPO Enhancement Tests")
    print("="*70 + "\n")
    
    test_env_observation_has_method_count()
    test_trainer_one_episode_runs()
    
    print("\n" + "="*70)
    print("🎉 所有测试通过！ / All tests passed!")
    print("="*70)
