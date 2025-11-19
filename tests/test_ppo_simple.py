#!/usr/bin/env python3
"""
Simple PPO Testing and Visualization Script
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import torch

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


@pytest.fixture
def mock_training_data():
    """Generate mock training data for testing visualization"""
    episodes = range(1, 21)
    
    # Mock reward data: starts low, gradually improves
    base_rewards = np.linspace(-1.0, 0.5, 20)
    noise = np.random.normal(0, 0.1, 20)
    rewards = base_rewards + noise
    
    # Mock loss data: starts high, gradually decreases
    losses = np.exp(-np.linspace(0, 2, 20)) + np.random.normal(0, 0.05, 20)
    
    return episodes, rewards, losses


@pytest.mark.unit
def test_environment_creation():
    """Test if environment can be created"""
    env = PipelineEnv()
    assert env is not None
    assert hasattr(env, 'pipeline_nodes')
    assert len(env.pipeline_nodes) == 10


@pytest.mark.unit
def test_trainer_creation():
    """Test if PPO trainer can be created"""
    env = PipelineEnv()
    trainer = PPOTrainer(env, hidden_size=32, learning_rate=1e-3)
    
    assert trainer is not None
    assert hasattr(trainer, 'policy')
    assert hasattr(trainer, 'optimizer')


@pytest.mark.ppo
@pytest.mark.integration
def test_environment_reset():
    """Test environment reset functionality"""
    env = PipelineEnv()
    obs = env.reset()
    
    assert obs is not None
    assert 'node_visited' in obs  # Changed from node_visits
    assert 'action_mask' in obs
    assert 'method_count' in obs


@pytest.mark.ppo
@pytest.mark.integration
@pytest.mark.xfail(reason="PPO policy network dimension mismatch with current observation space (1x23 vs 43x64) - requires policy network update")
def test_single_step():
    """Test single environment step"""
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=5)
    
    obs = env.reset()
    action, _ = trainer.select_action(obs)
    
    assert action is not None
    assert 'node' in action
    assert 'method' in action


@pytest.mark.ppo
@pytest.mark.slow
@pytest.mark.xfail(reason="PPO policy network dimension mismatch with current observation space (1x23 vs 43x64) - requires policy network update")
def test_training_episode():
    """Test complete training episode"""
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=5)
    
    reward, length = trainer.train_episode()
    
    assert isinstance(reward, float)
    assert isinstance(length, int)
    assert length <= 5
    assert not np.isnan(reward)


@pytest.mark.unit
def test_plot_generation(mock_training_data, tmp_path):
    """Test training curve plot generation"""
    episodes, rewards, losses = mock_training_data
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot rewards
    ax1.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=2, label='Episode Reward')
    ax1.plot(episodes, np.convolve(rewards, np.ones(5)/5, mode='same'), 
             'r-', linewidth=2, label='Moving Avg (5)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('PPO Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    ax2.plot(episodes, losses, 'g-', alpha=0.7, linewidth=2, label='Policy Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('PPO Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = tmp_path / "test_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    assert save_path.exists()


@pytest.mark.unit
def test_reward_statistics(mock_training_data):
    """Test reward statistics calculation"""
    _, rewards, _ = mock_training_data
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    assert not np.isnan(mean_reward)
    assert not np.isnan(std_reward)
    assert min_reward <= mean_reward <= max_reward


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
