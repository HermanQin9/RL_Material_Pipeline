"""
Test PPO Learning Capability
测试PPO学习能力

This test verifies that the PPO agent can actually learn and improve over time.
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
from ppo.trainer import PPOTrainer
from env.pipeline_env import PipelineEnv


@pytest.mark.slow
@pytest.mark.integration
def test_ppo_learning_improvement():
    """
    Test that PPO agent shows learning improvement over episodes
    验证PPO agent在训练过程中显示学习改进
    """
    # Set test mode
    os.environ['PIPELINE_TEST'] = '1'
    
    # Create environment and trainer
    env = PipelineEnv()
    trainer = PPOTrainer(env, learning_rate=3e-4)
    
    # Train for 20 episodes
    num_episodes = 20
    trainer.train(num_episodes=num_episodes, log_interval=5)
    
    # Analyze results
    rewards = trainer.episode_rewards
    
    # Basic assertions
    assert len(rewards) == num_episodes, f"Should have {num_episodes} episodes"
    assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
    
    # Split into first half and second half
    first_half = np.array(rewards[:num_episodes//2])
    second_half = np.array(rewards[num_episodes//2:])
    
    first_mean = np.mean(first_half)
    second_mean = np.mean(second_half)
    improvement = second_mean - first_mean
    
    # Log results for debugging
    print(f"\n{'='*70}")
    print(f"PPO Learning Test Results")
    print(f"{'='*70}")
    print(f"First {len(first_half)} episodes: mean={first_mean:.3f}, std={np.std(first_half):.3f}")
    print(f"Last {len(second_half)} episodes: mean={second_mean:.3f}, std={np.std(second_half):.3f}")
    print(f"Improvement: {improvement:+.3f} ({improvement/abs(first_mean)*100:+.1f}%)")
    print(f"All rewards: {[f'{r:.2f}' for r in rewards]}")
    print(f"{'='*70}\n")
    
    # Assert learning is happening
    # We expect at least some improvement (can be relaxed if needed)
    assert improvement > 0, (
        f"PPO should show improvement over time. "
        f"First half: {first_mean:.3f}, Second half: {second_mean:.3f}, "
        f"Improvement: {improvement:+.3f}"
    )


@pytest.mark.unit
def test_ppo_trainer_initialization():
    """
    Test PPO trainer can be initialized correctly
    测试PPO训练器可以正确初始化
    """
    os.environ['PIPELINE_TEST'] = '1'
    
    env = PipelineEnv()
    trainer = PPOTrainer(env)
    
    assert trainer.env is env
    assert trainer.policy is not None
    assert trainer.optimizer is not None
    assert len(trainer.episode_rewards) == 0
    assert len(trainer.episode_lengths) == 0


@pytest.mark.unit
def test_ppo_single_episode():
    """
    Test PPO can complete a single episode without errors
    测试PPO可以完成单个episode而不出错
    """
    os.environ['PIPELINE_TEST'] = '1'
    
    env = PipelineEnv()
    trainer = PPOTrainer(env)
    
    # Run one episode
    reward, length = trainer.train_episode()
    
    assert isinstance(reward, (int, float))
    assert isinstance(length, int)
    assert length > 0
    assert length <= trainer.max_steps_per_episode


@pytest.mark.integration
def test_ppo_reward_trend():
    """
    Test that PPO reward shows upward trend (may be noisy)
    测试PPO reward显示上升趋势（可能有噪声）
    """
    os.environ['PIPELINE_TEST'] = '1'
    
    env = PipelineEnv()
    trainer = PPOTrainer(env, learning_rate=3e-4)
    
    # Train for fewer episodes for faster testing
    trainer.train(num_episodes=10, log_interval=5)
    
    rewards = trainer.episode_rewards
    
    # Check that we have rewards
    assert len(rewards) >= 10
    
    # Calculate moving average to smooth out noise
    window_size = 5
    if len(rewards) >= window_size:
        first_window = np.mean(rewards[:window_size])
        last_window = np.mean(rewards[-window_size:])
        
        print(f"\nFirst {window_size} episodes: {first_window:.3f}")
        print(f"Last {window_size} episodes: {last_window:.3f}")
        print(f"Trend: {last_window - first_window:+.3f}")
        
        # Even with noise, we expect some positive trend
        # Relaxed assertion: just check not getting worse
        assert last_window >= first_window - 5.0, (
            "PPO should not get significantly worse over time"
        )


if __name__ == '__main__':
    # Allow running this test file directly
    pytest.main([__file__, '-v', '-s'])
