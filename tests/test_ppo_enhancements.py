#!/usr/bin/env python3
"""
PPO - PPO Enhancement Tests

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


@pytest.mark.ppo
@pytest.mark.unit
def test_env_observation_has_method_count():
    """Test if environment observation has method count
    
    Verifies observation space contains available method count for each node
    """
    env = PipelineEnv()
    obs = env.reset()

    assert 'method_count' in obs, "Observation should contain method_count"
    assert len(obs['method_count']) == len(env.pipeline_nodes), \
        "Method count length should equal number of nodes"
    
    # Verify method counts are positive
    for count in obs['method_count']:
        assert count > 0, "Each node should have at least one method"


@pytest.mark.ppo
@pytest.mark.integration
def test_trainer_one_episode_runs():
    """Test trainer one episode run
    
    Verifies PPO trainer can successfully run a complete training episode
    """
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=5)

    reward, length = trainer.train_episode()

    assert isinstance(reward, float), "Reward should be float"
    assert length == 5, "Episode length should be 5"
    assert not np.isnan(reward), "Reward should not be NaN"


if __name__ == "__main__":
    # Allow running directly with: python test_ppo_enhancements.py
    pytest.main([__file__, '-v', '--tb=short'])
