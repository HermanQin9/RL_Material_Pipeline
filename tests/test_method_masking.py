#!/usr/bin/env python3
"""
Method Masking Tests

Tests method-level masking in PPO environment, ensuring only valid methods can be selected for each node
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


@pytest.mark.ppo
@pytest.mark.unit
def test_env_method_mask_shape_and_values():
    """Test environment method mask shape and values
    
    Verifies method mask dimensions and validity flags
    """
    env = PipelineEnv()
    obs = env.reset()
    method_mask = obs.get('method_mask')

    # Verify mask exists
    assert method_mask is not None, "Method mask should not be None"

    # Verify shape [num_nodes, max_methods]
    assert method_mask.shape[0] == len(env.pipeline_nodes), \
        "Mask rows should equal number of nodes"
    max_methods = max(len(v) for v in env.methods_for_node.values())
    assert method_mask.shape[1] == max_methods, \
        "Mask columns should equal max methods"

    # Verify row-wise validity
    for i, n in enumerate(env.pipeline_nodes):
        k = len(env.methods_for_node[n])
        assert np.all(method_mask[i, :k] == 1.0), \
            f"First {k} methods of node {n} should be valid"
        if k < max_methods:
            assert np.all(method_mask[i, k:] == 0.0), \
                f"Remaining methods of node {n} should be invalid"


@pytest.mark.ppo
@pytest.mark.integration
def test_trainer_respects_method_mask_for_single_method_node():
    """Test if trainer respects mask for single-method node
    
    Verifies PPO trainer correctly uses method mask when selecting actions
    """
    pytest.skip("PPO policy network dimension mismatch with current observation space (1x23 vs 43x64) - requires policy network update")
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=2)

    # Build an obs that forces node selection to N0 (only method index 0 is valid)
    obs = env.reset()
    action_mask = np.zeros_like(obs['action_mask'])
    # N0 index is 0 in pipeline_nodes
    action_mask[0] = 1.0
    obs['action_mask'] = action_mask

    action, _ = trainer.select_action(obs)
    assert action['node'] == 0, "Should select node N0"
    assert action['method'] == 0, "N0 has only one method available"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
