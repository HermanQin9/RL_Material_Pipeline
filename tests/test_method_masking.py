import numpy as np
import torch

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def test_env_method_mask_shape_and_values():
    env = PipelineEnv()
    obs = env.reset()
    method_mask = obs.get('method_mask')
    assert method_mask is not None
    # shape [num_nodes, max_methods]
    assert method_mask.shape[0] == len(env.pipeline_nodes)
    max_methods = max(len(v) for v in env.methods_for_node.values())
    assert method_mask.shape[1] == max_methods
    # row-wise validity
    for i, n in enumerate(env.pipeline_nodes):
        k = len(env.methods_for_node[n])
        assert np.all(method_mask[i, :k] == 1.0)
        if k < max_methods:
            assert np.all(method_mask[i, k:] == 0.0)


def test_trainer_respects_method_mask_for_single_method_node():
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=2)

    # Build an obs that forces node selection to N0 (only method index 0 is valid)
    obs = env.reset()
    action_mask = np.zeros_like(obs['action_mask'])
    # N0 index is 0 in pipeline_nodes
    action_mask[0] = 1.0
    obs['action_mask'] = action_mask

    action, _ = trainer.select_action(obs)
    assert action['node'] == 0  # N0
    assert action['method'] == 0  # only one method available
