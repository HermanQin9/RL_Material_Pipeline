#!/usr/bin/env python3
"""
 / Method Masking Tests

PPO
Tests method-level masking in PPO environment, ensuring only valid methods can be selected for each node
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def test_env_method_mask_shape_and_values():
 """
 / Test environment method mask shape and values

 
 Verifies method mask dimensions and validity flags
 """
 print("🧪 / Testing method mask shape and values...")
 env = PipelineEnv()
 obs = env.reset()
 method_mask = obs.get('method_mask')

 # / Verify mask exists
 assert method_mask is not None, "None / Method mask should not be None"

 # [, ] / Verify shape [num_nodes, max_methods]
 assert method_mask.shape[0] == len(env.pipeline_nodes), \
 f" / Mask rows should equal number of nodes"
 max_methods = max(len(v) for v in env.methods_for_node.values())
 assert method_mask.shape[1] == max_methods, \
 f" / Mask columns should equal max methods"

 # / Verify row-wise validity
 for i, n in enumerate(env.pipeline_nodes):
 k = len(env.methods_for_node[n])
 assert np.all(method_mask[i, :k] == 1.0), \
 f" {n} {k} / First {k} methods of node {n} should be valid"
 if k < max_methods:
 assert np.all(method_mask[i, k:] == 0.0), \
 f" {n} / Remaining methods of node {n} should be invalid"

 print(f" : {method_mask.shape} / Method mask shape: {method_mask.shape}")
 print(f" : {len(env.pipeline_nodes)} / Number of nodes: {len(env.pipeline_nodes)}")
 print(f" : {max_methods} / Max methods: {max_methods}")
 print("SUCCESS / Method mask shape and values test passed")


def test_trainer_respects_method_mask_for_single_method_node():
 """
 / Test if trainer respects mask for single-method node

 PPO
 Verifies PPO trainer correctly uses method mask when selecting actions
 """
 print("\n🧪 / Testing trainer method mask compliance...")
 env = PipelineEnv()
 trainer = PPOTrainer(env, max_steps_per_episode=2)

 # N00
 # Build an obs that forces node selection to N0 (only method index 0 is valid)
 obs = env.reset()
 action_mask = np.zeros_like(obs['action_mask'])
 # N0pipeline_nodes0 / N0 index is 0 in pipeline_nodes
 action_mask[0] = 1.0
 obs['action_mask'] = action_mask

 action, _ = trainer.select_action(obs)
 assert action['node'] == 0, "N0 / Should select node N0"
 assert action['method'] == 0, "N0 / N0 has only one method available"

 print(f" : N{action['node']} / Selected node: N{action['node']}")
 print(f" : {action['method']} / Selected method: {action['method']}")
 print("SUCCESS / Trainer method mask compliance test passed")


if __name__ == "__main__":
 print("\n" + "="*70)
 print("START / Starting Method Masking Tests")
 print("="*70 + "\n")

 test_env_method_mask_shape_and_values()
 test_trainer_respects_method_mask_for_single_method_node()

 print("\n" + "="*70)
 print(" / All tests passed!")
 print("="*70)
