#!/usr/bin/env python3
"""
方法掩码测试 / Method Masking Tests

测试PPO环境中的方法级掩码功能，确保每个节点只能选择有效的方法
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
    测试环境的方法掩码形状和值 / Test environment method mask shape and values
    
    验证方法掩码的维度正确性和有效性标记
    Verifies method mask dimensions and validity flags
    """
    print("🧪 测试方法掩码形状和值 / Testing method mask shape and values...")
    env = PipelineEnv()
    obs = env.reset()
    method_mask = obs.get('method_mask')
    
    # 验证掩码存在 / Verify mask exists
    assert method_mask is not None, "方法掩码不应为None / Method mask should not be None"
    
    # 验证形状 [节点数, 最大方法数] / Verify shape [num_nodes, max_methods]
    assert method_mask.shape[0] == len(env.pipeline_nodes), \
        f"掩码行数应等于节点数 / Mask rows should equal number of nodes"
    max_methods = max(len(v) for v in env.methods_for_node.values())
    assert method_mask.shape[1] == max_methods, \
        f"掩码列数应等于最大方法数 / Mask columns should equal max methods"
    
    # 验证每行的有效性 / Verify row-wise validity
    for i, n in enumerate(env.pipeline_nodes):
        k = len(env.methods_for_node[n])
        assert np.all(method_mask[i, :k] == 1.0), \
            f"节点 {n} 的前 {k} 个方法应为有效 / First {k} methods of node {n} should be valid"
        if k < max_methods:
            assert np.all(method_mask[i, k:] == 0.0), \
                f"节点 {n} 的后续方法应为无效 / Remaining methods of node {n} should be invalid"
    
    print(f"   ✓ 方法掩码形状: {method_mask.shape} / Method mask shape: {method_mask.shape}")
    print(f"   ✓ 节点数: {len(env.pipeline_nodes)} / Number of nodes: {len(env.pipeline_nodes)}")
    print(f"   ✓ 最大方法数: {max_methods} / Max methods: {max_methods}")
    print("✅ 方法掩码形状和值测试通过 / Method mask shape and values test passed")


def test_trainer_respects_method_mask_for_single_method_node():
    """
    测试训练器是否遵守单方法节点的掩码 / Test if trainer respects mask for single-method node
    
    验证PPO训练器在选择动作时正确使用方法掩码
    Verifies PPO trainer correctly uses method mask when selecting actions
    """
    print("\n🧪 测试训练器方法掩码遵守性 / Testing trainer method mask compliance...")
    env = PipelineEnv()
    trainer = PPOTrainer(env, max_steps_per_episode=2)

    # 构建一个强制选择N0节点的观察（只有方法索引0有效）
    # Build an obs that forces node selection to N0 (only method index 0 is valid)
    obs = env.reset()
    action_mask = np.zeros_like(obs['action_mask'])
    # N0索引在pipeline_nodes中为0 / N0 index is 0 in pipeline_nodes
    action_mask[0] = 1.0
    obs['action_mask'] = action_mask

    action, _ = trainer.select_action(obs)
    assert action['node'] == 0, "应选择节点N0 / Should select node N0"
    assert action['method'] == 0, "N0只有一个方法可用 / N0 has only one method available"
    
    print(f"   ✓ 选择的节点: N{action['node']} / Selected node: N{action['node']}")
    print(f"   ✓ 选择的方法: {action['method']} / Selected method: {action['method']}")
    print("✅ 训练器方法掩码遵守性测试通过 / Trainer method mask compliance test passed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 开始方法掩码测试 / Starting Method Masking Tests")
    print("="*70 + "\n")
    
    test_env_method_mask_shape_and_values()
    test_trainer_respects_method_mask_for_single_method_node()
    
    print("\n" + "="*70)
    print("🎉 所有测试通过！ / All tests passed!")
    print("="*70)
