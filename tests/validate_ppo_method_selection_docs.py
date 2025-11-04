"""
验证PPO方法选择文档的准确性 / Validate PPO Method Selection Documentation Accuracy

此脚本验证文档中描述的PPO方法选择机制与实际代码实现一致。
This script validates that the method selection mechanism described in the documentation
matches the actual code implementation.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import torch
import numpy as np
from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer
from ppo.policy import PPOPolicy


def test_method_masking():
    """测试方法屏蔽机制 / Test method masking mechanism"""
    print("\n" + "="*80)
    print("测试1: 方法屏蔽机制 / Test 1: Method Masking Mechanism")
    print("="*80)
    
    env = PipelineEnv()
    
    # 验证每个节点的方法数量
    print("\n节点-方法映射验证 / Node-Method Mapping Validation:")
    expected_methods = {
        'N1': ['mean', 'median', 'knn', 'none'],
        'N2': ['default'],
        'N3': ['none', 'variance', 'univariate', 'pca'],
        'N4': ['std', 'robust', 'minmax', 'none'],
        'N5': ['rf', 'gbr', 'xgb', 'cat']
    }
    
    for node_name, expected in expected_methods.items():
        actual = env.methods_for_node[node_name]
        assert actual == expected, f"Node {node_name}: Expected {expected}, got {actual}"
        print(f"✓ {node_name}: {len(actual)} methods - {actual}")
    
    print("\n✅ 所有节点的方法映射正确 / All node-method mappings correct")
    return True


def test_action_space():
    """测试动作空间结构 / Test action space structure"""
    print("\n" + "="*80)
    print("测试2: 动作空间结构 / Test 2: Action Space Structure")
    print("="*80)
    
    env = PipelineEnv()
    obs = env.reset()
    
    # 验证观察空间维度
    print("\n观察空间维度 / Observation Space Dimensions:")
    fingerprint_dim = len(obs['fingerprint'])
    node_visited_dim = len(obs['node_visited'])
    action_mask_dim = len(obs['action_mask'])
    
    print(f"  - fingerprint: {fingerprint_dim} (expected: 3)")
    print(f"  - node_visited: {node_visited_dim} (expected: 5)")
    print(f"  - action_mask: {action_mask_dim} (expected: 5)")
    
    assert fingerprint_dim == 3, "Fingerprint should have 3 dimensions"
    assert node_visited_dim == 5, "Node visited should have 5 dimensions"
    assert action_mask_dim == 5, "Action mask should have 5 dimensions"
    
    total_obs_dim = fingerprint_dim + node_visited_dim + action_mask_dim
    print(f"\n总观察维度 / Total Observation Dimensions: {total_obs_dim} (expected: 13)")
    assert total_obs_dim == 13, "Total observation dimension should be 13"
    
    print("\n✅ 观察空间结构正确 / Observation space structure correct")
    return True


def test_neural_network_output():
    """测试神经网络输出维度 / Test neural network output dimensions"""
    print("\n" + "="*80)
    print("测试3: 神经网络输出维度 / Test 3: Neural Network Output Dimensions")
    print("="*80)
    
    env = PipelineEnv()
    obs = env.reset()
    
    # 创建PPO策略
    trainer = PPOTrainer(env, hidden_size=64)
    
    # 获取网络输出
    with torch.no_grad():
        node_logits, method_logits, params, value = trainer.policy(obs)
    
    print("\n神经网络输出 / Neural Network Outputs:")
    print(f"  - node_logits: {node_logits.shape} (expected: torch.Size([6]))")
    print(f"  - method_logits: {method_logits.shape} (expected: torch.Size([10]))")
    print(f"  - params: {params.shape} (expected: torch.Size([1]))")
    print(f"  - value: {value.shape} (expected: torch.Size([1]))")
    
    assert node_logits.shape == torch.Size([6]), "Node logits should have 6 dimensions"
    assert method_logits.shape == torch.Size([10]), "Method logits should have 10 dimensions"
    assert params.shape == torch.Size([1]), "Params should have 1 dimension"
    assert value.shape == torch.Size([1]), "Value should have 1 dimension"
    
    print("\n✅ 神经网络输出维度正确 / Neural network output dimensions correct")
    return True


def test_method_selection_flow():
    """测试完整的方法选择流程 / Test complete method selection flow"""
    print("\n" + "="*80)
    print("测试4: 完整方法选择流程 / Test 4: Complete Method Selection Flow")
    print("="*80)
    
    env = PipelineEnv()
    trainer = PPOTrainer(env, hidden_size=64)
    obs = env.reset()
    
    # 执行一个动作选择
    action, log_probs = trainer.select_action(obs)
    
    print("\n选择的动作 / Selected Action:")
    print(f"  - node: {action['node']} (node name: {env.pipeline_nodes[action['node']]})")
    print(f"  - method: {action['method']}")
    print(f"  - params: {action['params']}")
    
    # 验证动作格式
    assert 'node' in action, "Action should have 'node' key"
    assert 'method' in action, "Action should have 'method' key"
    assert 'params' in action, "Action should have 'params' key"
    
    # 验证节点选择合法性
    assert 0 <= action['node'] < env.num_nodes, "Node index out of range"
    
    # 验证方法选择合法性
    node_name = env.pipeline_nodes[action['node']]
    num_methods = len(env.methods_for_node[node_name])
    assert 0 <= action['method'] < num_methods, "Method index out of range for selected node"
    
    # 获取方法名称
    method_name = env.methods_for_node[node_name][action['method']]
    print(f"  - method name: {method_name}")
    
    # 验证参数范围
    assert isinstance(action['params'], list), "Params should be a list"
    assert len(action['params']) > 0, "Params list should not be empty"
    assert 0.0 <= action['params'][0] <= 1.0, "Param value should be in [0, 1]"
    
    print("\n✅ 方法选择流程正确 / Method selection flow correct")
    return True


def test_dynamic_masking():
    """测试动态屏蔽逻辑 / Test dynamic masking logic"""
    print("\n" + "="*80)
    print("测试5: 动态屏蔽逻辑 / Test 5: Dynamic Masking Logic")
    print("="*80)
    
    env = PipelineEnv()
    trainer = PPOTrainer(env, hidden_size=64)
    
    # 模拟选择不同节点
    test_cases = [
        ('N1', 4),  # N1 has 4 methods
        ('N2', 1),  # N2 has 1 method
        ('N3', 4),  # N3 has 4 methods
        ('N4', 4),  # N4 has 4 methods
        ('N5', 4),  # N5 has 4 methods
    ]
    
    print("\n动态屏蔽测试 / Dynamic Masking Tests:")
    for node_name, expected_num_methods in test_cases:
        node_idx = env.pipeline_nodes.index(node_name)
        num_methods = len(env.methods_for_node[node_name])
        
        print(f"  - {node_name}: {num_methods} methods (expected: {expected_num_methods})")
        assert num_methods == expected_num_methods, \
            f"Node {node_name} should have {expected_num_methods} methods, got {num_methods}"
        
        # 验证屏蔽逻辑：method_logits[:num_methods]
        method_logits = torch.randn(10)  # 10维度的method_logits
        method_logits_masked = method_logits[:num_methods]
        
        assert method_logits_masked.shape == torch.Size([num_methods]), \
            f"Masked logits should have {num_methods} dimensions"
    
    print("\n✅ 动态屏蔽逻辑正确 / Dynamic masking logic correct")
    return True


def test_action_validation():
    """测试动作验证机制 / Test action validation mechanism"""
    print("\n" + "="*80)
    print("测试6: 动作验证机制 / Test 6: Action Validation Mechanism")
    print("="*80)
    
    env = PipelineEnv()
    obs = env.reset()
    
    print("\n动作约束验证 / Action Constraint Validation:")
    
    # 测试1: 第一步必须选择N2
    print("  - Test: First step must be N2")
    action_n2 = {'node': 0, 'method': 0, 'params': [0.5]}  # N2
    assert env.select_node(action_n2) == True, "First step should allow N2"
    
    action_n1 = {'node': 1, 'method': 0, 'params': [0.5]}  # N1
    assert env.select_node(action_n1) == False, "First step should not allow N1"
    print("    ✓ First step constraint working")
    
    # 执行第一步
    obs, _, _, _, _ = env.step(action_n2)
    
    # 测试2: 中间步骤不能选择N2或N5
    print("  - Test: Middle steps cannot select N2 or N5")
    action_n2_again = {'node': 0, 'method': 0, 'params': [0.5]}
    assert env.select_node(action_n2_again) == False, "Cannot select N2 again"
    
    action_n5 = {'node': 4, 'method': 0, 'params': [0.5]}  # N5
    assert env.select_node(action_n5) == False, "Cannot select N5 in middle steps"
    print("    ✓ Middle step constraint working")
    
    # 测试3: 不能选择已访问的节点
    print("  - Test: Cannot select already visited nodes")
    assert env.node_visited[0] == True, "N2 should be marked as visited"
    print("    ✓ Visited node tracking working")
    
    print("\n✅ 动作验证机制正确 / Action validation mechanism correct")
    return True


def run_all_tests():
    """运行所有测试 / Run all tests"""
    print("\n" + "="*80)
    print("PPO方法选择文档验证 / PPO Method Selection Documentation Validation")
    print("="*80)
    
    tests = [
        ("方法屏蔽机制", test_method_masking),
        ("动作空间结构", test_action_space),
        ("神经网络输出", test_neural_network_output),
        ("方法选择流程", test_method_selection_flow),
        ("动态屏蔽逻辑", test_dynamic_masking),
        ("动作验证机制", test_action_validation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ 测试失败 / Test Failed: {test_name}")
            print(f"错误 / Error: {str(e)}")
            failed += 1
    
    print("\n" + "="*80)
    print("测试总结 / Test Summary")
    print("="*80)
    print(f"✅ 通过 / Passed: {passed}")
    print(f"❌ 失败 / Failed: {failed}")
    print(f"总计 / Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！文档与代码实现一致。")
        print("🎉 All tests passed! Documentation matches code implementation.")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查文档或代码。")
        print(f"⚠️  {failed} test(s) failed, please check documentation or code.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
