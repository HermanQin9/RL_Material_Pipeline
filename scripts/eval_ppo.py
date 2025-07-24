"""
PPO evaluation script
PPO策略评估脚本
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env.pipeline_env import PipelineEnv
from ppo.policy import PPOPolicy
from ppo.trainer import PPOTrainer

def evaluate_policy(policy_path: str, num_episodes: int = 10, render: bool = False):
    """
    评估训练好的PPO策略
    Evaluate trained PPO policy
    
    Args:
        policy_path: 策略模型路径
        num_episodes: 评估回合数
        render: 是否显示详细信息
    """
    # 创建环境
    env = PipelineEnv()
    
    # 加载策略
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取观察和动作空间维度
    obs = env.reset()
    obs_dim = len(env._get_obs())
    action_dim = 3  # node + method + param
    
    # 创建策略网络
    policy = PPOPolicy(obs_dim, action_dim).to(device)
    
    # 加载训练好的权重
    try:
        checkpoint = torch.load(policy_path, map_location=device)
        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            policy.load_state_dict(checkpoint)
        print(f"✅ 成功加载策略: {policy_path}")
    except Exception as e:
        print(f"❌ 加载策略失败: {e}")
        return
    
    policy.eval()
    
    # 评估统计
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\n🎯 开始评估 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        if render:
            print(f"\n--- 回合 {episode + 1} ---")
        
        while not done and steps < 100:  # 防止无限循环
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(env._get_obs()).unsqueeze(0).to(device)
                action_probs = policy.actor(obs_tensor)
                
                # 贪婪选择动作 (评估时不使用随机性)
                node_idx = torch.argmax(action_probs[:, :env.num_nodes]).item()
                
                # 计算方法索引范围
                method_start = env.num_nodes
                node_name = env.pipeline_nodes[node_idx]
                num_methods = len(env.methods_for_node[node_name])
                method_idx = torch.argmax(action_probs[:, method_start:method_start+num_methods]).item()
                
                # 参数值
                param_idx = env.num_nodes + max(len(methods) for methods in env.methods_for_node.values())
                param_value = torch.sigmoid(action_probs[:, param_idx]).item()
                
                action = {
                    'node': node_idx,
                    'method': method_idx,
                    'params': [param_value]
                }
            
            if render:
                method_name = env.methods_for_node[node_name][method_idx]
                print(f"  步骤 {steps}: {node_name}.{method_name}(param={param_value:.3f})")
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if total_reward > 0:  # 假设正奖励表示成功
            success_count += 1
        
        if render:
            print(f"  回合奖励: {total_reward:.3f}, 步数: {steps}")
    
    # 计算统计信息
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    print(f"\n📊 评估结果:")
    print(f"  平均奖励: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"  平均步数: {avg_length:.1f}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  最佳奖励: {max(episode_rewards):.3f}")
    print(f"  最差奖励: {min(episode_rewards):.3f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards
    }

def compare_policies(policy_paths: list, num_episodes: int = 10):
    """
    比较多个策略的性能
    Compare performance of multiple policies
    """
    results = {}
    
    for policy_path in policy_paths:
        print(f"\n{'='*50}")
        print(f"评估策略: {policy_path}")
        print(f"{'='*50}")
        
        result = evaluate_policy(policy_path, num_episodes, render=False)
        results[policy_path] = result
    
    # 打印比较结果
    print(f"\n{'='*60}")
    print("📋 策略比较结果")
    print(f"{'='*60}")
    
    print(f"{'策略':<30} {'平均奖励':<12} {'成功率':<10} {'平均步数':<10}")
    print("-" * 60)
    
    for policy_path, result in results.items():
        policy_name = Path(policy_path).stem
        print(f"{policy_name:<30} {result['avg_reward']:<12.3f} {result['success_rate']:<10.1%} {result['avg_length']:<10.1f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO policy')
    parser.add_argument('--policy-path', type=str, required=True, help='Path to trained policy')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Show detailed episode information')
    parser.add_argument('--compare', nargs='+', help='Compare multiple policies')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较多个策略
        compare_policies(args.compare, args.episodes)
    else:
        # 评估单个策略
        evaluate_policy(args.policy_path, args.episodes, args.render)

if __name__ == "__main__":
    main()
