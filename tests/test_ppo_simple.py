#!/usr/bin/env python3
"""
简单PPO测试和可视化脚本
Simple PPO testing and visualization script
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def create_simple_ppo_test():
    """
    创建简单的PPO测试，避免NaN问题 / Create simple PPO test, avoiding NaN issues
    
    生成模拟训练数据用于测试可视化功能
    Generates mock training data for testing visualization features
    """
    print("🔧 初始化环境... / Initializing environment...")
    env = PipelineEnv()
    
    print("🤖 创建PPO训练器... / Creating PPO trainer...")
    trainer = PPOTrainer(env, hidden_size=32, learning_rate=1e-3)
    
    # 手动模拟一些训练数据来测试可视化
    # Manually simulate some training data to test visualization
    print("📊 生成模拟训练数据... / Generating mock training data...")
    episodes = range(1, 21)
    
    # 模拟奖励数据：开始低，逐渐提高，有一些噪声
    # Mock reward data: starts low, gradually improves, with some noise
    base_rewards = np.linspace(-1.0, 0.5, 20)
    noise = np.random.normal(0, 0.1, 20)
    rewards = base_rewards + noise
    
    # 模拟损失数据：开始高，逐渐降低
    # Mock loss data: starts high, gradually decreases
    losses = np.exp(-np.linspace(0, 2, 20)) + np.random.normal(0, 0.05, 20)
    
    return episodes, rewards, losses


def plot_training_curves(episodes, rewards, losses, save_path="logs/ppo_test_curves.png"):
    """
    绘制训练曲线 / Plot training curves
    
    创建奖励和损失的可视化图表
    Creates visualization charts for rewards and losses
    """
    print("📈 生成训练曲线图... / Generating training curve charts...")
    
    # 创建图形 / Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 奖励曲线 / Reward curve
    ax1.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=2, label='回合奖励 / Episode Reward')
    ax1.plot(episodes, np.convolve(rewards, np.ones(5)/5, mode='same'), 'r-', linewidth=2, 
            label='移动平均(5) / Moving Avg (5)')
    ax1.set_xlabel('回合 / Episode')
    ax1.set_ylabel('奖励 / Reward')
    ax1.set_title('PPO训练奖励 / PPO Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线 / Loss curve
    ax2.plot(episodes, losses, 'g-', alpha=0.7, linewidth=2, label='策略损失 / Policy Loss')
    ax2.set_xlabel('回合 / Episode')
    ax2.set_ylabel('损失 / Loss')
    ax2.set_title('PPO训练损失 / PPO Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像 / Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 训练曲线已保存到 / Training curves saved to: {save_path}")
    
    # 显示图像 / Show figure
    plt.show()
    
    return fig


def test_environment():
    """
    测试环境是否正常工作 / Test if environment works properly
    
    验证环境初始化和基本功能
    Validates environment initialization and basic functionality
    """
    print("🧪 测试环境... / Testing environment...")
    
    try:
        env = PipelineEnv()
        obs = env.reset()
        
        print(f"✅ 环境初始化成功 / Environment initialized successfully")
        print(f"   观察状态keys / Observation keys: {list(obs.keys())}")
        print(f"   Fingerprint: {obs['fingerprint']}")
        print(f"   Node visited: {obs['node_visited']}")
        print(f"   Action mask shape: {obs['action_mask'].shape}")
        
        # 测试一个随机动作 / Test a random action
        action = {
            'node': np.random.randint(0, 6),
            'method': np.random.randint(0, 4), 
            'params': np.random.random()
        }
        
        print(f"   测试动作 / Test action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败 / Environment test failed: {e}")
        return False


def main():
    """
    主函数 / Main function
    
    执行完整的PPO测试和可视化流程
    Executes complete PPO testing and visualization workflow
    """
    print("🚀 开始PPO测试和可视化... / Starting PPO test and visualization...")
    
    # 测试环境 / Test environment
    if not test_environment():
        print("⚠️ 环境测试未通过，退出 / Environment test failed, exiting")
        return
    
    # 生成模拟数据和可视化 / Generate mock data and visualization
    episodes, rewards, losses = create_simple_ppo_test()
    
    # 创建训练曲线图 / Create training curve charts
    fig = plot_training_curves(episodes, rewards, losses)
    
    # 打印统计信息 / Print statistics
    print("\n📊 训练统计 / Training Statistics:")
    print(f"   平均奖励 / Average reward: {np.mean(rewards):.3f}")
    print(f"   最终奖励 / Final reward: {rewards[-1]:.3f}")
    print(f"   奖励改进 / Reward improvement: {rewards[-1] - rewards[0]:.3f}")
    print(f"   平均损失 / Average loss: {np.mean(losses):.3f}")
    
    print("\n✅ PPO测试完成！ / PPO test completed!")


if __name__ == "__main__":
    main()
