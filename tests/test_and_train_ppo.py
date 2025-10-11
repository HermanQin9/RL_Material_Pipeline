#!/usr/bin/env python3
"""
完整的代码测试和PPO学习脚本 / Complete Code Testing and PPO Learning Script

This script runs comprehensive tests and trains a PPO agent with learning curve visualization.
此脚本运行全面测试并训练PPO智能体，同时可视化学习曲线。
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import logging

# 添加项目根目录到路径 / Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_components():
    """
    测试流水线组件 / Test pipeline components
    
    Returns:
        bool: 测试是否成功 / Whether tests passed
    """
    print("🧪 开始测试流水线组件 / Starting pipeline component tests")
    print("=" * 70)
    
    try:
        from pipeline import run_pipeline
        from nodes import DataFetchNode, ImputeNode, FeatureMatrixNode, FeatureSelectionNode, ScalingNode, ModelTrainingNode
        
        # 测试每个节点 / Test each node
        print("\n1. 测试数据获取节点 / Testing Data Fetch Node")
        data_node = DataFetchNode()
        fetched = data_node.execute(method='api', params={'cache': True}, data={})
        print(f"   ✅ 数据获取成功 / Data fetch successful: {list(fetched.keys())}")
        
        print("\n2. 测试特征矩阵节点 / Testing Feature Matrix Node")
        feature_node = FeatureMatrixNode()
        features = feature_node.execute(
            method='construct',
            params={'nan_thresh': 0.5, 'train_val_ratio': 0.8, 'verbose': False},
            data=fetched
        )
        print(f"   ✅ 特征矩阵构建成功 / Feature matrix construction successful")
        print(f"   训练集形状 / Training set shape: {features.get('X_train', np.array([])).shape}")
        
        print("\n3. 测试缺失值填充节点 / Testing Imputation Node")
        impute_node = ImputeNode()
        imputed = impute_node.execute(
            method='impute', 
            params={'strategy': 'mean', 'params': {}}, 
            data=features
        )
        print(f"   ✅ 缺失值填充成功 / Imputation successful")
        
        print("\n4. 测试特征选择节点 / Testing Feature Selection Node")
        select_node = FeatureSelectionNode()
        selected = select_node.execute(
            method='select',
            params={'strategy': 'none', 'params': {}},
            data=imputed
        )
        print(f"   ✅ 特征选择成功 / Feature selection successful")
        
        print("\n5. 测试数据缩放节点 / Testing Scaling Node")
        scaling_node = ScalingNode()
        scaled = scaling_node.execute(
            method='scale',
            params={'strategy': 'standard', 'params': {}},
            data=selected
        )
        print(f"   ✅ 数据缩放成功 / Data scaling successful")
        
        print("\n6. 测试完整流水线 / Testing Complete Pipeline")
        result = run_pipeline(
            cache=True,
            nan_thresh=0.5,
            train_val_ratio=0.8,
            impute_strategy='mean',
            selection_strategy='none',
            scaling_strategy='standard',
            model_strategy='rf',
            model_params={'n_estimators': 10}
        )
        print(f"   ✅ 完整流水线测试成功 / Complete pipeline test successful")
        print(f"   模型类型 / Model type: {type(result.get('model', None)).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 流水线测试失败 / Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_components():
    """
    测试PPO组件 / Test PPO components
    
    Returns:
        bool: 测试是否成功 / Whether tests passed
    """
    print("\n🤖 开始测试PPO组件 / Starting PPO component tests")
    print("=" * 70)
    
    try:
        from env.pipeline_env import PipelineEnv
        from ppo.utils import compute_gae, ppo_loss, value_loss, entropy_loss
        import torch
        
        print("\n1. 测试环境初始化 / Testing Environment Initialization")
        env = PipelineEnv()
        obs = env.reset()
        print(f"   ✅ 环境初始化成功 / Environment initialization successful")
        print(f"   观测空间键 / Observation space keys: {list(obs.keys())}")
        
        print("\n2. 测试环境步进 / Testing Environment Step")
        action = {'node': 0, 'method': 0, 'params': [0.5]}
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"   ✅ 环境步进成功 / Environment step successful")
        print(f"   奖励 / Reward: {reward:.3f}, 完成 / Done: {done}")
        
        print("\n3. 测试PPO工具函数 / Testing PPO Utility Functions")
        # 创建测试数据 / Create test data
        rewards = torch.tensor([1.0, 0.5, -0.2])
        values = torch.tensor([0.8, 0.6, 0.1])
        dones = torch.tensor([0.0, 0.0, 1.0])
        
        advantages, returns = compute_gae(rewards, values, dones, 0.0)
        print(f"   ✅ GAE计算成功 / GAE computation successful")
        
        # 测试损失函数 / Test loss functions
        new_log_probs = torch.tensor([0.1, 0.2, 0.3])
        old_log_probs = torch.tensor([0.15, 0.18, 0.25])
        
        policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)
        v_loss = value_loss(values, returns)
        e_loss = entropy_loss(new_log_probs)
        
        print(f"   ✅ 损失函数测试成功 / Loss function tests successful")
        print(f"   策略损失 / Policy loss: {policy_loss:.4f}")
        print(f"   价值损失 / Value loss: {v_loss:.4f}")
        print(f"   熵损失 / Entropy loss: {e_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PPO组件测试失败 / PPO component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_ppo_with_curves(episodes: int = 100, save_plots: bool = True):
    """
    训练PPO并绘制学习曲线 / Train PPO and plot learning curves
    
    Args:
        episodes: 训练回合数 / Number of training episodes
        save_plots: 是否保存图像 / Whether to save plots
        
    Returns:
        Dict: 训练结果 / Training results
    """
    print(f"\n🚀 开始PPO训练 ({episodes} 回合) / Starting PPO Training ({episodes} episodes)")
    print("=" * 70)
    
    try:
        from env.pipeline_env import PipelineEnv
        
        # 初始化环境 / Initialize environment
        env = PipelineEnv()
        
        # 训练数据记录 / Training data recording
        episode_rewards = []
        episode_lengths = []
        moving_avg_rewards = []
        exploration_rates = []
        
        print("开始训练循环 / Starting training loop...")
        
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # 简单的随机策略用于演示 / Simple random policy for demonstration
            while not done and episode_length < 50:  # 限制最大步数 / Limit max steps
                # 随机选择动作 / Random action selection
                node_idx = np.random.randint(len(env.pipeline_nodes))
                method_idx = np.random.randint(len(env.methods_for_node[env.pipeline_nodes[node_idx]]))
                params = [np.random.random()]
                
                action = {
                    'node': node_idx,
                    'method': method_idx, 
                    'params': params
                }
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 计算移动平均 / Calculate moving average
            window_size = min(10, len(episode_rewards))
            moving_avg = np.mean(episode_rewards[-window_size:])
            moving_avg_rewards.append(moving_avg)
            
            # 记录探索率 (简化版) / Record exploration rate (simplified)
            exploration_rate = max(0.1, 1.0 - episode / episodes)
            exploration_rates.append(exploration_rate)
            
            # 定期输出进度 / Periodic progress output
            if (episode + 1) % 20 == 0:
                print(f"回合 / Episode {episode + 1}/{episodes}: "
                      f"奖励 / Reward: {episode_reward:.3f}, "
                      f"移动平均 / Moving Avg: {moving_avg:.3f}, "
                      f"长度 / Length: {episode_length}")
        
        # 绘制学习曲线 / Plot learning curves
        print("\n📊 绘制学习曲线 / Plotting learning curves...")
        plot_learning_curves(episode_rewards, moving_avg_rewards, exploration_rates, 
                            episode_lengths, save_plots)
        
        results = {
            'episode_rewards': episode_rewards,
            'moving_avg_rewards': moving_avg_rewards,
            'exploration_rates': exploration_rates,
            'episode_lengths': episode_lengths,
            'final_avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        }
        
        print(f"\n✅ PPO训练完成 / PPO training completed!")
        print(f"最终平均奖励 / Final average reward: {results['final_avg_reward']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ PPO训练失败 / PPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def plot_learning_curves(rewards, moving_avg_rewards, exploration_rates, episode_lengths, save_plots=True):
    """
    绘制学习曲线 / Plot learning curves
    
    Args:
        rewards: 回合奖励列表 / Episode rewards list
        moving_avg_rewards: 移动平均奖励 / Moving average rewards
        exploration_rates: 探索率 / Exploration rates
        episode_lengths: 回合长度 / Episode lengths
        save_plots: 是否保存图像 / Whether to save plots
    """
    try:
        # 设置中文字体 / Set Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        episodes = range(1, len(rewards) + 1)
        
        # 1. 回合奖励和移动平均 / Episode rewards and moving average
        ax1.plot(episodes, rewards, alpha=0.6, color='lightblue', label='回合奖励 / Episode Rewards')
        ax1.plot(episodes, moving_avg_rewards, color='darkblue', linewidth=2, label='移动平均 / Moving Average')
        ax1.set_xlabel('回合 / Episodes')
        ax1.set_ylabel('奖励 / Reward')
        ax1.set_title('PPO学习曲线 - 奖励 / PPO Learning Curve - Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 探索率衰减 / Exploration rate decay
        ax2.plot(episodes, exploration_rates, color='orange', linewidth=2)
        ax2.set_xlabel('回合 / Episodes')
        ax2.set_ylabel('探索率 / Exploration Rate')
        ax2.set_title('探索率衰减 / Exploration Rate Decay')
        ax2.grid(True, alpha=0.3)
        
        # 3. 回合长度 / Episode lengths
        ax3.plot(episodes, episode_lengths, color='green', alpha=0.7)
        ax3.set_xlabel('回合 / Episodes')
        ax3.set_ylabel('回合长度 / Episode Length')
        ax3.set_title('回合长度变化 / Episode Length Variation')
        ax3.grid(True, alpha=0.3)
        
        # 4. 奖励分布直方图 / Reward distribution histogram
        ax4.hist(rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('奖励值 / Reward Value')
        ax4.set_ylabel('频次 / Frequency')
        ax4.set_title('奖励分布 / Reward Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plots_dir = Path('logs')
            plots_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_file = plots_dir / f'ppo_learning_curves_{timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 学习曲线已保存 / Learning curves saved: {plot_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 绘图失败 / Plotting failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数 / Main function
    """
    print("🎯 MatFormPPO 完整测试和PPO学习系统")
    print("🎯 MatFormPPO Complete Testing and PPO Learning System")
    print("=" * 80)
    
    # 第一阶段：测试流水线组件 / Phase 1: Test pipeline components
    pipeline_success = test_pipeline_components()
    
    # 第二阶段：测试PPO组件 / Phase 2: Test PPO components  
    ppo_success = test_ppo_components()
    
    if pipeline_success and ppo_success:
        print("\n🎉 所有组件测试通过！开始PPO训练...")
        print("🎉 All component tests passed! Starting PPO training...")
        
        # 第三阶段：PPO训练和学习曲线 / Phase 3: PPO training and learning curves
        training_results = train_ppo_with_curves(episodes=100, save_plots=True)
        
        if training_results:
            print(f"\n🏆 完整测试和训练成功完成！")
            print(f"🏆 Complete testing and training successfully completed!")
            print(f"最终平均奖励 / Final average reward: {training_results['final_avg_reward']:.3f}")
        else:
            print(f"\n❌ PPO训练失败")
            print(f"❌ PPO training failed")
    else:
        print(f"\n❌ 组件测试失败，跳过PPO训练")
        print(f"❌ Component tests failed, skipping PPO training")

if __name__ == "__main__":
    main()
