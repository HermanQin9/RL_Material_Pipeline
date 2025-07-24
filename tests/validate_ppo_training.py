#!/usr/bin/env python3
"""
PPO训练模式验证 - 多轮训练并绘制学习曲线
PPO Training Mode Validation - Multiple rounds with learning curves
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 设置训练模式环境变量
os.environ['PIPELINE_TEST'] = '0'

sys.path.append('.')
from ppo.trainer import PPOTrainer
from env.pipeline_env import PipelineEnv

def run_multiple_training_rounds(num_rounds=3, episodes_per_round=30):
    """
    运行多轮PPO训练
    Run multiple rounds of PPO training
    """
    print("🚀 开始PPO训练模式验证 / Starting PPO Training Mode Validation")
    print(f"📊 配置 / Configuration:")
    print(f"  - 训练轮数 / Training Rounds: {num_rounds}")
    print(f"  - 每轮回合数 / Episodes per Round: {episodes_per_round}")
    print(f"  - 总回合数 / Total Episodes: {num_rounds * episodes_per_round}")
    print("=" * 60)
    
    all_rewards = []
    all_episode_numbers = []
    round_summaries = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n🔄 第 {round_num} 轮训练 / Round {round_num} Training")
        print("-" * 40)
        
        # 创建环境和训练器
        env = PipelineEnv()
        trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
        
        round_rewards = []
        round_episodes = []
        
        try:
            # 训练指定回合数
            for episode in range(episodes_per_round):
                obs = env.reset()
                total_reward = 0
                steps = 0
                done = False
                
                while not done and steps < 10:  # 限制最大步数
                    action, _ = trainer.select_action(obs)  # 修正：获取action和log_prob
                    obs, reward, done, _, info = env.step(action)
                    total_reward += reward
                    steps += 1
                
                round_rewards.append(total_reward)
                episode_num = (round_num - 1) * episodes_per_round + episode + 1
                round_episodes.append(episode_num)
                
                # 每5个回合打印一次进度
                if (episode + 1) % 5 == 0:
                    avg_reward = np.mean(round_rewards[-5:])
                    print(f"  回合 {episode + 1:2d}/30: 平均奖励 = {avg_reward:.3f}")
            
            # 记录本轮结果
            round_avg = np.mean(round_rewards)
            round_std = np.std(round_rewards)
            round_max = np.max(round_rewards)
            round_min = np.min(round_rewards)
            
            round_summary = {
                'round': round_num,
                'avg_reward': round_avg,
                'std_reward': round_std,
                'max_reward': round_max,
                'min_reward': round_min,
                'episodes': len(round_rewards)
            }
            round_summaries.append(round_summary)
            
            print(f"\n📈 第 {round_num} 轮结果 / Round {round_num} Results:")
            print(f"  平均奖励 / Average Reward: {round_avg:.3f} ± {round_std:.3f}")
            print(f"  最大奖励 / Max Reward: {round_max:.3f}")
            print(f"  最小奖励 / Min Reward: {round_min:.3f}")
            
            # 累积所有数据
            all_rewards.extend(round_rewards)
            all_episode_numbers.extend(round_episodes)
            
        except Exception as e:
            print(f"❌ 第 {round_num} 轮训练出错: {e}")
            continue
    
    return all_rewards, all_episode_numbers, round_summaries

def plot_learning_curves(rewards, episodes, round_summaries):
    """
    绘制学习曲线
    Plot learning curves
    """
    print("\n📊 绘制学习曲线 / Plotting Learning Curves...")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 原始奖励曲线
    ax1.plot(episodes, rewards, 'b-', alpha=0.6, linewidth=1, label='Episode Rewards')
    
    # 计算移动平均
    window_size = 10
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_episodes = episodes[window_size-1:]
        ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} episodes)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('PPO Learning Curve - Training Mode (4000 samples)\nPPO学习曲线 - 训练模式 (4000样本)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 每轮平均奖励
    if round_summaries:
        round_nums = [r['round'] for r in round_summaries]
        round_avgs = [r['avg_reward'] for r in round_summaries]
        round_stds = [r['std_reward'] for r in round_summaries]
        
        ax2.errorbar(round_nums, round_avgs, yerr=round_stds, 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax2.set_xlabel('Training Round / 训练轮次')
        ax2.set_ylabel('Average Reward / 平均奖励')
        ax2.set_title('Average Reward per Training Round\n每轮训练的平均奖励', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (x, y) in enumerate(zip(round_nums, round_avgs)):
            ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/ppo_training_curves_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 学习曲线已保存: {filename}")
    
    return filename

def analyze_results(rewards, round_summaries):
    """
    分析训练结果
    Analyze training results
    """
    print("\n" + "="*60)
    print("📊 PPO训练结果分析 / PPO Training Results Analysis")
    print("="*60)
    
    # 整体统计
    overall_avg = np.mean(rewards)
    overall_std = np.std(rewards)
    overall_max = np.max(rewards)
    overall_min = np.min(rewards)
    
    print(f"\n🎯 整体性能 / Overall Performance:")
    print(f"  总回合数 / Total Episodes: {len(rewards)}")
    print(f"  平均奖励 / Average Reward: {overall_avg:.3f} ± {overall_std:.3f}")
    print(f"  最佳奖励 / Best Reward: {overall_max:.3f}")
    print(f"  最差奖励 / Worst Reward: {overall_min:.3f}")
    print(f"  奖励范围 / Reward Range: {overall_max - overall_min:.3f}")
    
    # 学习趋势分析
    if len(rewards) >= 20:
        first_half = rewards[:len(rewards)//2]
        second_half = rewards[len(rewards)//2:]
        
        improvement = np.mean(second_half) - np.mean(first_half)
        improvement_pct = (improvement / abs(np.mean(first_half))) * 100 if np.mean(first_half) != 0 else 0
        
        print(f"\n📈 学习趋势 / Learning Trend:")
        print(f"  前半段平均奖励 / First Half Average: {np.mean(first_half):.3f}")
        print(f"  后半段平均奖励 / Second Half Average: {np.mean(second_half):.3f}")
        print(f"  改进幅度 / Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        if improvement > 0.05:
            print("  ✅ 显著学习改进 / Significant learning improvement detected!")
        elif improvement > 0:
            print("  ⚡ 轻微学习改进 / Slight learning improvement detected")
        else:
            print("  ⚠️ 未观察到明显改进 / No significant improvement observed")
    
    # 每轮对比
    if len(round_summaries) > 1:
        print(f"\n🔄 轮次对比 / Round Comparison:")
        for i, summary in enumerate(round_summaries):
            print(f"  第{summary['round']}轮 / Round {summary['round']}: "
                  f"{summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")
    
    # 稳定性分析
    stability = 1.0 / (1.0 + overall_std)  # 简单的稳定性指标
    print(f"\n🎲 训练稳定性 / Training Stability:")
    print(f"  奖励方差 / Reward Variance: {overall_std**2:.3f}")
    print(f"  稳定性评分 / Stability Score: {stability:.3f} (0-1, 越高越稳定)")
    
    if stability > 0.7:
        print("  ✅ 训练稳定 / Training is stable")
    elif stability > 0.5:
        print("  ⚡ 训练较稳定 / Training is moderately stable")
    else:
        print("  ⚠️ 训练不稳定 / Training is unstable")

if __name__ == "__main__":
    try:
        # 运行多轮训练
        rewards, episodes, summaries = run_multiple_training_rounds(num_rounds=3, episodes_per_round=30)
        
        if len(rewards) > 0:
            # 绘制学习曲线
            curve_file = plot_learning_curves(rewards, episodes, summaries)
            
            # 分析结果
            analyze_results(rewards, summaries)
            
            print(f"\n🎉 PPO训练验证完成! / PPO Training Validation Complete!")
            print(f"📈 学习曲线文件: {curve_file}")
            
        else:
            print("❌ 没有收集到训练数据 / No training data collected")
            
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
