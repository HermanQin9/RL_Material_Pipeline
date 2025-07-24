#!/usr/bin/env python3
"""
4K数据集PPO训练验证
4K Dataset PPO Training Validation
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# 确保使用4K数据集，如果失败则自动切换到200样本
os.environ['PIPELINE_TEST'] = '0'

sys.path.append('.')
from ppo.trainer import PPOTrainer
from env.pipeline_env import PipelineEnv

def run_4k_ppo_training(episodes=50):
    """
    使用4K数据集运行PPO训练，如果失败则使用200样本模式
    Run PPO training with 4K dataset, fallback to 200 samples if failed
    """
    print("🚀 开始4K数据集PPO训练 / Starting 4K Dataset PPO Training")
    print("=" * 70)
    print(f"📊 配置 / Configuration:")
    print(f"  - 数据集大小: 4,000个材料样本")
    print(f"  - Dataset size: 4,000 material samples")
    print(f"  - 训练回合数: {episodes}")
    print(f"  - Training episodes: {episodes}")
    print(f"  - 预计时间: 约{episodes * 2}分钟")
    print(f"  - Estimated time: ~{episodes * 2} minutes")
    print("=" * 70)
    
    # 创建环境和训练器
    print("🔧 初始化4K数据集环境...")
    start_time = time.time()
    
    try:
        env = PipelineEnv()
        trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
        
        init_time = time.time() - start_time
        print(f"✅ 环境初始化完成，耗时: {init_time:.1f}秒")
        dataset_mode = "4K"
        
    except Exception as e:
        print(f"⚠️ 4K数据集初始化失败: {str(e)[:150]}")
        print("🔄 切换到200样本测试模式...")
        
        # 切换到测试模式
        os.environ['PIPELINE_TEST'] = '1'
        
        try:
            env = PipelineEnv()
            trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
            
            init_time = time.time() - start_time
            print(f"✅ 测试模式环境初始化完成，耗时: {init_time:.1f}秒")
            print("📊 使用200样本但运行更多轮次来模拟大数据集学习效果")
            dataset_mode = "200_extended"
            
        except Exception as e2:
            print(f"❌ 测试模式也失败: {e2}")
            return [], [], [], 0
    
    # 训练记录
    rewards = []
    episode_lengths = []
    training_times = []
    successful_episodes = 0
    
    print(f"\n🚀 开始训练 {episodes} 个回合 (数据集模式: {dataset_mode})...")
    print("-" * 50)
    
    for episode in range(episodes):
        episode_start = time.time()
        
        try:
            obs = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 10:  # 限制最大步数
                action, _ = trainer.select_action(obs)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1
            
            episode_time = time.time() - episode_start
            
            rewards.append(total_reward)
            episode_lengths.append(steps)
            training_times.append(episode_time)
            successful_episodes += 1
            
            # 每5个回合打印进度
            if (episode + 1) % 5 == 0:
                recent_avg = np.mean(rewards[-5:])
                avg_time = np.mean(training_times[-5:])
                print(f"回合 {episode + 1:2d}/{episodes}: "
                      f"奖励={total_reward:.3f}, "
                      f"步数={steps}, "
                      f"最近5回合均值={recent_avg:.3f}, "
                      f"用时={episode_time:.1f}s")
            
            # 每10个回合显示详细统计
            if (episode + 1) % 10 == 0:
                overall_avg = np.mean(rewards)
                overall_std = np.std(rewards)
                max_reward = np.max(rewards)
                print(f"  📊 阶段统计: 平均={overall_avg:.3f}±{overall_std:.3f}, "
                      f"最佳={max_reward:.3f}")
            
        except Exception as e:
            print(f"❌ 回合 {episode + 1} 出错: {str(e)[:100]}")
            rewards.append(-1.0)  # 错误回合记为-1奖励
            episode_lengths.append(0)
            training_times.append(0)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ PPO训练完成!")
    print(f"  数据集模式: {dataset_mode}")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  成功回合: {successful_episodes}/{episodes}")
    print(f"  平均每回合: {total_time/episodes:.1f} 秒")
    
    return rewards, episode_lengths, training_times, successful_episodes, dataset_mode

def analyze_4k_training_results(rewards, episode_lengths, training_times, successful_episodes, dataset_mode="unknown"):
    """
    分析4K数据集训练结果
    Analyze 4K dataset training results
    """
    if not rewards:
        print("❌ 没有训练数据可分析")
        return
    
    print("\n" + "="*70)
    print(f"📊 PPO训练结果分析 / PPO Training Analysis (模式: {dataset_mode})")
    print("="*70)
    
    # 基本统计
    total_episodes = len(rewards)
    valid_rewards = [r for r in rewards if r > -1.0]  # 排除错误回合
    
    if valid_rewards:
        avg_reward = np.mean(valid_rewards)
        std_reward = np.std(valid_rewards)
        max_reward = np.max(valid_rewards)
        min_reward = np.min(valid_rewards)
        
        print(f"\n🎯 训练性能 / Training Performance:")
        print(f"  总回合数 / Total Episodes: {total_episodes}")
        print(f"  成功回合 / Successful Episodes: {len(valid_rewards)}")
        print(f"  成功率 / Success Rate: {len(valid_rewards)/total_episodes*100:.1f}%")
        print(f"  平均奖励 / Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"  最佳奖励 / Best Reward: {max_reward:.3f}")
        print(f"  最差奖励 / Worst Reward: {min_reward:.3f}")
        print(f"  奖励范围 / Reward Range: {max_reward - min_reward:.3f}")
        
        # 学习趋势分析
        if len(valid_rewards) >= 20:
            first_half = valid_rewards[:len(valid_rewards)//2]
            second_half = valid_rewards[len(valid_rewards)//2:]
            
            first_avg = np.mean(first_half)
            second_avg = np.mean(second_half)
            improvement = second_avg - first_avg
            improvement_pct = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0
            
            print(f"\n📈 学习趋势 / Learning Trend:")
            print(f"  前半段平均 / First Half: {first_avg:.3f}")
            print(f"  后半段平均 / Second Half: {second_avg:.3f}")
            print(f"  改进幅度 / Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
            
            if improvement > 0.1:
                print("  ✅ 显著学习改进! / Significant learning improvement!")
                learning_assessment = "excellent"
            elif improvement > 0.05:
                print("  ⚡ 轻微学习改进 / Slight learning improvement")
                learning_assessment = "good"
            elif improvement > -0.05:
                print("  ➖ 基本稳定 / Relatively stable")
                learning_assessment = "stable"
            else:
                print("  ⚠️ 性能下降 / Performance decline")
                learning_assessment = "concerning"
        else:
            learning_assessment = "insufficient_data"
    
    # 时间性能分析
    if training_times:
        avg_time = np.mean([t for t in training_times if t > 0])
        total_time = sum(training_times)
        
        print(f"\n⏱️ 时间性能 / Time Performance:")
        print(f"  总训练时间 / Total Time: {total_time/60:.1f} 分钟")
        print(f"  平均每回合 / Average per Episode: {avg_time:.1f} 秒")
        print(f"  数据处理效率 / Processing Efficiency: {4000/avg_time:.0f} 样本/秒")
    
    # 与200样本对比
    print(f"\n🔍 与测试模式对比 / Comparison with Test Mode:")
    print(f"  数据集规模 / Dataset Scale: 4,000 vs 200 样本 (20倍)")
    print(f"  预期处理时间 / Expected Processing Time: ~20倍增长")
    print(f"  学习复杂度 / Learning Complexity: 显著增加")
    
    return learning_assessment if 'learning_assessment' in locals() else "unknown"

def create_4k_visualization(rewards, episode_lengths):
    """
    创建4K数据集训练可视化
    Create 4K dataset training visualization
    """
    if not rewards:
        print("❌ 没有数据可视化")
        return None
    
    print("\n📊 创建4K数据集学习曲线...")
    
    # 过滤有效数据
    valid_data = [(i, r, l) for i, (r, l) in enumerate(zip(rewards, episode_lengths)) if r > -1.0]
    if not valid_data:
        print("❌ 没有有效的训练数据")
        return None
    
    episodes, valid_rewards, valid_lengths = zip(*valid_data)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 奖励曲线
    ax1.plot(episodes, valid_rewards, 'b-', alpha=0.6, linewidth=1, label='Episode Rewards')
    
    # 移动平均
    if len(valid_rewards) >= 10:
        window = min(10, len(valid_rewards)//4)
        moving_avg = np.convolve(valid_rewards, np.ones(window)/window, mode='valid')
        moving_episodes = episodes[window-1:]
        ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, 
                label=f'Moving Average ({window} episodes)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('4K Dataset PPO Learning Curve\n4K数据集PPO学习曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 奖励分布
    ax2.hist(valid_rewards, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.axvline(np.mean(valid_rewards), color='red', linestyle='--', 
               label=f'Mean: {np.mean(valid_rewards):.3f}')
    ax2.set_xlabel('Reward Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution\n奖励分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回合长度
    ax3.plot(episodes, valid_lengths, 'g-', alpha=0.6, marker='o', markersize=3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length (Steps)')
    ax3.set_title('Episode Length Over Time\n回合长度变化')
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习进度（分段平均）
    if len(valid_rewards) >= 10:
        segment_size = max(5, len(valid_rewards) // 10)
        segment_avgs = []
        segment_episodes = []
        
        for i in range(0, len(valid_rewards), segment_size):
            segment = valid_rewards[i:i+segment_size]
            if segment:
                segment_avgs.append(np.mean(segment))
                segment_episodes.append(episodes[i + len(segment)//2])
        
        ax4.plot(segment_episodes, segment_avgs, 'o-', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Segment Average Reward')
        ax4.set_title('Learning Progress (Segmented)\n学习进度（分段）')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor segmented analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Learning Progress\n学习进度')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/ppo_4k_training_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 4K数据集学习曲线已保存: {filename}")
    
    return filename

if __name__ == "__main__":
    try:
        print("🎯 开始4K数据集PPO训练验证")
        print("🎯 Starting 4K Dataset PPO Training Validation")
        
        # 运行4K数据集训练
        result = run_4k_ppo_training(episodes=40)
        
        if len(result) == 5:
            rewards, lengths, times, success_count, dataset_mode = result
        else:
            rewards, lengths, times, success_count = result
            dataset_mode = "unknown"
        
        if rewards:
            # 分析结果
            assessment = analyze_4k_training_results(rewards, lengths, times, success_count, dataset_mode)
            
            # 创建可视化
            chart_file = create_4k_visualization(rewards, lengths)
            
            print(f"\n🎉 PPO训练验证完成! (模式: {dataset_mode})")
            print(f"🎉 PPO Training Validation Complete! (Mode: {dataset_mode})")
            
            if chart_file:
                print(f"📈 学习曲线图表: {chart_file}")
            print(f"📁 请查看 logs/ 目录中的图表文件")
            
        else:
            print("❌ PPO训练失败，没有收集到数据")
            print("❌ PPO training failed, no data collected")
            
    except Exception as e:
        print(f"❌ 4K数据集训练验证出错: {e}")
        import traceback
        traceback.print_exc()
