#!/usr/bin/env python3
"""
简化PPO多轮训练验证
Simplified PPO Multi-Round Training Validation
"""
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 确保使用测试模式
os.environ['PIPELINE_TEST'] = '1'
sys.path.append('.')

def run_ppo_multiple_rounds():
    """运行多轮PPO训练"""
    print("🚀 开始多轮PPO训练验证 / Starting Multi-Round PPO Training")
    print("=" * 70)
    
    rounds_config = [
        {"episodes": 20, "name": "第1轮", "desc": "Round 1"},
        {"episodes": 25, "name": "第2轮", "desc": "Round 2"},
        {"episodes": 30, "name": "第3轮", "desc": "Round 3"}
    ]
    
    all_rewards = []
    round_summaries = []
    
    for i, config in enumerate(rounds_config):
        print(f"\n🔄 {config['name']} ({config['desc']}) - {config['episodes']} 个回合")
        print("-" * 50)
        
        try:
            # 运行训练
            result = subprocess.run([
                "D:\\conda_envs\\summer_project_2025\\python.exe",
                "train_ppo_safe.py", 
                "--episodes", str(config['episodes'])
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # 从输出中提取奖励
                rewards = extract_rewards_from_output(result.stdout)
                
                if rewards:
                    round_summary = {
                        'round': i + 1,
                        'name': config['name'],
                        'rewards': rewards,
                        'avg': np.mean(rewards),
                        'std': np.std(rewards),
                        'max': np.max(rewards),
                        'min': np.min(rewards)
                    }
                    round_summaries.append(round_summary)
                    all_rewards.extend(rewards)
                    
                    print(f"✅ {config['name']}完成:")
                    print(f"   📊 实际回合数: {len(rewards)}")
                    print(f"   📈 平均奖励: {round_summary['avg']:.3f} ± {round_summary['std']:.3f}")
                    print(f"   📊 奖励范围: {round_summary['min']:.3f} ~ {round_summary['max']:.3f}")
                else:
                    print(f"⚠️ {config['name']}未能提取奖励数据")
            else:
                print(f"❌ {config['name']}失败: {result.stderr[:200]}")
                
        except Exception as e:
            print(f"❌ {config['name']}出错: {e}")
    
    return round_summaries, all_rewards

def extract_rewards_from_output(output):
    """从训练输出中提取奖励值"""
    rewards = []
    lines = output.split('\\n')
    
    for line in lines:
        if "奖励:" in line and "长度:" in line:
            try:
                # 提取 "奖励: -1.000" 这样的格式
                reward_part = line.split("奖励:")[1].split(",")[0].strip()
                reward = float(reward_part)
                rewards.append(reward)
            except (ValueError, IndexError):
                continue
    
    return rewards

def create_training_visualization(round_summaries, all_rewards):
    """创建训练可视化"""
    if not round_summaries:
        print("❌ 没有数据可视化")
        return None
    
    print("\\n📊 创建学习曲线可视化...")
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 各轮奖励时间序列
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    episode_counter = 0
    
    for i, summary in enumerate(round_summaries):
        episodes = range(episode_counter, episode_counter + len(summary['rewards']))
        ax1.plot(episodes, summary['rewards'], 'o-', 
                alpha=0.7, color=colors[i % len(colors)], 
                label=summary['name'])
        episode_counter += len(summary['rewards'])
    
    # 添加移动平均
    if len(all_rewards) >= 8:
        window = 8
        moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        moving_episodes = range(window-1, len(all_rewards))
        ax1.plot(moving_episodes, moving_avg, 'k-', linewidth=3, 
                label=f'{window}回合移动平均', alpha=0.8)
    
    ax1.set_xlabel('累积回合数')
    ax1.set_ylabel('奖励')
    ax1.set_title('PPO多轮学习曲线\\nPPO Multi-Round Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 各轮平均奖励对比
    round_nums = [s['round'] for s in round_summaries]
    avg_rewards = [s['avg'] for s in round_summaries]
    std_rewards = [s['std'] for s in round_summaries]
    
    bars = ax2.bar(round_nums, avg_rewards, yerr=std_rewards, 
                  capsize=5, alpha=0.7, color=colors[:len(round_summaries)])
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('平均奖励')
    ax2.set_title('各轮平均奖励对比\\nAverage Reward Comparison')
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(round_nums, avg_rewards)):
        ax2.text(x, y + std_rewards[i] + 0.02, f'{y:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励分布
    ax3.hist(all_rewards, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.axvline(np.mean(all_rewards), color='red', linestyle='--', 
               label=f'平均值: {np.mean(all_rewards):.3f}')
    ax3.set_xlabel('奖励值')
    ax3.set_ylabel('频次')
    ax3.set_title('奖励分布\\nReward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习改进趋势
    if len(round_summaries) > 1:
        improvements = []
        for i in range(1, len(round_summaries)):
            improvement = round_summaries[i]['avg'] - round_summaries[i-1]['avg']
            improvements.append(improvement)
        
        ax4.plot(range(2, len(round_summaries)+1), improvements, 'o-', 
                linewidth=2, markersize=8, color='green')
        ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('轮次')
        ax4.set_ylabel('奖励改进')
        ax4.set_title('轮次间改进趋势\\nImprovement Trend')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '需要更多轮次\\n显示改进趋势', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('改进趋势\\nImprovement Trend')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/ppo_multi_round_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 多轮学习分析图表已保存: {filename}")
    
    return filename

def analyze_multi_round_results(round_summaries, all_rewards):
    """分析多轮训练结果"""
    print("\\n" + "="*70)
    print("📊 PPO多轮训练结果分析 / Multi-Round Training Analysis")
    print("="*70)
    
    if not round_summaries:
        print("❌ 没有训练结果")
        return
    
    # 总体统计
    total_episodes = sum(len(s['rewards']) for s in round_summaries)
    overall_avg = np.mean(all_rewards)
    overall_std = np.std(all_rewards)
    overall_max = np.max(all_rewards)
    overall_min = np.min(all_rewards)
    
    print(f"\\n🎯 总体表现 / Overall Performance:")
    print(f"  训练轮数: {len(round_summaries)} 轮")
    print(f"  总回合数: {total_episodes} 个")
    print(f"  平均奖励: {overall_avg:.3f} ± {overall_std:.3f}")
    print(f"  最佳奖励: {overall_max:.3f}")
    print(f"  最差奖励: {overall_min:.3f}")
    print(f"  奖励范围: {overall_max - overall_min:.3f}")
    
    # 各轮对比
    print(f"\\n📈 各轮详细对比 / Round-by-Round Details:")
    for i, summary in enumerate(round_summaries):
        improvement = ""
        if i > 0:
            prev_avg = round_summaries[i-1]['avg']
            change = summary['avg'] - prev_avg
            improvement = f" ({change:+.3f})"
        
        print(f"  {summary['name']}: {summary['avg']:.3f} ± {summary['std']:.3f}{improvement}")
        print(f"    └─ 范围: {summary['min']:.3f} ~ {summary['max']:.3f}, 回合数: {len(summary['rewards'])}")
    
    # 学习趋势
    if len(round_summaries) >= 2:
        first_avg = round_summaries[0]['avg']
        last_avg = round_summaries[-1]['avg']
        total_improvement = last_avg - first_avg
        improvement_pct = (total_improvement / abs(first_avg)) * 100 if first_avg != 0 else 0
        
        print(f"\\n🚀 学习趋势分析 / Learning Trend Analysis:")
        print(f"  首轮平均: {first_avg:.3f}")
        print(f"  末轮平均: {last_avg:.3f}")
        print(f"  总体改进: {total_improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        if total_improvement > 0.15:
            print("  ✅ 显著学习改进! / Significant improvement!")
            assessment = "excellent"
        elif total_improvement > 0.05:
            print("  ⚡ 轻微学习改进 / Slight improvement")
            assessment = "good"
        elif total_improvement > -0.05:
            print("  ➖ 基本稳定 / Relatively stable")
            assessment = "stable"
        else:
            print("  ⚠️ 性能下降 / Performance decline")
            assessment = "concerning"
    else:
        assessment = "insufficient_data"
    
    # 稳定性评估
    avg_stability = np.mean([1.0 / (1.0 + s['std']) for s in round_summaries])
    print(f"\\n🎲 训练稳定性 / Training Stability:")
    print(f"  平均稳定性评分: {avg_stability:.3f} (0-1, 越高越稳定)")
    
    if avg_stability > 0.7:
        print("  ✅ 训练非常稳定 / Very stable training")
        stability = "high"
    elif avg_stability > 0.5:
        print("  ⚡ 训练较稳定 / Moderately stable")
        stability = "medium"
    else:
        print("  ⚠️ 训练不稳定 / Unstable training")
        stability = "low"
    
    # 综合评估和建议
    print(f"\\n💡 综合评估与建议 / Assessment & Recommendations:")
    
    if assessment == "excellent":
        print("  🌟 PPO学习效果优秀!")
        print("  🌟 Excellent PPO learning performance!")
        print("  📈 建议继续扩展训练回合数以进一步提升")
    elif assessment == "good":
        print("  ✅ PPO学习效果良好")
        print("  ✅ Good PPO learning performance")
        print("  🔧 可考虑微调超参数优化")
    elif assessment == "stable":
        print("  ➖ PPO学习效果稳定")
        print("  ➖ Stable PPO learning performance")
        print("  🔧 建议调整学习率或奖励机制")
    else:
        print("  ⚠️ PPO学习效果需要改进")
        print("  ⚠️ PPO learning needs improvement")
        print("  🔧 建议检查环境设计和奖励函数")
    
    if stability == "low":
        print("  📊 建议增加训练稳定性措施")
        print("  📊 Consider adding stability measures")

if __name__ == "__main__":
    try:
        print("🎯 PPO多轮训练验证开始 / Multi-Round PPO Training Validation")
        
        # 运行多轮训练
        round_summaries, all_rewards = run_ppo_multiple_rounds()
        
        if round_summaries and all_rewards:
            # 创建可视化
            chart_file = create_training_visualization(round_summaries, all_rewards)
            
            # 分析结果
            analyze_multi_round_results(round_summaries, all_rewards)
            
            print(f"\\n🎉 多轮PPO训练验证完成! / Multi-Round Training Complete!")
            if chart_file:
                print(f"📈 分析图表: {chart_file}")
            print(f"📁 请查看 logs/ 目录中的图表文件")
            
        else:
            print("❌ 没有收集到训练数据 / No training data collected")
            
    except Exception as e:
        print(f"❌ 训练验证出错: {e}")
        import traceback
        traceback.print_exc()
