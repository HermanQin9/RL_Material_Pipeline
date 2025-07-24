#!/usr/bin/env python3
"""
扩展PPO训练验证 - 多轮训练分析
Extended PPO Training Validation - Multi-round Training Analysis
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei',        if results:
            # 创建综合分析
            analysis_result = create_comprehensive_analysis(results)
            if analysis_result:
                chart_file, all_rewards, all_episodes = analysis_result
                
                # 打印分析报告
                print_comprehensive_analysis(results, all_rewards)
                
                print(f"\n🎉 PPO扩展训练验证完成! / Extended PPO Training Validation Complete!")
                print(f"📈 综合分析图表: {chart_file}")
                print(f"📁 请查看 logs/ 目录中的详细图表文件")
            else:
                print("⚠️ 图表生成失败，但训练数据可用")
                all_rewards = []
                for r in results:
                    all_rewards.extend(r['rewards'])
                print_comprehensive_analysis(results, all_rewards)ans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 确保使用测试模式
os.environ['PIPELINE_TEST'] = '1'

sys.path.append('.')

def run_extended_ppo_training():
    """运行扩展的PPO训练验证"""
    print("🚀 开始扩展PPO训练验证 / Starting Extended PPO Training Validation")
    print("=" * 80)
    
    # 运行多轮不同配置的训练
    training_configs = [
        {"episodes": 20, "name": "第1轮训练", "description": "Round 1"},
        {"episodes": 25, "name": "第2轮训练", "description": "Round 2"}, 
        {"episodes": 30, "name": "第3轮训练", "description": "Round 3"},
        {"episodes": 35, "name": "第4轮训练", "description": "Round 4"}
    ]
    
    all_results = []
    cumulative_episodes = 0
    
    for round_idx, config in enumerate(training_configs, 1):
        print(f"\n🔄 {config['name']} / {config['description']}")
        print(f"📊 回合数: {config['episodes']}")
        print("-" * 60)
        
        # 运行训练
        try:
            import subprocess
            result = subprocess.run([
                "D:\\conda_envs\\summer_project_2025\\python.exe", 
                "train_ppo_safe.py", 
                "--episodes", str(config['episodes'])
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # 解析输出获取奖励信息
                output_lines = result.stdout.split('\\n')
                rewards = []
                
                for line in output_lines:
                    if "奖励:" in line and "长度:" in line:
                        try:
                            reward_str = line.split("奖励:")[1].split(",")[0].strip()
                            reward = float(reward_str)
                            rewards.append(reward)
                        except:
                            continue
                
                if rewards:
                    round_stats = {
                        'round': round_idx,
                        'episodes': config['episodes'],
                        'rewards': rewards,
                        'avg_reward': np.mean(rewards),
                        'std_reward': np.std(rewards),
                        'max_reward': np.max(rewards),
                        'min_reward': np.min(rewards),
                        'cumulative_start': cumulative_episodes,
                        'cumulative_end': cumulative_episodes + len(rewards)
                    }
                    all_results.append(round_stats)
                    cumulative_episodes += len(rewards)
                    
                    print(f"✅ {config['name']}完成:")
                    print(f"   📈 平均奖励: {round_stats['avg_reward']:.3f} ± {round_stats['std_reward']:.3f}")
                    print(f"   📊 奖励范围: {round_stats['min_reward']:.3f} ~ {round_stats['max_reward']:.3f}")
                    print(f"   🎯 实际回合数: {len(rewards)}")
                else:
                    print(f"⚠️ {config['name']}未获取到奖励数据")
            else:
                print(f"❌ {config['name']}训练失败: {result.stderr}")
                
        except Exception as e:
            print(f"❌ {config['name']}运行出错: {e}")
    
    return all_results

def create_comprehensive_analysis(results):
    """创建综合分析图表"""
    if not results:
        print("❌ 没有训练结果可分析")
        return
    
    print("\\n📊 绘制综合学习曲线...")
    
    # 创建综合图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 所有奖励的时间序列图
    ax1 = plt.subplot(2, 3, 1)
    all_rewards = []
    all_episodes = []
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, result in enumerate(results):
        episode_range = range(result['cumulative_start'], result['cumulative_end'])
        all_rewards.extend(result['rewards'])
        all_episodes.extend(episode_range)
        
        plt.plot(episode_range, result['rewards'], 'o-', 
                alpha=0.7, color=colors[i % len(colors)], 
                label=f"第{result['round']}轮")
    
    # 添加总体移动平均
    if len(all_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        moving_episodes = all_episodes[window-1:]
        plt.plot(moving_episodes, moving_avg, 'k-', linewidth=3, 
                label=f'{window}回合移动平均', alpha=0.8)
    
    plt.xlabel('累积回合数 / Cumulative Episodes')
    plt.ylabel('奖励 / Reward')
    plt.title('PPO学习曲线 - 多轮训练\\nPPO Learning Curve - Multi-Round Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 每轮平均奖励对比
    ax2 = plt.subplot(2, 3, 2)
    round_nums = [r['round'] for r in results]
    avg_rewards = [r['avg_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    
    bars = plt.bar(round_nums, avg_rewards, yerr=std_rewards, 
                  capsize=5, alpha=0.7, color=colors[:len(results)])
    plt.xlabel('训练轮次 / Training Round')
    plt.ylabel('平均奖励 / Average Reward')
    plt.title('各轮平均奖励对比\\nAverage Reward per Round')
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(round_nums, avg_rewards)):
        plt.text(x, y + std_rewards[i] + 0.02, f'{y:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # 3. 奖励分布直方图
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(all_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('奖励值 / Reward Value')
    plt.ylabel('频次 / Frequency')
    plt.title('奖励分布直方图\\nReward Distribution')
    plt.axvline(float(np.mean(all_rewards)), color='red', linestyle='--', 
               label=f'平均值: {np.mean(all_rewards):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 学习趋势分析
    ax4 = plt.subplot(2, 3, 4)
    round_improvements = []
    for i in range(1, len(results)):
        improvement = results[i]['avg_reward'] - results[i-1]['avg_reward']
        round_improvements.append(improvement)
    
    if round_improvements:
        plt.plot(range(2, len(results)+1), round_improvements, 'o-', 
                linewidth=2, markersize=8, color='green')
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('轮次 / Round')
        plt.ylabel('奖励改进 / Reward Improvement')
        plt.title('轮次间改进趋势\\nImprovement Trend Between Rounds')
        plt.grid(True, alpha=0.3)
    
    # 5. 稳定性分析
    ax5 = plt.subplot(2, 3, 5)
    stability_scores = [1.0 / (1.0 + r['std_reward']) for r in results]
    plt.plot(round_nums, stability_scores, 'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('训练轮次 / Training Round')
    plt.ylabel('稳定性评分 / Stability Score')
    plt.title('训练稳定性趋势\\nTraining Stability Trend')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 6. 累积性能指标
    ax6 = plt.subplot(2, 3, 6)
    cumulative_avg = []
    cumulative_rewards = []
    
    for result in results:
        cumulative_rewards.extend(result['rewards'])
        cumulative_avg.append(np.mean(cumulative_rewards))
    
    plt.plot(round_nums, cumulative_avg, 'o-', linewidth=2, markersize=8, color='purple')
    plt.xlabel('训练轮次 / Training Round')
    plt.ylabel('累积平均奖励 / Cumulative Average Reward')
    plt.title('累积学习效果\\nCumulative Learning Effect')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/extended_ppo_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 综合分析图表已保存: {filename}")
    
    return filename, all_rewards, all_episodes

def print_comprehensive_analysis(results, all_rewards):
    """打印综合分析报告"""
    print("\\n" + "="*80)
    print("📊 PPO扩展训练综合分析报告 / Comprehensive PPO Training Analysis")
    print("="*80)
    
    if not results:
        print("❌ 没有训练结果可分析")
        return
    
    # 总体统计
    total_episodes = sum(len(r['rewards']) for r in results)
    overall_avg = np.mean(all_rewards)
    overall_std = np.std(all_rewards)
    overall_max = np.max(all_rewards)
    overall_min = np.min(all_rewards)
    
    print(f"\\n🎯 总体性能 / Overall Performance:")
    print(f"  训练轮数 / Training Rounds: {len(results)}")
    print(f"  总回合数 / Total Episodes: {total_episodes}")
    print(f"  平均奖励 / Average Reward: {overall_avg:.3f} ± {overall_std:.3f}")
    print(f"  最佳奖励 / Best Reward: {overall_max:.3f}")
    print(f"  最差奖励 / Worst Reward: {overall_min:.3f}")
    print(f"  奖励范围 / Reward Range: {overall_max - overall_min:.3f}")
    
    # 各轮对比
    print(f"\\n📈 各轮训练对比 / Round-by-Round Comparison:")
    for result in results:
        improvement = ""
        if result['round'] > 1:
            prev_avg = results[result['round']-2]['avg_reward']
            change = result['avg_reward'] - prev_avg
            improvement = f" ({change:+.3f})"
        
        print(f"  第{result['round']}轮: {result['avg_reward']:.3f} ± {result['std_reward']:.3f}{improvement}")
    
    # 学习趋势分析
    if len(results) >= 2:
        first_round_avg = results[0]['avg_reward']
        last_round_avg = results[-1]['avg_reward']
        total_improvement = last_round_avg - first_round_avg
        improvement_pct = (total_improvement / abs(first_round_avg)) * 100 if first_round_avg != 0 else 0
        
        print(f"\\n🚀 学习趋势 / Learning Trend:")
        print(f"  首轮平均奖励 / First Round Average: {first_round_avg:.3f}")
        print(f"  末轮平均奖励 / Last Round Average: {last_round_avg:.3f}")
        print(f"  总体改进 / Total Improvement: {total_improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        if total_improvement > 0.1:
            print("  ✅ 显著学习改进 / Significant learning improvement!")
        elif total_improvement > 0:
            print("  ⚡ 轻微学习改进 / Slight learning improvement")
        else:
            print("  ⚠️ 未观察到明显改进 / No significant improvement observed")
    
    # 稳定性分析
    avg_stability = np.mean([1.0 / (1.0 + r['std_reward']) for r in results])
    print(f"\\n🎲 训练稳定性 / Training Stability:")
    print(f"  平均稳定性评分 / Average Stability Score: {avg_stability:.3f}")
    
    if avg_stability > 0.7:
        print("  ✅ 训练整体稳定 / Training is generally stable")
    elif avg_stability > 0.5:
        print("  ⚡ 训练较稳定 / Training is moderately stable")
    else:
        print("  ⚠️ 训练不稳定 / Training is unstable")
    
    # 推荐
    print(f"\\n💡 训练建议 / Training Recommendations:")
    if overall_avg < -0.5:
        print("  🔧 建议调整超参数以提高奖励")
        print("  🔧 Consider adjusting hyperparameters to improve rewards")
    if overall_std > 0.5:
        print("  📊 建议增加训练稳定性措施")
        print("  📊 Consider adding measures to improve training stability")
    if total_improvement > 0:
        print("  ✅ 学习算法有效，可继续扩展训练")
        print("  ✅ Learning algorithm is effective, can extend training")

if __name__ == "__main__":
    try:
        # 运行扩展训练
        results = run_extended_ppo_training()
        
        if results:
            # 创建综合分析
            chart_file, all_rewards, all_episodes = create_comprehensive_analysis(results)
            
            # 打印分析报告
            print_comprehensive_analysis(results, all_rewards)
            
            print(f"\\n🎉 PPO扩展训练验证完成! / Extended PPO Training Validation Complete!")
            print(f"📈 综合分析图表: {chart_file}")
            print(f"📁 请查看 logs/ 目录中的详细图表文件")
            
        else:
            print("❌ 没有收集到训练结果 / No training results collected")
            
    except Exception as e:
        print(f"❌ 扩展训练过程出错: {e}")
        import traceback
        traceback.print_exc()
