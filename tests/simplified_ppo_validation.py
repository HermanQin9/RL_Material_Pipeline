#!/usr/bin/env python3
"""
PPO
Simplified PPO Multi-Round Training Validation
"""
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 
os.environ['PIPELINE_TEST'] = '1'
sys.path.append('.')

def run_ppo_multiple_rounds():
 """PPO"""
 print("START PPO / Starting Multi-Round PPO Training")
 print("=" * 70)

 rounds_config = [
 {"episodes": 20, "name": "1", "desc": "Round 1"},
 {"episodes": 25, "name": "2", "desc": "Round 2"},
 {"episodes": 30, "name": "3", "desc": "Round 3"}
 ]

 all_rewards = []
 round_summaries = []

 for i, config in enumerate(rounds_config):
 print(f"\n {config['name']} ({config['desc']}) - {config['episodes']} ")
 print("-" * 50)

 try:
 # 
 result = subprocess.run([
 "D:\\conda_envs\\summer_project_2025\\python.exe",
 "train_ppo_safe.py", 
 "--episodes", str(config['episodes'])
 ], capture_output=True, text=True, cwd=".")

 if result.returncode == 0:
 # 
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

 print(f"SUCCESS {config['name']}:")
 print(f" : {len(rewards)}")
 print(f" : {round_summary['avg']:.3f} ± {round_summary['std']:.3f}")
 print(f" : {round_summary['min']:.3f} ~ {round_summary['max']:.3f}")
 else:
 print(f"WARNING {config['name']}")
 else:
 print(f"ERROR {config['name']}: {result.stderr[:200]}")

 except Exception as e:
 print(f"ERROR {config['name']}: {e}")

 return round_summaries, all_rewards

def extract_rewards_from_output(output):
 """"""
 rewards = []
 lines = output.split('\\n')

 for line in lines:
 if ":" in line and ":" in line:
 try:
 # ": -1.000" 
 reward_part = line.split(":")[1].split(",")[0].strip()
 reward = float(reward_part)
 rewards.append(reward)
 except (ValueError, IndexError):
 continue

 return rewards

def create_training_visualization(round_summaries, all_rewards):
 """"""
 if not round_summaries:
 print("ERROR ")
 return None

 print("\\n ...")

 # 
 fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

 # 1. 
 colors = ['blue', 'red', 'green', 'purple', 'orange']
 episode_counter = 0

 for i, summary in enumerate(round_summaries):
 episodes = range(episode_counter, episode_counter + len(summary['rewards']))
 ax1.plot(episodes, summary['rewards'], 'o-', 
 alpha=0.7, color=colors[i % len(colors)], 
 label=summary['name'])
 episode_counter += len(summary['rewards'])

 # 
 if len(all_rewards) >= 8:
 window = 8
 moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
 moving_episodes = range(window-1, len(all_rewards))
 ax1.plot(moving_episodes, moving_avg, 'k-', linewidth=3, 
 label=f'{window}', alpha=0.8)

 ax1.set_xlabel('')
 ax1.set_ylabel('')
 ax1.set_title('PPO\\nPPO Multi-Round Learning Curve')
 ax1.legend()
 ax1.grid(True, alpha=0.3)

 # 2. 
 round_nums = [s['round'] for s in round_summaries]
 avg_rewards = [s['avg'] for s in round_summaries]
 std_rewards = [s['std'] for s in round_summaries]

 bars = ax2.bar(round_nums, avg_rewards, yerr=std_rewards, 
 capsize=5, alpha=0.7, color=colors[:len(round_summaries)])
 ax2.set_xlabel('')
 ax2.set_ylabel('')
 ax2.set_title('\\nAverage Reward Comparison')

 # 
 for i, (x, y) in enumerate(zip(round_nums, avg_rewards)):
 ax2.text(x, y + std_rewards[i] + 0.02, f'{y:.3f}', 
 ha='center', va='bottom', fontweight='bold')
 ax2.grid(True, alpha=0.3)

 # 3. 
 ax3.hist(all_rewards, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
 ax3.axvline(np.mean(all_rewards), color='red', linestyle='--', 
 label=f': {np.mean(all_rewards):.3f}')
 ax3.set_xlabel('')
 ax3.set_ylabel('')
 ax3.set_title('\\nReward Distribution')
 ax3.legend()
 ax3.grid(True, alpha=0.3)

 # 4. 
 if len(round_summaries) > 1:
 improvements = []
 for i in range(1, len(round_summaries)):
 improvement = round_summaries[i]['avg'] - round_summaries[i-1]['avg']
 improvements.append(improvement)

 ax4.plot(range(2, len(round_summaries)+1), improvements, 'o-', 
 linewidth=2, markersize=8, color='green')
 ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
 ax4.set_xlabel('')
 ax4.set_ylabel('')
 ax4.set_title('\\nImprovement Trend')
 ax4.grid(True, alpha=0.3)
 else:
 ax4.text(0.5, 0.5, '\\n', 
 ha='center', va='center', transform=ax4.transAxes)
 ax4.set_title('\\nImprovement Trend')

 plt.tight_layout()

 # 
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 filename = f"logs/ppo_multi_round_analysis_{timestamp}.png"
 plt.savefig(filename, dpi=300, bbox_inches='tight')
 print(f"SUCCESS : {filename}")

 return filename

def analyze_multi_round_results(round_summaries, all_rewards):
 """"""
 print("\\n" + "="*70)
 print(" PPO / Multi-Round Training Analysis")
 print("="*70)

 if not round_summaries:
 print("ERROR ")
 return

 # 
 total_episodes = sum(len(s['rewards']) for s in round_summaries)
 overall_avg = np.mean(all_rewards)
 overall_std = np.std(all_rewards)
 overall_max = np.max(all_rewards)
 overall_min = np.min(all_rewards)

 print(f"\\n / Overall Performance:")
 print(f" : {len(round_summaries)} ")
 print(f" : {total_episodes} ")
 print(f" : {overall_avg:.3f} ± {overall_std:.3f}")
 print(f" : {overall_max:.3f}")
 print(f" : {overall_min:.3f}")
 print(f" : {overall_max - overall_min:.3f}")

 # 
 print(f"\\n / Round-by-Round Details:")
 for i, summary in enumerate(round_summaries):
 improvement = ""
 if i > 0:
 prev_avg = round_summaries[i-1]['avg']
 change = summary['avg'] - prev_avg
 improvement = f" ({change:+.3f})"

 print(f" {summary['name']}: {summary['avg']:.3f} ± {summary['std']:.3f}{improvement}")
 print(f" : {summary['min']:.3f} ~ {summary['max']:.3f}, : {len(summary['rewards'])}")

 # 
 if len(round_summaries) >= 2:
 first_avg = round_summaries[0]['avg']
 last_avg = round_summaries[-1]['avg']
 total_improvement = last_avg - first_avg
 improvement_pct = (total_improvement / abs(first_avg)) * 100 if first_avg != 0 else 0

 print(f"\\nSTART / Learning Trend Analysis:")
 print(f" : {first_avg:.3f}")
 print(f" : {last_avg:.3f}")
 print(f" : {total_improvement:+.3f} ({improvement_pct:+.1f}%)")

 if total_improvement > 0.15:
 print(" SUCCESS ! / Significant improvement!")
 assessment = "excellent"
 elif total_improvement > 0.05:
 print(" / Slight improvement")
 assessment = "good"
 elif total_improvement > -0.05:
 print(" / Relatively stable")
 assessment = "stable"
 else:
 print(" WARNING / Performance decline")
 assessment = "concerning"
 else:
 assessment = "insufficient_data"

 # 
 avg_stability = np.mean([1.0 / (1.0 + s['std']) for s in round_summaries])
 print(f"\\n / Training Stability:")
 print(f" : {avg_stability:.3f} (0-1, )")

 if avg_stability > 0.7:
 print(" SUCCESS / Very stable training")
 stability = "high"
 elif avg_stability > 0.5:
 print(" / Moderately stable")
 stability = "medium"
 else:
 print(" WARNING / Unstable training")
 stability = "low"

 # 
 print(f"\\n / Assessment & Recommendations:")

 if assessment == "excellent":
 print(" PPO!")
 print(" Excellent PPO learning performance!")
 print(" ")
 elif assessment == "good":
 print(" SUCCESS PPO")
 print(" SUCCESS Good PPO learning performance")
 print(" ")
 elif assessment == "stable":
 print(" PPO")
 print(" Stable PPO learning performance")
 print(" ")
 else:
 print(" WARNING PPO")
 print(" WARNING PPO learning needs improvement")
 print(" ")

 if stability == "low":
 print(" ")
 print(" Consider adding stability measures")

if __name__ == "__main__":
 try:
 print(" PPO / Multi-Round PPO Training Validation")

 # 
 round_summaries, all_rewards = run_ppo_multiple_rounds()

 if round_summaries and all_rewards:
 # 
 chart_file = create_training_visualization(round_summaries, all_rewards)

 # 
 analyze_multi_round_results(round_summaries, all_rewards)

 print(f"\\n PPO! / Multi-Round Training Complete!")
 if chart_file:
 print(f" : {chart_file}")
 print(f" logs/ ")

 else:
 print("ERROR / No training data collected")

 except Exception as e:
 print(f"ERROR : {e}")
 import traceback
 traceback.print_exc()
