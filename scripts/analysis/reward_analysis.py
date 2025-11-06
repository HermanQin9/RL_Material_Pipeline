#!/usr/bin/env python3
"""
PPO
PPO Reward Function Analysis and Improvement Suggestions
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_reward_function():
 """"""
 print(" PPO")
 print(" Detailed PPO Reward Function Analysis")
 print("=" * 60)

 print(" :")
 print(" - : -1.000 (90%)")
 print(" - : [-1.0, ~-0.9]")
 print(" - : (~0.02)")
 print()

 print("ERROR :")
 print(" 1. :")
 print(" - -1.0")
 print(" - ''''")
 print(" - ")
 print()

 print(" 2. :")
 print(" - ''")
 print(" - ")
 print(" - ")
 print()

 print(" 3. WARNING :")
 print(" - 'list index out of range' ")
 print(" - ")
 print(" - ")
 print()

 return True

def suggest_reward_improvements():
 """"""
 print(" ")
 print(" Reward Function Improvement Suggestions")
 print("=" * 60)

 print(" 1: ")
 print(" :")
 print(" • : +0.1 ()")
 print(" • : +0.2 ()")
 print(" • : +0.3 ()")
 print(" • : +0.4 ()")
 print(" • : +0.0~0.3 ()")
 print()

 print(" 2: ")
 print(" :")
 print(" • 1: (-0.5 ~ 0.0)")
 print(" • 2: (0.0 ~ 0.5)")
 print(" • 3: (0.5 ~ 1.0)")
 print(" • : -1.0 ()")
 print()

 print(" 3: ")
 print(" :")
 print(" • : 40%")
 print(" • : 30%")
 print(" • : 20%")
 print(" • : 10%")
 print()

 return True

def create_reward_comparison_plot():
 """"""
 print(" ...")

 episodes = np.arange(1, 41)

 # 
 current_rewards = np.full(40, -1.0)
 current_rewards[11] = -0.95
 current_rewards[15] = -0.98
 current_rewards[20] = -0.96
 current_rewards[25] = -0.94

 # 
 improved_rewards = []
 base_reward = -0.8
 for i in range(40):
 # 
 progress = min(i / 30, 1.0)
 noise = np.random.normal(0, 0.1)
 reward = base_reward + progress * 1.5 + noise
 # 
 if i in [8, 15, 22, 28, 35]:
 reward += np.random.uniform(0.3, 0.8)
 # 
 if i in [5, 12, 18, 25]:
 reward = -1.0 + np.random.uniform(-0.2, 0.1)
 improved_rewards.append(max(-1.2, min(1.0, reward)))

 # 
 fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

 # 1. 
 ax1.plot(episodes, current_rewards, 'r-', linewidth=2, label='Current Rewards')
 ax1.fill_between(episodes, current_rewards, -1.1, alpha=0.3, color='red')
 ax1.set_ylabel('Reward / ')
 ax1.set_title('Current Reward Function (Observed)\n')
 ax1.legend()
 ax1.grid(True, alpha=0.3)
 ax1.set_ylim(-1.1, -0.8)

 # 2. 
 ax2.plot(episodes, improved_rewards, 'g-', linewidth=2, label='Improved Rewards')
 ax2.fill_between(episodes, improved_rewards, -1.2, alpha=0.3, color='green')

 # 
 window = 5
 if len(improved_rewards) >= window:
 moving_avg = np.convolve(improved_rewards, np.ones(window)/window, mode='valid')
 moving_episodes = episodes[window-1:]
 ax2.plot(moving_episodes, moving_avg, 'darkgreen', linewidth=3, 
 label=f'Moving Average')

 ax2.set_ylabel('Reward / ')
 ax2.set_title('Improved Reward Function (Simulation)\n')
 ax2.legend()
 ax2.grid(True, alpha=0.3)
 ax2.set_ylim(-1.2, 1.0)

 # 3. 
 current_avg = np.mean(current_rewards)
 improved_avg = np.mean(improved_rewards)
 current_std = np.std(current_rewards)
 improved_std = np.std(improved_rewards)

 metrics = ['\nMean', '\nStd Dev', '\nMax', '\nTrend']
 current_values = [current_avg, current_std, max(current_rewards), 0.001]
 improved_values = [improved_avg, improved_std, max(improved_rewards), 0.025]

 x = np.arange(len(metrics))
 width = 0.35

 bars1 = ax3.bar(x - width/2, current_values, width, label='Current', color='lightcoral', alpha=0.7)
 bars2 = ax3.bar(x + width/2, improved_values, width, label='Improved', color='lightgreen', alpha=0.7)

 ax3.set_ylabel('Value / ')
 ax3.set_title('Reward Function Comparison\n')
 ax3.set_xticks(x)
 ax3.set_xticklabels(metrics)
 ax3.legend()
 ax3.grid(True, alpha=0.3, axis='y')

 # 
 for bars in [bars1, bars2]:
 for bar in bars:
 height = bar.get_height()
 ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

 plt.tight_layout()

 # 
 filename = "logs/reward_function_analysis.png"
 plt.savefig(filename, dpi=300, bbox_inches='tight')
 print(f"SUCCESS : {filename}")

 return filename

def recommend_next_steps():
 """"""
 print("\n" + "=" * 60)
 print("START ")
 print("START Recommended Next Steps")
 print("=" * 60)

 print(" 1: ")
 print(" 1. :")
 print(" - ")
 print(" - ")
 print(" - ")
 print()

 print(" 2. :")
 print(" - 'list index out of range'")
 print(" - ")
 print(" - ")
 print()

 print(" 2: ")
 print(" 1. :")
 print(" - 40100-200")
 print(" - ")
 print(" - ")
 print()

 print(" 2. :")
 print(" - ")
 print(" - ")
 print(" - ")
 print()

 print(" 3: ")
 print(" 1. :")
 print(" - ")
 print(" - ")
 print(" - ")
 print()

 print(" 2. :")
 print(" - ")
 print(" - ")
 print(" - ")

 return True

def main():
 """"""
 print(" PPO")
 print(" In-depth PPO Reward Function Analysis")
 print("=" * 70)

 # 
 analyze_reward_function()
 suggest_reward_improvements()
 chart_file = create_reward_comparison_plot()
 recommend_next_steps()

 # 
 print("\n" + "=" * 70)
 print(" ")
 print(" Analysis Summary")
 print("=" * 70)

 print(" :")
 print(" ERROR ")
 print(" ERROR ")
 print(" ERROR ")
 print()

 print(" :")
 print(" SUCCESS ")
 print(" SUCCESS ")
 print(" SUCCESS ")
 print(" SUCCESS ")

 print(f"\n : {chart_file}")

 return True

if __name__ == "__main__":
 main()
