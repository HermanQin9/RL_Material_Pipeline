#!/usr/bin/env python3
"""
PPO - 
PPO Training Mode Validation - Multiple rounds with learning curves
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 
os.environ['PIPELINE_TEST'] = '0'

sys.path.append('.')
from ppo.trainer import PPOTrainer
from env.pipeline_env import PipelineEnv

def run_multiple_training_rounds(num_rounds=3, episodes_per_round=30):
 """
 PPO
 Run multiple rounds of PPO training
 """
 print("START PPO / Starting PPO Training Mode Validation")
 print(f" / Configuration:")
 print(f" - / Training Rounds: {num_rounds}")
 print(f" - / Episodes per Round: {episodes_per_round}")
 print(f" - / Total Episodes: {num_rounds * episodes_per_round}")
 print("=" * 60)

 all_rewards = []
 all_episode_numbers = []
 round_summaries = []

 for round_num in range(1, num_rounds + 1):
 print(f"\n {round_num} / Round {round_num} Training")
 print("-" * 40)

 # 
 env = PipelineEnv()
 trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)

 round_rewards = []
 round_episodes = []

 try:
 # 
 for episode in range(episodes_per_round):
 obs = env.reset()
 total_reward = 0
 steps = 0
 done = False

 while not done and steps < 10: # 
 action, _ = trainer.select_action(obs) # actionlog_prob
 obs, reward, done, _, info = env.step(action)
 total_reward += reward
 steps += 1

 round_rewards.append(total_reward)
 episode_num = (round_num - 1) * episodes_per_round + episode + 1
 round_episodes.append(episode_num)

 # 5
 if (episode + 1) % 5 == 0:
 avg_reward = np.mean(round_rewards[-5:])
 print(f" {episode + 1:2d}/30: = {avg_reward:.3f}")

 # 
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

 print(f"\n {round_num} / Round {round_num} Results:")
 print(f" / Average Reward: {round_avg:.3f} ± {round_std:.3f}")
 print(f" / Max Reward: {round_max:.3f}")
 print(f" / Min Reward: {round_min:.3f}")

 # 
 all_rewards.extend(round_rewards)
 all_episode_numbers.extend(round_episodes)

 except Exception as e:
 print(f"ERROR {round_num} : {e}")
 continue

 return all_rewards, all_episode_numbers, round_summaries

def plot_learning_curves(rewards, episodes, round_summaries):
 """

 Plot learning curves
 """
 print("\n / Plotting Learning Curves...")

 # 
 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

 # 1: 
 ax1.plot(episodes, rewards, 'b-', alpha=0.6, linewidth=1, label='Episode Rewards')

 # 
 window_size = 10
 if len(rewards) >= window_size:
 moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
 moving_episodes = episodes[window_size-1:]
 ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} episodes)')

 ax1.set_xlabel('Episode')
 ax1.set_ylabel('Reward')
 ax1.set_title('PPO Learning Curve - Training Mode (4000 samples)\nPPO - (4000)', fontsize=14)
 ax1.legend()
 ax1.grid(True, alpha=0.3)

 # 2: 
 if round_summaries:
 round_nums = [r['round'] for r in round_summaries]
 round_avgs = [r['avg_reward'] for r in round_summaries]
 round_stds = [r['std_reward'] for r in round_summaries]

 ax2.errorbar(round_nums, round_avgs, yerr=round_stds, 
 marker='o', linewidth=2, markersize=8, capsize=5)
 ax2.set_xlabel('Training Round / ')
 ax2.set_ylabel('Average Reward / ')
 ax2.set_title('Average Reward per Training Round\n', fontsize=14)
 ax2.grid(True, alpha=0.3)

 # 
 for i, (x, y) in enumerate(zip(round_nums, round_avgs)):
 ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
 xytext=(0,10), ha='center', fontsize=10)

 plt.tight_layout()

 # 
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 filename = f"logs/ppo_training_curves_{timestamp}.png"
 plt.savefig(filename, dpi=300, bbox_inches='tight')
 print(f"SUCCESS : {filename}")

 return filename

def analyze_results(rewards, round_summaries):
 """

 Analyze training results
 """
 print("\n" + "="*60)
 print(" PPO / PPO Training Results Analysis")
 print("="*60)

 # 
 overall_avg = np.mean(rewards)
 overall_std = np.std(rewards)
 overall_max = np.max(rewards)
 overall_min = np.min(rewards)

 print(f"\n / Overall Performance:")
 print(f" / Total Episodes: {len(rewards)}")
 print(f" / Average Reward: {overall_avg:.3f} ± {overall_std:.3f}")
 print(f" / Best Reward: {overall_max:.3f}")
 print(f" / Worst Reward: {overall_min:.3f}")
 print(f" / Reward Range: {overall_max - overall_min:.3f}")

 # 
 if len(rewards) >= 20:
 first_half = rewards[:len(rewards)//2]
 second_half = rewards[len(rewards)//2:]

 improvement = np.mean(second_half) - np.mean(first_half)
 improvement_pct = (improvement / abs(np.mean(first_half))) * 100 if np.mean(first_half) != 0 else 0

 print(f"\n / Learning Trend:")
 print(f" / First Half Average: {np.mean(first_half):.3f}")
 print(f" / Second Half Average: {np.mean(second_half):.3f}")
 print(f" / Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

 if improvement > 0.05:
 print(" SUCCESS / Significant learning improvement detected!")
 elif improvement > 0:
 print(" / Slight learning improvement detected")
 else:
 print(" WARNING / No significant improvement observed")

 # 
 if len(round_summaries) > 1:
 print(f"\n / Round Comparison:")
 for i, summary in enumerate(round_summaries):
 print(f" {summary['round']} / Round {summary['round']}: "
 f"{summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")

 # 
 stability = 1.0 / (1.0 + overall_std) # 
 print(f"\n / Training Stability:")
 print(f" / Reward Variance: {overall_std**2:.3f}")
 print(f" / Stability Score: {stability:.3f} (0-1, )")

 if stability > 0.7:
 print(" SUCCESS / Training is stable")
 elif stability > 0.5:
 print(" / Training is moderately stable")
 else:
 print(" WARNING / Training is unstable")

if __name__ == "__main__":
 try:
 # 
 rewards, episodes, summaries = run_multiple_training_rounds(num_rounds=3, episodes_per_round=30)

 if len(rewards) > 0:
 # 
 curve_file = plot_learning_curves(rewards, episodes, summaries)

 # 
 analyze_results(rewards, summaries)

 print(f"\n PPO! / PPO Training Validation Complete!")
 print(f" : {curve_file}")

 else:
 print("ERROR / No training data collected")

 except Exception as e:
 print(f"ERROR : {e}")
 import traceback
 traceback.print_exc()
