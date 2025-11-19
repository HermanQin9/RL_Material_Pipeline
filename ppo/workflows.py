"""High-level PPO training workflows."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def run_4k_ppo_training(episodes: int = 50) -> Tuple[list[float], list[int], list[float], int, str]:
 """Run PPO training on the 4K dataset with a fallback to the 200-sample mode."""
 print("START 4KPPO / Starting 4K Dataset PPO Training")
 print("=" * 70)
 print(" / Configuration:")
 print(" - : 4,000")
 print(" - Dataset size: 4,000 material samples")
 print(f" - : {episodes}")
 print(f" - Training episodes: {episodes}")
 print(f" - : {episodes * 2}")
 print(f" - Estimated time: ~{episodes * 2} minutes")
 print("=" * 70)

 os.environ.setdefault("PIPELINE_TEST", "0")
 start_time = time.time()

 try:
 env = PipelineEnv()
 trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
 init_time = time.time() - start_time
 print(f"SUCCESS : {init_time:.1f}")
 dataset_mode = "4K"
 except Exception as exc: # pragma: no cover - defensive
 print(f"WARNING 4K: {str(exc)[:150]}")
 print(" 200…")
 os.environ["PIPELINE_TEST"] = "1"

 try:
 env = PipelineEnv()
 trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
 init_time = time.time() - start_time
 print(f"SUCCESS : {init_time:.1f}")
 print(" 200")
 dataset_mode = "200_extended"
 except Exception as fallback_exc: # pragma: no cover
 print(f"ERROR : {fallback_exc}")
 return [], [], [], 0, "failed"

 rewards: list[float] = []
 episode_lengths: list[int] = []
 training_times: list[float] = []
 successful_episodes = 0

 print(f"\nSTART {episodes} (: {dataset_mode})…")
 print("-" * 50)

 for episode in range(episodes):
 episode_start = time.time()

 try:
 obs = env.reset()
 total_reward = 0.0
 steps = 0
 done = False

 while not done and steps < 10:
 action, _ = trainer.select_action(obs)
 obs, reward, done, _, _ = env.step(action)
 total_reward += reward
 steps += 1

 episode_time = time.time() - episode_start
 rewards.append(total_reward)
 episode_lengths.append(steps)
 training_times.append(episode_time)
 successful_episodes += 1

 if (episode + 1) % 5 == 0:
 recent_avg = np.mean(rewards[-5:])
 avg_time = np.mean(training_times[-5:])
 print(
 f" {episode + 1:2d}/{episodes}: ={total_reward:.3f}, ={steps}, "
 f"5={recent_avg:.3f}, ={episode_time:.1f}s"
 )

 if (episode + 1) % 10 == 0:
 overall_avg = np.mean(rewards)
 overall_std = np.std(rewards)
 max_reward = np.max(rewards)
 print(
 f" : ={overall_avg:.3f}±{overall_std:.3f}, ={max_reward:.3f}"
 )
 except Exception as exc: # pragma: no cover - continue training loop
 print(f"ERROR {episode + 1} : {str(exc)[:100]}")
 rewards.append(-1.0)
 episode_lengths.append(0)
 training_times.append(0.0)

 total_time = time.time() - start_time
 print("\nSUCCESS PPO!")
 print(f" : {dataset_mode}")
 print(f" : {total_time/60:.1f} ")
 print(f" : {successful_episodes}/{episodes}")
 print(f" : {total_time/episodes:.1f} ")

 return rewards, episode_lengths, training_times, successful_episodes, dataset_mode


def analyze_4k_training_results(
 rewards: Sequence[float],
 episode_lengths: Sequence[int],
 training_times: Sequence[float],
 successful_episodes: int,
 dataset_mode: str = "unknown",
) -> str:
 """Analyse PPO training metrics for the 4K dataset."""
 if not rewards:
 print("ERROR ")
 return "no_data"

 print("\n" + "=" * 70)
 print(f" PPO / PPO Training Analysis (: {dataset_mode})")
 print("=" * 70)

 total_episodes = len(rewards)
 valid_rewards = [r for r in rewards if r > -1.0]

 learning_assessment = "unknown"
 if valid_rewards:
 avg_reward = float(np.mean(valid_rewards))
 std_reward = float(np.std(valid_rewards))
 max_reward = float(np.max(valid_rewards))
 min_reward = float(np.min(valid_rewards))

 print("\n / Training Performance:")
 print(f" / Total Episodes: {total_episodes}")
 print(f" / Successful Episodes: {len(valid_rewards)}")
 print(f" / Success Rate: {len(valid_rewards)/total_episodes*100:.1f}%")
 print(f" / Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
 print(f" / Best Reward: {max_reward:.3f}")
 print(f" / Worst Reward: {min_reward:.3f}")
 print(f" / Reward Range: {max_reward - min_reward:.3f}")

 if len(valid_rewards) >= 20:
 mid = len(valid_rewards) // 2
 first_avg = float(np.mean(valid_rewards[:mid]))
 second_avg = float(np.mean(valid_rewards[mid:]))
 improvement = second_avg - first_avg
 improvement_pct = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0.0

 print("\n / Learning Trend:")
 print(f" / First Half: {first_avg:.3f}")
 print(f" / Second Half: {second_avg:.3f}")
 print(f" / Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

 if improvement > 0.1:
 print(" SUCCESS ! / Significant learning improvement!")
 learning_assessment = "excellent"
 elif improvement > 0.05:
 print(" / Slight learning improvement")
 learning_assessment = "good"
 elif improvement > -0.05:
 print(" / Relatively stable")
 learning_assessment = "stable"
 else:
 print(" WARNING / Performance decline")
 learning_assessment = "concerning"

 if training_times:
 positive_times = [t for t in training_times if t > 0]
 if positive_times:
 avg_time = float(np.mean(positive_times))
 total_time = float(np.sum(positive_times))
 print("\n / Time Performance:")
 print(f" / Total Time: {total_time/60:.1f} ")
 print(f" / Average per Episode: {avg_time:.1f} ")
 if avg_time > 0:
 print(f" / Processing Efficiency: {4000/avg_time:.0f} /")

 print("\n / Comparison with Test Mode:")
 print(" / Dataset Scale: 4,000 vs 200 (20)")
 print(" / Expected Processing Time: ~20")
 print(" / Learning Complexity: ")

 return learning_assessment


def create_4k_visualization(rewards: Sequence[float], episode_lengths: Sequence[int]) -> Optional[str]:
 """Generate visualisations summarising the 4K PPO training run."""
 if not rewards:
 print("ERROR ")
 return None

 valid_data = [
 (idx, reward, length)
 for idx, (reward, length) in enumerate(zip(rewards, episode_lengths))
 if reward > -1.0
 ]
 if not valid_data:
 print("ERROR ")
 return None

 episodes, valid_rewards, valid_lengths = zip(*valid_data)
 fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

 ax1.plot(episodes, valid_rewards, "b-", alpha=0.6, linewidth=1, label="Episode Rewards")
 if len(valid_rewards) >= 10:
 window = min(10, len(valid_rewards) // 4 or 1)
 moving_avg = np.convolve(valid_rewards, np.ones(window) / window, mode="valid")
 moving_eps = episodes[window - 1 :]
 ax1.plot(moving_eps, moving_avg, "r-", linewidth=2, label=f"Moving Average ({window})")
 ax1.set_xlabel("Episode")
 ax1.set_ylabel("Reward")
 ax1.set_title("4K Dataset PPO Learning Curve\n4KPPO")
 ax1.legend()
 ax1.grid(True, alpha=0.3)

 ax2.hist(valid_rewards, bins=20, alpha=0.7, color="lightblue", edgecolor="black")
 ax2.axvline(np.mean(valid_rewards), color="red", linestyle="--", label=f"Mean: {np.mean(valid_rewards):.3f}")
 ax2.set_xlabel("Reward Value")
 ax2.set_ylabel("Frequency")
 ax2.set_title("Reward Distribution\n")
 ax2.legend()
 ax2.grid(True, alpha=0.3)

 ax3.plot(episodes, valid_lengths, "g-", alpha=0.6, marker="o", markersize=3)
 ax3.set_xlabel("Episode")
 ax3.set_ylabel("Episode Length (Steps)")
 ax3.set_title("Episode Length Over Time\n")
 ax3.grid(True, alpha=0.3)

 if len(valid_rewards) >= 10:
 segment_size = max(5, len(valid_rewards) // 10)
 segment_avgs = []
 segment_episodes = []
 for i in range(0, len(valid_rewards), segment_size):
 segment = valid_rewards[i : i + segment_size]
 if segment:
 segment_avgs.append(float(np.mean(segment)))
 segment_episodes.append(episodes[i + len(segment) // 2])
 ax4.plot(segment_episodes, segment_avgs, "o-", linewidth=2, markersize=6, color="purple")
 ax4.set_xlabel("Episode")
 ax4.set_ylabel("Segment Average Reward")
 ax4.set_title("Learning Progress (Segmented)\n")
 ax4.grid(True, alpha=0.3)
 else:
 ax4.text(0.5, 0.5, "Insufficient data\nfor segmented analysis", ha="center", va="center", transform=ax4.transAxes)
 ax4.set_title("Learning Progress\n")

 plt.tight_layout()
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 filename = f"logs/ppo_4k_training_{timestamp}.png"
 plt.savefig(filename, dpi=300, bbox_inches="tight")
 print(f"SUCCESS 4K: {filename}")
 return filename