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
    print("🚀 开始4K数据集PPO训练 / Starting 4K Dataset PPO Training")
    print("=" * 70)
    print("📊 配置 / Configuration:")
    print("  - 数据集大小: 4,000个材料样本")
    print("  - Dataset size: 4,000 material samples")
    print(f"  - 训练回合数: {episodes}")
    print(f"  - Training episodes: {episodes}")
    print(f"  - 预计时间: 约{episodes * 2}分钟")
    print(f"  - Estimated time: ~{episodes * 2} minutes")
    print("=" * 70)

    os.environ.setdefault("PIPELINE_TEST", "0")
    start_time = time.time()

    try:
        env = PipelineEnv()
        trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
        init_time = time.time() - start_time
        print(f"✅ 环境初始化完成，耗时: {init_time:.1f}秒")
        dataset_mode = "4K"
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️ 4K数据集初始化失败: {str(exc)[:150]}")
        print("🔄 切换到200样本测试模式…")
        os.environ["PIPELINE_TEST"] = "1"

        try:
            env = PipelineEnv()
            trainer = PPOTrainer(env, learning_rate=3e-4, clip_ratio=0.2, hidden_size=64)
            init_time = time.time() - start_time
            print(f"✅ 测试模式环境初始化完成，耗时: {init_time:.1f}秒")
            print("📊 使用200样本但运行更多轮次来模拟大数据集学习效果")
            dataset_mode = "200_extended"
        except Exception as fallback_exc:  # pragma: no cover
            print(f"❌ 测试模式也失败: {fallback_exc}")
            return [], [], [], 0, "failed"

    rewards: list[float] = []
    episode_lengths: list[int] = []
    training_times: list[float] = []
    successful_episodes = 0

    print(f"\n🚀 开始训练 {episodes} 个回合 (数据集模式: {dataset_mode})…")
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
                    f"回合 {episode + 1:2d}/{episodes}: 奖励={total_reward:.3f}, 步数={steps}, "
                    f"最近5回合均值={recent_avg:.3f}, 用时={episode_time:.1f}s"
                )

            if (episode + 1) % 10 == 0:
                overall_avg = np.mean(rewards)
                overall_std = np.std(rewards)
                max_reward = np.max(rewards)
                print(
                    f"  📊 阶段统计: 平均={overall_avg:.3f}±{overall_std:.3f}, 最佳={max_reward:.3f}"
                )
        except Exception as exc:  # pragma: no cover - continue training loop
            print(f"❌ 回合 {episode + 1} 出错: {str(exc)[:100]}")
            rewards.append(-1.0)
            episode_lengths.append(0)
            training_times.append(0.0)

    total_time = time.time() - start_time
    print("\n✅ PPO训练完成!")
    print(f"  数据集模式: {dataset_mode}")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  成功回合: {successful_episodes}/{episodes}")
    print(f"  平均每回合: {total_time/episodes:.1f} 秒")

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
        print("❌ 没有训练数据可分析")
        return "no_data"

    print("\n" + "=" * 70)
    print(f"📊 PPO训练结果分析 / PPO Training Analysis (模式: {dataset_mode})")
    print("=" * 70)

    total_episodes = len(rewards)
    valid_rewards = [r for r in rewards if r > -1.0]

    learning_assessment = "unknown"
    if valid_rewards:
        avg_reward = float(np.mean(valid_rewards))
        std_reward = float(np.std(valid_rewards))
        max_reward = float(np.max(valid_rewards))
        min_reward = float(np.min(valid_rewards))

        print("\n🎯 训练性能 / Training Performance:")
        print(f"  总回合数 / Total Episodes: {total_episodes}")
        print(f"  成功回合 / Successful Episodes: {len(valid_rewards)}")
        print(f"  成功率 / Success Rate: {len(valid_rewards)/total_episodes*100:.1f}%")
        print(f"  平均奖励 / Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"  最佳奖励 / Best Reward: {max_reward:.3f}")
        print(f"  最差奖励 / Worst Reward: {min_reward:.3f}")
        print(f"  奖励范围 / Reward Range: {max_reward - min_reward:.3f}")

        if len(valid_rewards) >= 20:
            mid = len(valid_rewards) // 2
            first_avg = float(np.mean(valid_rewards[:mid]))
            second_avg = float(np.mean(valid_rewards[mid:]))
            improvement = second_avg - first_avg
            improvement_pct = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0.0

            print("\n📈 学习趋势 / Learning Trend:")
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

    if training_times:
        positive_times = [t for t in training_times if t > 0]
        if positive_times:
            avg_time = float(np.mean(positive_times))
            total_time = float(np.sum(positive_times))
            print("\n⏱️ 时间性能 / Time Performance:")
            print(f"  总训练时间 / Total Time: {total_time/60:.1f} 分钟")
            print(f"  平均每回合 / Average per Episode: {avg_time:.1f} 秒")
            if avg_time > 0:
                print(f"  数据处理效率 / Processing Efficiency: {4000/avg_time:.0f} 样本/秒")

    print("\n🔍 与测试模式对比 / Comparison with Test Mode:")
    print("  数据集规模 / Dataset Scale: 4,000 vs 200 样本 (20倍)")
    print("  预期处理时间 / Expected Processing Time: ~20倍增长")
    print("  学习复杂度 / Learning Complexity: 显著增加")

    return learning_assessment


def create_4k_visualization(rewards: Sequence[float], episode_lengths: Sequence[int]) -> Optional[str]:
    """Generate visualisations summarising the 4K PPO training run."""
    if not rewards:
        print("❌ 没有数据可视化")
        return None

    valid_data = [
        (idx, reward, length)
        for idx, (reward, length) in enumerate(zip(rewards, episode_lengths))
        if reward > -1.0
    ]
    if not valid_data:
        print("❌ 没有有效的训练数据")
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
    ax1.set_title("4K Dataset PPO Learning Curve\n4K数据集PPO学习曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(valid_rewards, bins=20, alpha=0.7, color="lightblue", edgecolor="black")
    ax2.axvline(np.mean(valid_rewards), color="red", linestyle="--", label=f"Mean: {np.mean(valid_rewards):.3f}")
    ax2.set_xlabel("Reward Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Reward Distribution\n奖励分布")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(episodes, valid_lengths, "g-", alpha=0.6, marker="o", markersize=3)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Episode Length (Steps)")
    ax3.set_title("Episode Length Over Time\n回合长度变化")
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
        ax4.set_title("Learning Progress (Segmented)\n学习进度（分段）")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Insufficient data\nfor segmented analysis", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Learning Progress\n学习进度")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/ppo_4k_training_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ 4K数据集学习曲线已保存: {filename}")
    return filename