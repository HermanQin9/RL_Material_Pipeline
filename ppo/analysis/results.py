"""Analysis helpers for PPO training checkpoints."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")  # ensure headless plotting
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def find_latest_checkpoint(models_dir: Path, pattern: str = "ppo_agent*.pth") -> Path:
    candidates = sorted(models_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"在 {models_dir} 中未找到任何PPO模型 (pattern={pattern})")
    return candidates[0]


def load_training_data(checkpoint_path: Path) -> tuple[list[float], list[float]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    rewards = ckpt.get("episode_rewards")
    lengths = ckpt.get("episode_lengths")
    if not rewards:
        raise ValueError(f"检查点 {checkpoint_path} 中没有 episode_rewards")
    rewards = list(map(float, rewards))
    if not lengths:
        lengths = [0.0 for _ in rewards]
    else:
        lengths = list(map(float, lengths))
    return rewards, lengths


def rolling_mean(values: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(values, dtype=float)
    if window < 2 or arr.size < window:
        return None
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    episodes = np.arange(window, arr.size + 1)
    return smoothed, episodes


def compute_success_flags(rewards: Iterable[float], failure_threshold: float = -0.95) -> list[int]:
    arr = np.asarray(list(rewards), dtype=float)
    return (arr > failure_threshold).astype(int).tolist()


def create_visualizations(
    rewards: Sequence[float],
    success_flags: Sequence[int],
    episode_lengths: Sequence[float],
    output_path: Path,
    window: int,
) -> Path:
    rewards_arr = np.asarray(rewards, dtype=float)
    lengths_arr = np.asarray(episode_lengths, dtype=float)
    episodes = np.arange(1, len(rewards_arr) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(episodes, rewards_arr, color="#1f77b4", marker="o", markersize=3, linewidth=1, label="Episode Reward")
    smoothed = rolling_mean(rewards_arr.tolist(), window)
    if smoothed is not None:
        moving_avg, moving_eps = smoothed
        ax1.plot(moving_eps, moving_avg, color="#d62728", linewidth=2, label=f"{window}回合移动平均")
    ax1.set_title("PPO Episode Rewards\n每回合奖励")
    ax1.set_xlabel("Episode / 回合")
    ax1.set_ylabel("Reward / 奖励")
    ax1.grid(alpha=0.3)
    ax1.legend()

    success_arr = np.asarray(success_flags, dtype=float)
    cumulative_success = np.cumsum(success_arr) / np.arange(1, len(success_arr) + 1)
    ax2.plot(episodes, cumulative_success, color="#2ca02c", linewidth=2, marker="o", markersize=3)
    ax2.set_ylim(0, 1)
    ax2.set_title("Cumulative Success Rate\n累计成功率")
    ax2.set_xlabel("Episode / 回合")
    ax2.set_ylabel("Success Rate / 成功率")
    ax2.grid(alpha=0.3)

    ax3.hist(rewards_arr, bins=min(20, max(5, len(rewards_arr) // 3)), color="#9467bd", alpha=0.8)
    ax3.set_title("Reward Distribution\n奖励分布")
    ax3.set_xlabel("Reward / 奖励")
    ax3.set_ylabel("Frequency / 频数")
    ax3.grid(alpha=0.2)

    ax4.plot(episodes, lengths_arr, color="#ff7f0e", marker="o", markersize=3, linewidth=1, label="Episode Length")
    ax4.set_title("Episode Lengths\n每回合步数")
    ax4.set_xlabel("Episode / 回合")
    ax4.set_ylabel("Length / 步数")
    ax4.grid(alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


def summarize_rewards(rewards: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(rewards, dtype=float)
    stats: dict[str, float] = {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if arr.size >= 2:
        mid = arr.size // 2
        stats["first_half_mean"] = float(np.mean(arr[:mid]))
        stats["second_half_mean"] = float(np.mean(arr[mid:]))
        stats["improvement"] = stats["second_half_mean"] - stats["first_half_mean"]
    else:
        stats["first_half_mean"] = stats["second_half_mean"] = stats["improvement"] = float("nan")
    return stats


def print_summary(
    checkpoint_path: Path,
    rewards: Sequence[float],
    success_flags: Sequence[int],
    lengths: Sequence[float],
    figure_path: Path,
    window: int,
) -> None:
    stats = summarize_rewards(rewards)
    success_rate = float(np.mean(success_flags)) if success_flags else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0

    print("=" * 70)
    print("📊 PPO训练结果分析 / PPO Training Results Analysis")
    print("=" * 70)
    print(f"🔖 模型检查点 / Checkpoint: {checkpoint_path}")
    print(f"📈 总回合数 / Total Episodes: {int(stats['count'])}")
    print(f"🎯 平均奖励 / Mean Reward: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"🔝 最佳奖励 / Best Reward: {stats['max']:.3f}")
    print(f"🔻 最差奖励 / Worst Reward: {stats['min']:.3f}")
    print(f"✅ 成功率(> -0.95): {success_rate * 100:.1f}% ({int(np.sum(success_flags))}/{len(success_flags)})")
    print(f"⏱️ 平均步数 / Avg Episode Length: {avg_length:.1f}")

    improvement = stats.get("improvement")
    if improvement is not None and not np.isnan(improvement):
        trend = "➡️ 持平"
        if improvement > 0:
            trend = "⬆️ 改进"
        elif improvement < 0:
            trend = "⬇️ 退化"
        print(f"📉 前半段平均奖励: {stats['first_half_mean']:.3f}")
        print(f"📈 后半段平均奖励: {stats['second_half_mean']:.3f}")
        print(f"🚀 奖励变化 / Reward Shift: {improvement:+.3f} ({trend})")

    failures = [idx + 1 for idx, flag in enumerate(success_flags) if flag == 0]
    if failures:
        print(f"⚠️ 失败回合 (reward ≤ -0.95): {failures}")
    else:
        print("✅ 未检测到失败回合 / No failing episodes detected")

    print(f"📊 可视化图表已保存 / Figure saved to: {figure_path}")
    print(f"🪄 移动平均窗口 / Moving average window: {window}")
    print("=" * 70)


def analyze_checkpoint(
    checkpoint_path: Path,
    output_path: Path | None = None,
    window: int = 10,
    failure_threshold: float = -0.95,
) -> Path:
    rewards, lengths = load_training_data(checkpoint_path)
    success_flags = compute_success_flags(rewards, failure_threshold=failure_threshold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = Path("logs") / f"ppo_learning_curves_{timestamp}.png"
    figure_path = create_visualizations(rewards, success_flags, lengths, output_path or default_output, window)
    print_summary(checkpoint_path, rewards, success_flags, lengths, figure_path, window)
    return figure_path
