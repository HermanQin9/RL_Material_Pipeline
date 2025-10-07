#!/usr/bin/env python3
"""绘制最近一次PPO训练的学习曲线 / Plot learning curves from the latest PPO checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def rolling_mean(values: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(values, dtype=float)
    if window < 2 or arr.size < window:
        return None
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    episodes = np.arange(window, arr.size + 1)
    return smoothed, episodes


def plot_curves(rewards: Sequence[float], lengths: Sequence[float], output: Path, window: int) -> Path:
    rewards_arr = np.asarray(rewards, dtype=float)
    lengths_arr = np.asarray(lengths, dtype=float) if lengths else np.zeros_like(rewards_arr)
    episodes = np.arange(1, len(rewards_arr) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(episodes, rewards_arr, label="Episode Reward", color="#1f77b4", marker="o", markersize=3, linewidth=1)
    smoothed = rolling_mean(rewards_arr, window)
    if smoothed is not None:
        moving_avg, moving_eps = smoothed
        ax1.plot(moving_eps, moving_avg, label=f"{window}-episode Moving Avg", color="#d62728", linewidth=2)
    ax1.set_ylabel("Reward / 奖励")
    ax1.set_title("PPO Episode Rewards (Latest Run)\n最新训练轮次奖励轨迹")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(episodes, lengths_arr, label="Episode Length", color="#2ca02c", marker="o", markersize=3, linewidth=1)
    ax2.set_xlabel("Episode / 回合")
    ax2.set_ylabel("Length / 步数")
    ax2.set_title("Episode Lengths / 每轮步数")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    return output


def find_latest_checkpoint(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("ppo_agent*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No PPO checkpoints found in {models_dir}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves from the latest PPO checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to specific checkpoint (defaults to latest ppo_agent*.pth)")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (defaults to logs/ppo_learning_curves_<timestamp>.png)")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size")
    args = parser.parse_args()

    models_dir = Path("models")
    checkpoint_path = args.checkpoint or find_latest_checkpoint(models_dir)
    ckpt = load_checkpoint(checkpoint_path)

    rewards = ckpt.get("episode_rewards")
    lengths = ckpt.get("episode_lengths")
    if not rewards:
        raise ValueError(f"No episode rewards found in checkpoint {checkpoint_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or Path("logs") / f"ppo_learning_curves_{timestamp}.png"

    plot_curves(rewards, lengths or [], output_path, window=max(2, args.window))
    print(f"✅ 学习曲线已保存: {output_path}")
    print(f"   数据来源: {checkpoint_path}")
    print(f"   总回合数: {len(rewards)}")
    print(f"   奖励范围: {min(rewards):.3f} ~ {max(rewards):.3f}")


if __name__ == "__main__":
    main()
