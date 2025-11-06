#!/usr/bin/env python3
"""PPO / Plot learning curves from the latest PPO checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppo.analysis import find_latest_checkpoint, load_training_data, rolling_mean


def plot_curves(rewards: Sequence[float], lengths: Sequence[float], output: Path, window: int) -> Path:
 rewards_arr = np.asarray(rewards, dtype=float)
 lengths_arr = np.asarray(lengths, dtype=float) if lengths else np.zeros_like(rewards_arr)
 episodes = np.arange(1, len(rewards_arr) + 1)

 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

 ax1.plot(episodes, rewards_arr, label="Episode Reward", color="#1f77b4", marker="o", markersize=3, linewidth=1)
 smoothed = rolling_mean(rewards_arr.tolist(), window)
 if smoothed is not None:
 moving_avg, moving_eps = smoothed
 ax1.plot(moving_eps, moving_avg, label=f"{window}-episode Moving Avg", color="#d62728", linewidth=2)
 ax1.set_ylabel("Reward / ")
 ax1.set_title("PPO Episode Rewards (Latest Run)\n")
 ax1.grid(alpha=0.3)
 ax1.legend()

 ax2.plot(episodes, lengths_arr, label="Episode Length", color="#2ca02c", marker="o", markersize=3, linewidth=1)
 ax2.set_xlabel("Episode / ")
 ax2.set_ylabel("Length / ")
 ax2.set_title("Episode Lengths / ")
 ax2.grid(alpha=0.3)
 ax2.legend()

 plt.tight_layout()
 output.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(output, dpi=300, bbox_inches="tight")
 return output


def main() -> None:
 parser = argparse.ArgumentParser(description="Plot learning curves from the latest PPO checkpoint")
 parser.add_argument("--checkpoint", type=Path, default=None, help="Path to specific checkpoint (defaults to latest ppo_agent*.pth)")
 parser.add_argument("--output", type=Path, default=None, help="Output PNG path (defaults to logs/ppo_learning_curves_<timestamp>.png)")
 parser.add_argument("--window", type=int, default=10, help="Moving average window size")
 args = parser.parse_args()

 models_dir = Path("models")
 checkpoint_path = args.checkpoint or find_latest_checkpoint(models_dir)

 rewards, lengths = load_training_data(checkpoint_path)

 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 output_path = args.output or Path("logs") / f"ppo_learning_curves_{timestamp}.png"

 plot_curves(rewards, lengths, output_path, window=max(2, args.window))
 print(f"SUCCESS : {output_path}")
 print(f" : {checkpoint_path}")
 print(f" : {len(rewards)}")
 print(f" : {min(rewards):.3f} ~ {max(rewards):.3f}")


if __name__ == "__main__":
 main()
