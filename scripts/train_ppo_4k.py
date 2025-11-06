#!/usr/bin/env python3
"""
4KPPO
Command-line entry point orchestrating the 4K PPO workflow utilities.
"""
from __future__ import annotations

from ppo.workflows import (
 analyze_4k_training_results,
 create_4k_visualization,
 run_4k_ppo_training,
)


def main(episodes: int = 40) -> None:
 print(" 4KPPO")
 print(" Starting 4K Dataset PPO Training Validation")

 rewards, lengths, times, success_count, dataset_mode = run_4k_ppo_training(episodes=episodes)
 if not rewards:
 print("ERROR PPO")
 return

 assessment = analyze_4k_training_results(rewards, lengths, times, success_count, dataset_mode)
 chart_file = create_4k_visualization(rewards, lengths)

 print(f"\n PPO! (: {dataset_mode})")
 if chart_file:
 print(f" : {chart_file}")
 print(" logs/ ")
 print(f" : {assessment}")


if __name__ == "__main__": # pragma: no cover - CLI entry
 main()
