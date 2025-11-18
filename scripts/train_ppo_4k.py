#!/usr/bin/env python3
"""
4K数据集PPO训练验证脚本（命令行入口）
Command-line entry point orchestrating the 4K PPO workflow utilities.
"""
from __future__ import annotations

from ppo.workflows import (
    analyze_4k_training_results,
    create_4k_visualization,
    run_4k_ppo_training,
)


def main(episodes: int = 40) -> None:
    print("🎯 开始4K数据集PPO训练验证")
    print("🎯 Starting 4K Dataset PPO Training Validation")

    rewards, lengths, times, success_count, dataset_mode = run_4k_ppo_training(episodes=episodes)
    if not rewards:
        print("❌ PPO训练失败，没有收集到数据")
        return

    assessment = analyze_4k_training_results(rewards, lengths, times, success_count, dataset_mode)
    chart_file = create_4k_visualization(rewards, lengths)

    print(f"\n🎉 PPO训练验证完成! (模式: {dataset_mode})")
    if chart_file:
        print(f"📈 学习曲线图表: {chart_file}")
    print("📁 请查看 logs/ 目录中的图表文件")
    print(f"📝 学习效果总结: {assessment}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
