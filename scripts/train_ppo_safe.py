#!/usr/bin/env python3
"""
安全PPO训练与可视化脚本（命令行入口）
Command-line entry point delegating to the safe trainer utilities.
"""
from __future__ import annotations

from ppo.safe_trainer import run_safe_training


def main(num_episodes: int = 15) -> None:
    print("🤖 开始PPO训练和学习分析…")
    logs, _ = run_safe_training(num_episodes=num_episodes)
    if logs["rewards"]:
        print("\n✅ PPO训练和分析完成！")
        print("📁 请查看 logs/ 目录中的图像文件")
    else:
        print("\n⚠️ 未产生训练日志，请检查配置")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
