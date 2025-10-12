#!/usr/bin/env python3
"""PPO训练结果分析和可视化 / PPO Training Results Analysis and Visualization"""
from __future__ import annotations
from config import MODEL_DIR
from ppo.analysis.results import analyze_checkpoint, find_latest_checkpoint

def main() -> None:
    try:
        checkpoint = find_latest_checkpoint(MODEL_DIR)
        print(f"🔍 使用最新的模型检查点: {checkpoint}")
        analyze_checkpoint(checkpoint)
    except Exception as exc:  # pragma: no cover - CLI feedback
        print("❌ 分析失败 / Analysis failed")
        print(f"错误信息 / Error: {exc}")
        raise

if __name__ == "__main__":
    main()
