#!/usr/bin/env python3
"""
项目运行器 / Project Runner

This script ensures all project commands use the correct conda environment.
此脚本确保所有项目命令使用正确的conda环境。
"""

import os
import sys
import subprocess
from pathlib import Path

# 正确的Python路径
PYTHON_PATH = r"D:\conda_envs\summer_project_2025\python.exe"

def run_command(script_name, *args):
    """使用正确的环境运行命令"""
    cmd = [PYTHON_PATH, script_name] + list(args)
    print(f"🚀 运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("""
使用方法 / Usage:
    python run.py <command> [args...]

可用命令 / Available commands:
    pipeline              - 运行机器学习流水线
    train-ppo              - 训练PPO智能体
    eval-ppo               - 评估PPO智能体
    example                - 运行示例用法
    test                   - 运行测试
    check-env              - 检查环境
        """)
        return

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "pipeline":
        return run_command("main.py", "--mode", "pipeline", *args)
    elif command == "train-ppo":
        return run_command("scripts/train_ppo.py", *args)
    elif command == "eval-ppo":
        return run_command("scripts/eval_ppo.py", *args)
    elif command == "example":
        return run_command("scripts/example_usage.py", *args)
    elif command == "test":
        return run_command("-m", "pytest", "tests/", *args)
    elif command == "check-env":
        return run_command("check_env.py", *args)
    else:
        print(f"❌ 未知命令: {command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
