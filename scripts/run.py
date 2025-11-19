#!/usr/bin/env python3
"""
 / Project Runner

This script ensures all project commands use the correct conda environment.
conda
"""

import os
import sys
import subprocess
from pathlib import Path

# Python
PYTHON_PATH = r"D:\conda_envs\summer_project_2025\python.exe"

def run_command(script_name, *args):
 """"""
 cmd = [PYTHON_PATH, script_name] + list(args)
 print(f"START : {' '.join(cmd)}")
 result = subprocess.run(cmd, cwd=os.getcwd())
 return result.returncode

def main():
 """"""
 if len(sys.argv) < 2:
 print("""
 / Usage:
 python run.py <command> [args...]

 / Available commands:
 pipeline - 
 train-ppo - PPO
 eval-ppo - PPO
 example - 
 test - 
 check-env - 
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
 print(f"ERROR : {command}")
 return 1

if __name__ == "__main__":
 sys.exit(main())
