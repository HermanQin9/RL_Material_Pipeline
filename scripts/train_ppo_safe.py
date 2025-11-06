#!/usr/bin/env python3
"""
PPO
Command-line entry point delegating to the safe trainer utilities.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ppo.safe_trainer import run_safe_training


def main(num_episodes: int = 15) -> None:
 print(" PPO…")
 logs, _ = run_safe_training(num_episodes=num_episodes)
 if logs["rewards"]:
 print("\nSUCCESS PPO")
 print(" logs/ ")
 else:
 print("\nWARNING ")


if __name__ == "__main__": # pragma: no cover - CLI entry
 main()
