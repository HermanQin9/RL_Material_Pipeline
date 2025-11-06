#!/usr/bin/env python3
"""PPO / PPO Training Results Analysis and Visualization"""
from __future__ import annotations
from config import MODEL_DIR
from ppo.analysis.results import analyze_checkpoint, find_latest_checkpoint

def main() -> None:
 try:
 checkpoint = find_latest_checkpoint(MODEL_DIR)
 print(f" : {checkpoint}")
 analyze_checkpoint(checkpoint)
 except Exception as exc: # pragma: no cover - CLI feedback
 print("ERROR / Analysis failed")
 print(f" / Error: {exc}")
 raise

if __name__ == "__main__":
 main()
