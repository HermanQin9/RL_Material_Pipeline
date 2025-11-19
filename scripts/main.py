#!/usr/bin/env python3
"""
 / Main entry point for the project

This file provides a convenient way to run the main pipeline functionality.

"""

import argparse
import sys
from pathlib import Path

# Import main modules
from pipeline import run_pipeline
from scripts.train_ppo import main as train_ppo_main


def main():
 """ / Main function"""
 parser = argparse.ArgumentParser(description='Machine Learning Pipeline with PPO RL')
 parser.add_argument('--mode', choices=['pipeline', 'train-ppo', 'example'], 
 default='pipeline', help=' / Run mode')

 # Pipeline arguments
 parser.add_argument('--cache', action='store_true', default=True, 
 help=' / Use cached data')
 parser.add_argument('--model', default='rf', choices=['rf', 'gbr', 'lgbm', 'xgb', 'cat'],
 help=' / Machine learning model')

 args = parser.parse_args()

 if args.mode == 'pipeline':
 print("START / Running ML Pipeline")
 results = run_pipeline(
 cache=args.cache,
 model_strategy=args.model
 )
 print(f"SUCCESS R²: {results.get('r2_score', 'N/A')}")

 elif args.mode == 'train-ppo':
 print(" PPO / Training PPO RL Agent")
 # You would need to adapt train_ppo_main to work here
 print(": python scripts/train_ppo.py")

 elif args.mode == 'example':
 print(" / Running Example Usage")
 from scripts.example_usage import main as example_main
 example_main()


if __name__ == "__main__":
 main()
