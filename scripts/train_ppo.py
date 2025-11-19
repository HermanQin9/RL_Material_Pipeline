"""
PPO / Main execution script for Pipeline PPO training

This script trains a PPO agent for automated pipeline optimization.
PPO
"""

import sys
import os
import argparse
from pathlib import Path

# / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ppo.trainer import PPOTrainer
from env.pipeline_env import PipelineEnv


def parse_args():
 """ / Parse command line arguments"""
 parser = argparse.ArgumentParser(description='PPO / Train PPO agent for pipeline optimization')
 parser.add_argument('--episodes', type=int, default=50, help=' / Number of training episodes')
 parser.add_argument('--save-freq', type=int, default=100, help='N / Save model every N episodes')
 parser.add_argument('--log-freq', type=int, default=50, help='N / Log progress every N episodes')
 parser.add_argument('--model-path', type=str, default='models/ppo_agent.pth', help='/ / Path to save/load model')
 parser.add_argument('--load-model', action='store_true', help=' / Load existing model')
 parser.add_argument('--eval-only', action='store_true', help=' / Only evaluate, no training')

 return parser.parse_args()


def main():
 """ / Main function"""
 args = parse_args()

 # Initialize environment
 env = PipelineEnv()

 # Initialize trainer
 trainer = PPOTrainer(env)

 # Load model if requested
 if args.load_model and os.path.exists(args.model_path):
     trainer.load_model(args.model_path)
     print(f"Loaded model from {args.model_path}")

 if args.eval_only:
     # Evaluation mode - run a few episodes to test
     print("Running evaluation...")
     trainer.train(num_episodes=5, log_interval=1)
 else:
     # Training mode
     print(f"Starting training for {args.episodes} episodes...")
     trainer.train(
         num_episodes=args.episodes,
         log_interval=args.log_freq
     )

     # Save model after training
     os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
     trainer.save_model(args.model_path)
     print(f"Model saved to {args.model_path}")

 print("Done!")


if __name__ == "__main__":
    main()
