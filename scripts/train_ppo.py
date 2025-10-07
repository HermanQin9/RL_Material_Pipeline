"""
流水线PPO训练主执行脚本 / Main execution script for Pipeline PPO training

This script trains a PPO agent for automated pipeline optimization.
此脚本训练用于自动化流水线优化的PPO智能体。
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径 / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ppo.trainer import PPOTrainer
from env.pipeline_env import PipelineEnv


def parse_args():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(description='训练用于流水线优化的PPO智能体 / Train PPO agent for pipeline optimization')
    parser.add_argument('--episodes', type=int, default=50, help='训练回合数 / Number of training episodes')
    parser.add_argument('--save-freq', type=int, default=100, help='每N回合保存模型 / Save model every N episodes')
    parser.add_argument('--log-freq', type=int, default=50, help='每N回合记录进度 / Log progress every N episodes')
    parser.add_argument('--model-path', type=str, default='models/ppo_agent.pth', help='模型保存/加载路径 / Path to save/load model')
    parser.add_argument('--load-model', action='store_true', help='加载现有模型 / Load existing model')
    parser.add_argument('--eval-only', action='store_true', help='仅评估，不训练 / Only evaluate, no training')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
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
