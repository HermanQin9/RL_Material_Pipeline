#!/usr/bin/env python3
"""
项目主入口文件 / Main entry point for the project

This file provides a convenient way to run the main pipeline functionality.
此文件提供运行主要流水线功能的便捷方式。
"""

import argparse
import sys
from pathlib import Path

# Import main modules
from pipeline import run_pipeline
from scripts.train_ppo import main as train_ppo_main


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline with PPO RL')
    parser.add_argument('--mode', choices=['pipeline', 'train-ppo', 'example'], 
                       default='pipeline', help='运行模式 / Run mode')
    
    # Pipeline arguments
    parser.add_argument('--cache', action='store_true', default=True, 
                       help='使用缓存数据 / Use cached data')
    parser.add_argument('--model', default='rf', choices=['rf', 'gbr', 'lgbm', 'xgb', 'cat'],
                       help='机器学习模型 / Machine learning model')
    
    args = parser.parse_args()
    
    if args.mode == 'pipeline':
        print("🚀 运行机器学习流水线 / Running ML Pipeline")
        results = run_pipeline(
            cache=args.cache,
            model_strategy=args.model
        )
        print(f"✅ 流水线完成，R²得分: {results.get('r2_score', 'N/A')}")
        
    elif args.mode == 'train-ppo':
        print("🤖 训练PPO强化学习智能体 / Training PPO RL Agent")
        # You would need to adapt train_ppo_main to work here
        print("请直接运行: python scripts/train_ppo.py")
        
    elif args.mode == 'example':
        print("📚 运行示例用法 / Running Example Usage")
        from scripts.example_usage import main as example_main
        example_main()


if __name__ == "__main__":
    main()
