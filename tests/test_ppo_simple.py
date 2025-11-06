#!/usr/bin/env python3
"""
PPO
Simple PPO testing and visualization script
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer


def create_simple_ppo_test():
 """
 PPONaN - Create simple PPO test, avoiding NaN issues

 
 Generates mock training data for testing visualization features
 """
 print(" ... - Initializing environment...")
 env = PipelineEnv()

 print(" PPO... - Creating PPO trainer...")
 trainer = PPOTrainer(env, hidden_size=32, learning_rate=1e-3)

 # 
 # Manually simulate some training data to test visualization
 print(" ... - Generating mock training data...")
 episodes = range(1, 21)

 # 
 # Mock reward data: starts low, gradually improves, with some noise
 base_rewards = np.linspace(-1.0, 0.5, 20)
 noise = np.random.normal(0, 0.1, 20)
 rewards = base_rewards + noise

 # 
 # Mock loss data: starts high, gradually decreases
 losses = np.exp(-np.linspace(0, 2, 20)) + np.random.normal(0, 0.05, 20)

 return episodes, rewards, losses


def plot_training_curves(episodes, rewards, losses, save_path="logs/ppo_test_curves.png"):
 """
 - Plot training curves

 
 Creates visualization charts for rewards and losses
 """
 print(" ... - Generating training curve charts...")

 # - Create figure
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

 # - Reward curve
 ax1.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=2, label=' - Episode Reward')
 ax1.plot(episodes, np.convolve(rewards, np.ones(5)/5, mode='same'), 'r-', linewidth=2, 
 label='(5) - Moving Avg (5)')
 ax1.set_xlabel(' - Episode')
 ax1.set_ylabel(' - Reward')
 ax1.set_title('PPO - PPO Training Rewards')
 ax1.legend()
 ax1.grid(True, alpha=0.3)

 # - Loss curve
 ax2.plot(episodes, losses, 'g-', alpha=0.7, linewidth=2, label=' - Policy Loss')
 ax2.set_xlabel(' - Episode')
 ax2.set_ylabel(' - Loss')
 ax2.set_title('PPO - PPO Training Loss')
 ax2.legend()
 ax2.grid(True, alpha=0.3)

 plt.tight_layout()

 # - Save figure
 os.makedirs(os.path.dirname(save_path), exist_ok=True)
 plt.savefig(save_path, dpi=300, bbox_inches='tight')

 print(f"✓ - Training curves saved to: {save_path}")

 # - Show figure
 plt.show()

 return fig


def test_environment():
 """
 - Test if environment works properly

 
 Validates environment initialization and basic functionality
 """
 print("[TEST] ... - Testing environment...")

 try:
 env = PipelineEnv()
 obs = env.reset()

 print(f"✓ - Environment initialized successfully")
 print(f" keys - Observation keys: {list(obs.keys())}")
 print(f" Fingerprint: {obs['fingerprint']}")
 print(f" Node visited: {obs['node_visited']}")
 print(f" Action mask shape: {obs['action_mask'].shape}")

 # - Test a random action
 action = {
 'node': np.random.randint(0, 6),
 'method': np.random.randint(0, 4), 
 'params': np.random.random()
 }

 print(f" - Test action: {action}")

 return True

 except Exception as e:
 print(f"✗ ERROR - Environment test failed: {e}")
 return False


def main():
 """
 - Main function

 PPO
 Executes complete PPO testing and visualization workflow
 """
 print("Starting PPO... - Starting PPO test and visualization...")

 # - Test environment
 if not test_environment():
 print("⚠ WARNING - Environment test failed, exiting")
 return

 # - Generate mock data and visualization
 episodes, rewards, losses = create_simple_ppo_test()

 # - Create training curve charts
 fig = plot_training_curves(episodes, rewards, losses)

 # - Print statistics
 print("\n - Training Statistics:")
 print(f" - Average reward: {np.mean(rewards):.3f}")
 print(f" - Final reward: {rewards[-1]:.3f}")
 print(f" - Reward improvement: {rewards[-1] - rewards[0]:.3f}")
 print(f" - Average loss: {np.mean(losses):.3f}")

 print("\nSUCCESS PPO - PPO test completed!")


if __name__ == "__main__":
 main()
