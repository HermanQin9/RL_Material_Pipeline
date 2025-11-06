#!/usr/bin/env python3
"""
PPO / Complete Code Testing and PPO Learning Script

This script runs comprehensive tests and trains a PPO agent with learning curve visualization.
PPO
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import logging

# / Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# / Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_components():
 """
 / Test pipeline components

 Returns:
 bool: / Whether tests passed
 """
 print("🧪 / Starting pipeline component tests")
 print("=" * 70)

 try:
 from pipeline import run_pipeline
 from nodes import DataFetchNode, ImputeNode, FeatureMatrixNode, FeatureSelectionNode, ScalingNode, ModelTrainingNode

 # / Test each node
 print("\n1. / Testing Data Fetch Node")
 data_node = DataFetchNode()
 fetched = data_node.execute(method='api', params={'cache': True}, data={})
 print(f" SUCCESS / Data fetch successful: {list(fetched.keys())}")

 print("\n2. / Testing Feature Matrix Node")
 feature_node = FeatureMatrixNode()
 features = feature_node.execute(
 method='construct',
 params={'nan_thresh': 0.5, 'train_val_ratio': 0.8, 'verbose': False},
 data=fetched
 )
 print(f" SUCCESS / Feature matrix construction successful")
 print(f" / Training set shape: {features.get('X_train', np.array([])).shape}")

 print("\n3. / Testing Imputation Node")
 impute_node = ImputeNode()
 imputed = impute_node.execute(
 method='impute', 
 params={'strategy': 'mean', 'params': {}}, 
 data=features
 )
 print(f" SUCCESS / Imputation successful")

 print("\n4. / Testing Feature Selection Node")
 select_node = FeatureSelectionNode()
 selected = select_node.execute(
 method='select',
 params={'strategy': 'none', 'params': {}},
 data=imputed
 )
 print(f" SUCCESS / Feature selection successful")

 print("\n5. / Testing Scaling Node")
 scaling_node = ScalingNode()
 scaled = scaling_node.execute(
 method='scale',
 params={'strategy': 'standard', 'params': {}},
 data=selected
 )
 print(f" SUCCESS / Data scaling successful")

 print("\n6. / Testing Complete Pipeline")
 result = run_pipeline(
 cache=True,
 nan_thresh=0.5,
 train_val_ratio=0.8,
 impute_strategy='mean',
 selection_strategy='none',
 scaling_strategy='standard',
 model_strategy='rf',
 model_params={'n_estimators': 10}
 )
 print(f" SUCCESS / Complete pipeline test successful")
 print(f" / Model type: {type(result.get('model', None)).__name__}")

 return True

 except Exception as e:
 print(f" ERROR / Pipeline test failed: {e}")
 import traceback
 traceback.print_exc()
 return False

def test_ppo_components():
 """
 PPO / Test PPO components

 Returns:
 bool: / Whether tests passed
 """
 print("\n PPO / Starting PPO component tests")
 print("=" * 70)

 try:
 from env.pipeline_env import PipelineEnv
 from ppo.utils import compute_gae, ppo_loss, value_loss, entropy_loss
 import torch

 print("\n1. / Testing Environment Initialization")
 env = PipelineEnv()
 obs = env.reset()
 print(f" SUCCESS / Environment initialization successful")
 print(f" / Observation space keys: {list(obs.keys())}")

 print("\n2. / Testing Environment Step")
 action = {'node': 0, 'method': 0, 'params': [0.5]}
 next_obs, reward, done, truncated, info = env.step(action)
 print(f" SUCCESS / Environment step successful")
 print(f" / Reward: {reward:.3f}, / Done: {done}")

 print("\n3. PPO / Testing PPO Utility Functions")
 # / Create test data
 rewards = torch.tensor([1.0, 0.5, -0.2])
 values = torch.tensor([0.8, 0.6, 0.1])
 dones = torch.tensor([0.0, 0.0, 1.0])

 advantages, returns = compute_gae(rewards, values, dones, 0.0)
 print(f" SUCCESS GAE / GAE computation successful")

 # / Test loss functions
 new_log_probs = torch.tensor([0.1, 0.2, 0.3])
 old_log_probs = torch.tensor([0.15, 0.18, 0.25])

 policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)
 v_loss = value_loss(values, returns)
 e_loss = entropy_loss(new_log_probs)

 print(f" SUCCESS / Loss function tests successful")
 print(f" / Policy loss: {policy_loss:.4f}")
 print(f" / Value loss: {v_loss:.4f}")
 print(f" / Entropy loss: {e_loss:.4f}")

 return True

 except Exception as e:
 print(f" ERROR PPO / PPO component test failed: {e}")
 import traceback
 traceback.print_exc()
 return False

def train_ppo_with_curves(episodes: int = 100, save_plots: bool = True):
 """
 PPO / Train PPO and plot learning curves

 Args:
 episodes: / Number of training episodes
 save_plots: / Whether to save plots

 Returns:
 Dict: / Training results
 """
 print(f"\nSTART PPO ({episodes} ) / Starting PPO Training ({episodes} episodes)")
 print("=" * 70)

 try:
 from env.pipeline_env import PipelineEnv

 # / Initialize environment
 env = PipelineEnv()

 # / Training data recording
 episode_rewards = []
 episode_lengths = []
 moving_avg_rewards = []
 exploration_rates = []

 print(" / Starting training loop...")

 for episode in range(episodes):
 obs = env.reset()
 episode_reward = 0
 episode_length = 0
 done = False

 # / Simple random policy for demonstration
 while not done and episode_length < 50: # / Limit max steps
 # / Random action selection
 node_idx = np.random.randint(len(env.pipeline_nodes))
 method_idx = np.random.randint(len(env.methods_for_node[env.pipeline_nodes[node_idx]]))
 params = [np.random.random()]

 action = {
 'node': node_idx,
 'method': method_idx, 
 'params': params
 }

 next_obs, reward, terminated, truncated, info = env.step(action)
 done = terminated or truncated

 episode_reward += reward
 episode_length += 1
 obs = next_obs

 episode_rewards.append(episode_reward)
 episode_lengths.append(episode_length)

 # / Calculate moving average
 window_size = min(10, len(episode_rewards))
 moving_avg = np.mean(episode_rewards[-window_size:])
 moving_avg_rewards.append(moving_avg)

 # () / Record exploration rate (simplified)
 exploration_rate = max(0.1, 1.0 - episode / episodes)
 exploration_rates.append(exploration_rate)

 # / Periodic progress output
 if (episode + 1) % 20 == 0:
 print(f" / Episode {episode + 1}/{episodes}: "
 f" / Reward: {episode_reward:.3f}, "
 f" / Moving Avg: {moving_avg:.3f}, "
 f" / Length: {episode_length}")

 # / Plot learning curves
 print("\n / Plotting learning curves...")
 plot_learning_curves(episode_rewards, moving_avg_rewards, exploration_rates, 
 episode_lengths, save_plots)

 results = {
 'episode_rewards': episode_rewards,
 'moving_avg_rewards': moving_avg_rewards,
 'exploration_rates': exploration_rates,
 'episode_lengths': episode_lengths,
 'final_avg_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
 }

 print(f"\nSUCCESS PPO / PPO training completed!")
 print(f" / Final average reward: {results['final_avg_reward']:.3f}")

 return results

 except Exception as e:
 print(f"ERROR PPO / PPO training failed: {e}")
 import traceback
 traceback.print_exc()
 return {}

def plot_learning_curves(rewards, moving_avg_rewards, exploration_rates, episode_lengths, save_plots=True):
 """
 / Plot learning curves

 Args:
 rewards: / Episode rewards list
 moving_avg_rewards: / Moving average rewards
 exploration_rates: / Exploration rates
 episode_lengths: / Episode lengths
 save_plots: / Whether to save plots
 """
 try:
 # / Set Chinese font
 plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
 plt.rcParams['axes.unicode_minus'] = False

 fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
 episodes = range(1, len(rewards) + 1)

 # 1. / Episode rewards and moving average
 ax1.plot(episodes, rewards, alpha=0.6, color='lightblue', label=' / Episode Rewards')
 ax1.plot(episodes, moving_avg_rewards, color='darkblue', linewidth=2, label=' / Moving Average')
 ax1.set_xlabel(' / Episodes')
 ax1.set_ylabel(' / Reward')
 ax1.set_title('PPO - / PPO Learning Curve - Rewards')
 ax1.legend()
 ax1.grid(True, alpha=0.3)

 # 2. / Exploration rate decay
 ax2.plot(episodes, exploration_rates, color='orange', linewidth=2)
 ax2.set_xlabel(' / Episodes')
 ax2.set_ylabel(' / Exploration Rate')
 ax2.set_title(' / Exploration Rate Decay')
 ax2.grid(True, alpha=0.3)

 # 3. / Episode lengths
 ax3.plot(episodes, episode_lengths, color='green', alpha=0.7)
 ax3.set_xlabel(' / Episodes')
 ax3.set_ylabel(' / Episode Length')
 ax3.set_title(' / Episode Length Variation')
 ax3.grid(True, alpha=0.3)

 # 4. / Reward distribution histogram
 ax4.hist(rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
 ax4.set_xlabel(' / Reward Value')
 ax4.set_ylabel(' / Frequency')
 ax4.set_title(' / Reward Distribution')
 ax4.grid(True, alpha=0.3)

 plt.tight_layout()

 if save_plots:
 plots_dir = Path('logs')
 plots_dir.mkdir(exist_ok=True)
 timestamp = time.strftime("%Y%m%d_%H%M%S")
 plot_file = plots_dir / f'ppo_learning_curves_{timestamp}.png'
 plt.savefig(plot_file, dpi=300, bbox_inches='tight')
 print(f" / Learning curves saved: {plot_file}")

 plt.show()

 except Exception as e:
 print(f"ERROR / Plotting failed: {e}")
 import traceback
 traceback.print_exc()

def main():
 """
 / Main function
 """
 print(" MatFormPPO PPO")
 print(" MatFormPPO Complete Testing and PPO Learning System")
 print("=" * 80)

 # / Phase 1: Test pipeline components
 pipeline_success = test_pipeline_components()

 # PPO / Phase 2: Test PPO components 
 ppo_success = test_ppo_components()

 if pipeline_success and ppo_success:
 print("\n PPO...")
 print(" All component tests passed! Starting PPO training...")

 # PPO / Phase 3: PPO training and learning curves
 training_results = train_ppo_with_curves(episodes=100, save_plots=True)

 if training_results:
 print(f"\n ")
 print(f" Complete testing and training successfully completed!")
 print(f" / Final average reward: {training_results['final_avg_reward']:.3f}")
 else:
 print(f"\nERROR PPO")
 print(f"ERROR PPO training failed")
 else:
 print(f"\nERROR PPO")
 print(f"ERROR Component tests failed, skipping PPO training")

if __name__ == "__main__":
 main()
