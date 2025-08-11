"""
PPO Training Module for Pipeline Optimization
PPO强化学习训练模块，用于流水线优化，支持node_action和select_node操作
Includes learning_rate optimization and gradient updates for feature_selection
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from env.pipeline_env import PipelineEnv, create_random_action
from .policy import PPOPolicy

# 设置随机种子 / Set random seeds for node_action reproducibility
torch.manual_seed(0)
np.random.seed(0)

# 日志配置 / Logging config for select_node operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO训练器
    PPO Trainer for pipeline optimization
    """
    
    def __init__(
        self,
        env: PipelineEnv,
        policy_type: str = 'mlp',
        hidden_size: int = 64,
        learning_rate: float = 3e-4,  # Added learning_rate keyword
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_steps_per_episode: int = 20  # 每个训练episode包含的总步数（可跨多个环境回合）
    ):
        self.env = env

        # Calculate dimensions for node_action and select_node operations
        sample_obs = env.reset()
        obs_dim = self._calculate_obs_dim(sample_obs)
        action_dim = self._calculate_action_dim(env)

        # Initialize PPO policy with correct dimensions for node_action
        self.policy = PPOPolicy(obs_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)  # Use learning_rate

        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        # 固定每个训练episode的步数；若环境提前done（非法动作或完成N5），立即reset并继续收集
        self.max_steps_per_episode = max_steps_per_episode

        # 训练统计 / Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _calculate_obs_dim(self, obs: Dict[str, Any]) -> int:
        """计算观察空间维度 / Calculate observation dimension for node_action"""
        total_dim = 0
        for key, value in obs.items():
            if isinstance(value, (list, np.ndarray)):
                total_dim += len(value)
            else:
                total_dim += 1
        return total_dim
    
    def _calculate_action_dim(self, env: PipelineEnv) -> int:
        """计算动作空间维度 / Calculate action dimension for select_node operations"""
        # node selection + method selection + params
        node_dim = len(env.pipeline_nodes)
        max_method_dim = max(len(methods) for methods in env.methods_for_node.values()) 
        param_dim = env.hyperparam_dim
        return node_dim + max_method_dim + param_dim

    def select_action(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        选择动作
        Select action based on current observation
        """
        with torch.no_grad():
            node_logits, method_logits, params, value = self.policy(obs)
            
            # 节点选择 / Node selection
            node_dist = torch.distributions.Categorical(logits=node_logits)
            node_action = node_dist.sample()
            node_log_prob = node_dist.log_prob(node_action)
            
            # 方法选择 / Method selection
            node_idx = int(node_action.item())
            node_name = self.env.pipeline_nodes[node_idx]
            num_methods = len(self.env.methods_for_node[node_name])
            
            method_logits_masked = method_logits[:num_methods]
            method_dist = torch.distributions.Categorical(logits=method_logits_masked)
            method_action = method_dist.sample()
            method_log_prob = method_dist.log_prob(method_action)
            
            # 参数采样 / Parameter sampling
            param_dist = torch.distributions.Normal(params, 0.1)
            param_action = torch.clamp(param_dist.sample(), 0.0, 1.0)
            param_log_prob = param_dist.log_prob(param_action).sum()
            
            action = {
                'node': int(node_action.item()),
                'method': int(method_action.item()),
                'params': param_action.numpy().tolist()
            }
            
            log_probs = {
                'node': node_log_prob,
                'method': method_log_prob,
                'param': param_log_prob,
                'value': value
            }
            
            return action, log_probs

    def train_episode(self) -> Tuple[float, int]:
        """
        训练一个回合（固定步数收集策略）
        Train one training episode with a fixed number of steps. If the env
        reaches done early (illegal action or pipeline end), we reset and continue
        collecting until max_steps_per_episode is reached.
        """
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        # 收集轨迹 / Collect trajectory
        observations: List[Dict] = []
        actions: List[Dict] = []
        rewards: List[float] = []
        log_probs: List[Dict] = []
        values: List[torch.Tensor] = []
        dones: List[bool] = []

        for _ in range(self.max_steps_per_episode):
            # 选择动作 / Select action
            action, log_prob_dict = self.select_action(obs)

            # 环境步进 / Environment step
            next_obs, reward, done, _, _ = self.env.step(action)

            # 存储经验 / Store experience
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            log_probs.append(log_prob_dict)
            values.append(log_prob_dict['value'])
            dones.append(bool(done))

            episode_reward += float(reward)
            episode_length += 1

            # 如果完成则重置环境继续收集 / Reset on done and continue
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        # 计算回报 / Calculate returns
        returns = self._compute_returns(rewards, values, dones)

        # PPO更新 / PPO update
        self._update_policy(observations, actions, log_probs, returns, values)

        return episode_reward, episode_length

    def _compute_returns(self, rewards: List[float], values: List[torch.Tensor], dones: List[bool], gamma: float = 0.99) -> List[float]:
        """计算折扣回报"""
        returns = []
        R = 0.0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0.0
            R = rewards[i] + gamma * R
            returns.insert(0, R)
            
        return returns

    def _update_policy(self, observations: List[Dict], actions: List[Dict], old_log_probs: List[Dict], returns: List[float], old_values: List[torch.Tensor]):
        """PPO策略更新"""
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        old_values_tensor = torch.stack(old_values)
        
        advantages = returns_tensor - old_values_tensor.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新 / Multiple update epochs
        for _ in range(4):
            for i in range(len(observations)):
                obs = observations[i]
                action = actions[i]
                old_log_prob = old_log_probs[i]
                advantage = advantages[i]
                ret = returns_tensor[i]
                
                # 重新计算概率 / Recompute probabilities
                node_logits, method_logits, params, value = self.policy(obs)
                
                # 节点概率 / Node probability
                node_dist = torch.distributions.Categorical(logits=node_logits)
                node_log_prob = node_dist.log_prob(torch.tensor(action['node']))
                
                # 方法概率 / Method probability
                node_idx = int(action['node'])
                node_name = self.env.pipeline_nodes[node_idx]
                num_methods = len(self.env.methods_for_node[node_name])
                method_logits_masked = method_logits[:num_methods]
                method_dist = torch.distributions.Categorical(logits=method_logits_masked)
                method_log_prob = method_dist.log_prob(torch.tensor(action['method']))
                
                # 参数概率 / Parameter probability
                param_tensor = torch.tensor(action['params'], dtype=torch.float32)
                param_dist = torch.distributions.Normal(params, 0.1)
                param_log_prob = param_dist.log_prob(param_tensor).sum()
                
                # 总对数概率 / Total log probability
                new_log_prob = node_log_prob + method_log_prob + param_log_prob
                old_total_log_prob = old_log_prob['node'] + old_log_prob['method'] + old_log_prob['param']
                
                # PPO损失 / PPO loss
                ratio = torch.exp(new_log_prob - old_total_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                policy_loss = -torch.min(surr1, surr2)
                
                # 价值损失 / Value loss
                value_loss = F.mse_loss(value, ret)
                
                # 熵奖励 / Entropy bonus
                entropy = node_dist.entropy() + method_dist.entropy() + param_dist.entropy().sum()
                
                # 总损失 / Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播 / Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, num_episodes: int = 100, log_interval: int = 10):
        """
        训练主循环
        Main training loop
        """
        logger.info(f"开始PPO训练，共{num_episodes}轮")
        
        for episode in range(num_episodes):
            episode_reward, episode_length = self.train_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if episode % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                
                logger.info(f"回合 {episode}: 平均奖励={avg_reward:.3f}, 平均长度={avg_length:.1f}")

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
        logger.info(f"模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        logger.info(f"模型已从 {path} 加载")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PPO Pipeline Optimization')
    parser.add_argument('--episodes', type=int, default=100, help='训练回合数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--steps_per_ep', type=int, default=20, help='每个训练episode包含的固定步数')
    parser.add_argument('--save_path', type=str, default='models/ppo_pipeline.pth', help='模型保存路径')
    
    args = parser.parse_args()
    
    # 创建环境和训练器 / Create environment and trainer for node_action optimization
    env = PipelineEnv()
    trainer = PPOTrainer(
        env,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        max_steps_per_episode=args.steps_per_ep
    )  # Use learning_rate parameter
    
    # 开始训练 / Start training
    trainer.train(num_episodes=args.episodes)
    
    # 保存模型 / Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    trainer.save_model(args.save_path)


if __name__ == '__main__':
    main()
