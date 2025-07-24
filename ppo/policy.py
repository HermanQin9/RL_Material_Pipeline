"""
PPO Policy Network Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np


class PPOPolicy(nn.Module):
    """
    PPO Policy Network for Pipeline Optimization
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PPOPolicy, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            obs: Observation tensor
            action_mask: Optional action mask for invalid actions
            
        Returns:
            action_probs: Action probabilities
            state_value: State value estimate
        """
        features = self.shared_layers(obs)
        
        # Get action probabilities
        action_logits = self.policy_head(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits + (action_mask - 1) * 1e8
            
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get state value
        state_value = self.value_head(features)
        
        return action_probs, state_value
        
    def get_action(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[int, float, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            obs: Observation tensor
            action_mask: Optional action mask
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            state_value: State value estimate
        """
        with torch.no_grad():
            action_probs, state_value = self.forward(obs, action_mask)
            
            # Create distribution and sample
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return int(action.item()), float(log_prob.item()), state_value.squeeze()
            
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, 
                        action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training
        
        Args:
            obs: Observation tensor
            actions: Actions taken
            action_mask: Optional action mask
            
        Returns:
            log_probs: Log probabilities of actions
            state_values: State value estimates
            entropy: Policy entropy
        """
        action_probs, state_values = self.forward(obs, action_mask)
        
        # Create distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Get log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(), entropy
