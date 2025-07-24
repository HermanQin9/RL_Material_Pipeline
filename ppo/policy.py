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
        
        # Node selection head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 6)  # 6 nodes: N0-N5
        )
        
        # Method selection head  
        self.method_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 10)  # Max methods across all nodes
        )
        
        # Parameter head
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)  # Single parameter value
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        
    def forward(self, obs, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            obs: Observation (dict or tensor)
            action_mask: Optional action mask for invalid actions
            
        Returns:
            node_logits: Node selection logits
            method_logits: Method selection logits  
            params: Parameter values
            value: State value estimate
        """
        # Handle dict observation format
        if isinstance(obs, dict):
            # Flatten dictionary observation to tensor
            fingerprint = torch.FloatTensor(obs['fingerprint']) if isinstance(obs['fingerprint'], np.ndarray) else obs['fingerprint']
            node_visited = torch.FloatTensor(obs['node_visited']) if isinstance(obs['node_visited'], np.ndarray) else obs['node_visited']
            action_mask_tensor = torch.FloatTensor(obs['action_mask']) if isinstance(obs['action_mask'], np.ndarray) else obs['action_mask']
            
            # Concatenate all features
            obs_tensor = torch.cat([fingerprint, node_visited, action_mask_tensor])
        else:
            obs_tensor = obs
        
        features = self.shared_layers(obs_tensor)
        
        # Get node selection logits
        node_logits = self.node_head(features)
        
        # Get method selection logits (simplified - select from all methods)
        method_logits = self.method_head(features)
        
        # Get parameters (simplified - single parameter value)
        params = torch.sigmoid(self.param_head(features))
        
        # Get state value
        value = self.value_head(features)
        
        return node_logits, method_logits, params, value

    # Note: The following methods are not currently used by the trainer
    # They would need to be updated to work with the new 4-output forward method
