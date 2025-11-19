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
        
        # Node selection head - 10 nodes (N0-N9)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 10)  # 10 nodes: N0-N9
        )
        
        # Method selection head - max 4 methods per node
        self.method_head = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 4)  # Max 4 methods across all nodes
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
        
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            obs: Flattened observation tensor [batch_size, obs_dim]
            action_mask: Optional action mask for invalid node actions [batch_size, num_nodes]
            
        Returns:
            node_logits: Node selection logits [batch_size, num_nodes]
            method_logits: Method selection logits [batch_size, max_methods]
            params: Parameter values [batch_size, param_dim]
            value: State value estimate [batch_size, 1]
        """
        # obs is already a flattened tensor from trainer
        features = self.shared_layers(obs)
        
        # Get node selection logits
        node_logits = self.node_head(features)
        
        # Apply action mask if provided (mask invalid nodes)
        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            node_logits = node_logits + (action_mask - 1.0) * 1e8
        
        # Get method selection logits
        method_logits = self.method_head(features)
        
        # Get parameters (single parameter value in [0, 1])
        params = torch.sigmoid(self.param_head(features))
        
        # Get state value
        value = self.value_head(features)
        
        return node_logits, method_logits, params, value

    # Note: The following methods are not currently used by the trainer
    # They would need to be updated to work with the new 4-output forward method
