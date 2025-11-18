"""
Tests for ppo/utils.py module
PPO工具函数测试
"""

import pytest
import torch
import numpy as np
from ppo.utils import (
    compute_gae,
    ppo_loss,
    value_loss,
    entropy_loss,
    explained_variance,
    normalize_advantages,
    get_action_dim,
    validate_node_action
)


@pytest.mark.unit
class TestComputeGAE:
    """Test GAE computation"""
    
    def test_basic_gae(self):
        """Test basic GAE calculation"""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        next_value = 0.0
        
        advantages, returns = compute_gae(rewards, values, dones, next_value)
        
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert isinstance(advantages, torch.Tensor)
        assert isinstance(returns, torch.Tensor)
    
    def test_gae_with_terminal(self):
        """Test GAE with terminal state"""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 1.0])
        next_value = 0.5
        
        advantages, returns = compute_gae(rewards, values, dones, next_value, gamma=0.99)
        
        # Last advantage should not depend on next_value due to done=1
        assert advantages.shape[0] == 3
        assert not torch.isnan(advantages).any()
    
    def test_gae_parameters(self):
        """Test GAE with different parameters"""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 0.0])
        
        adv1, ret1 = compute_gae(rewards, values, dones, 0.0, gamma=0.99, gae_lambda=0.95)
        adv2, ret2 = compute_gae(rewards, values, dones, 0.0, gamma=0.95, gae_lambda=0.90)
        
        # Different parameters should give different results
        assert not torch.allclose(adv1, adv2)


@pytest.mark.unit
class TestPPOLoss:
    """Test PPO loss computation"""
    
    def test_no_clipping(self):
        """Test when ratio is within clip range"""
        new_log_probs = torch.tensor([0.0, 0.0, 0.0])
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        
        loss = ppo_loss(new_log_probs, old_log_probs, advantages, clip_param=0.2)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 0  # Loss should be negative (we negate for minimization)
    
    def test_with_clipping(self):
        """Test when ratio exceeds clip range"""
        new_log_probs = torch.tensor([1.0, 1.0, 1.0])  # Much higher
        old_log_probs = torch.tensor([-1.0, -1.0, -1.0])  # Much lower
        advantages = torch.tensor([1.0, 1.0, 1.0])
        
        loss = ppo_loss(new_log_probs, old_log_probs, advantages, clip_param=0.2)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    
    def test_negative_advantages(self):
        """Test with negative advantages"""
        new_log_probs = torch.tensor([0.0, 0.0])
        old_log_probs = torch.tensor([0.0, 0.0])
        advantages = torch.tensor([-1.0, -1.0])
        
        loss = ppo_loss(new_log_probs, old_log_probs, advantages)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


@pytest.mark.unit
class TestValueLoss:
    """Test value loss computation"""
    
    def test_basic_value_loss(self):
        """Test basic MSE value loss"""
        predicted = torch.tensor([0.5, 0.6, 0.7])
        target = torch.tensor([0.4, 0.5, 0.8])
        
        loss = value_loss(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # MSE is always non-negative
    
    def test_perfect_prediction(self):
        """Test value loss with perfect prediction"""
        predicted = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        
        loss = value_loss(predicted, target)
        
        assert loss.item() < 1e-6  # Should be very close to 0
    
    def test_large_error(self):
        """Test value loss with large prediction error"""
        predicted = torch.tensor([1.0, 1.0])
        target = torch.tensor([10.0, 10.0])
        
        loss = value_loss(predicted, target)
        
        assert loss.item() > 10.0  # Should be large


@pytest.mark.unit
class TestEntropyLoss:
    """Test entropy loss computation"""
    
    def test_basic_entropy(self):
        """Test basic entropy calculation"""
        log_probs = torch.tensor([[-0.5, -0.5], [-0.3, -0.7]])
        
        ent_loss = entropy_loss(log_probs)
        
        assert isinstance(ent_loss, torch.Tensor)
        assert ent_loss.item() >= 0  # Entropy is non-negative
    
    def test_uniform_distribution(self):
        """Test entropy with uniform distribution"""
        # Uniform distribution has maximum entropy
        log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        
        ent_loss = entropy_loss(log_probs)
        
        assert ent_loss.item() > 0


@pytest.mark.unit
class TestExplainedVariance:
    """Test explained variance"""
    
    def test_perfect_prediction(self):
        """Test explained variance with perfect prediction"""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        ev = explained_variance(y_pred, y_true)
        
        assert isinstance(ev, float)
        assert abs(ev - 1.0) < 0.01  # Should be close to 1
    
    def test_poor_prediction(self):
        """Test explained variance with poor prediction"""
        y_pred = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        ev = explained_variance(y_pred, y_true)
        
        assert ev < 0.5  # Should be low
    
    def test_constant_values(self):
        """Test with constant true values"""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([2.0, 2.0, 2.0])
        
        ev = explained_variance(y_pred, y_true)
        
        # Should handle division by zero gracefully
        assert not np.isnan(ev)
        assert not np.isinf(ev)


@pytest.mark.unit
class TestNormalizeAdvantages:
    """Test advantage normalization"""
    
    def test_basic_normalization(self):
        """Test basic normalization"""
        advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        normalized = normalize_advantages(advantages)
        
        assert normalized.shape == advantages.shape
        assert abs(normalized.mean().item()) < 0.1  # Mean should be ~0
        assert abs(normalized.std().item() - 1.0) < 0.1  # Std should be ~1
    
    def test_constant_advantages(self):
        """Test normalization with constant values"""
        advantages = torch.tensor([2.0, 2.0, 2.0, 2.0])
        
        normalized = normalize_advantages(advantages)
        
        # Should handle zero std gracefully
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


@pytest.mark.unit
class TestGetActionDim:
    """Test action dimension calculation"""
    
    def test_basic_action_space(self):
        """Test with basic action space"""
        action_space = {
            'node': ['N0', 'N2', 'N1', 'N3', 'N4', 'N5'],
            'method': {
                'N0': ['api'],
                'N2': ['construct'],
                'N1': ['mean', 'median'],
                'N3': ['none', 'variance'],
                'N4': ['std'],
                'N5': ['rf', 'gbr']
            }
        }
        
        action_dim = get_action_dim(action_space)
        
        # 6 nodes + 2 max methods + 1 param = 9
        assert action_dim == 9
    
    def test_empty_action_space(self):
        """Test with empty action space"""
        action_space = {'node': [], 'method': {}}
        
        # Empty action space should raise ValueError or return minimal dimension
        try:
            action_dim = get_action_dim(action_space)
            assert action_dim >= 1  # At least param dimension
        except ValueError:
            # Expected for empty method dict
            pass


@pytest.mark.unit
class TestValidateNodeAction:
    """Test node action validation"""
    
    def test_valid_action(self):
        """Test valid action"""
        action = {'node': 0, 'method': 0}
        action_space = {
            'node': ['N1', 'N2', 'N3'],
            'method': {'N1': ['method1', 'method2']}
        }
        
        is_valid = validate_node_action(action, action_space)
        
        assert is_valid is True
    
    def test_missing_keys(self):
        """Test action with missing keys"""
        action = {'node': 0}  # Missing 'method'
        action_space = {'node': ['N1'], 'method': {}}
        
        is_valid = validate_node_action(action, action_space)
        
        assert is_valid is False
    
    def test_invalid_node_index(self):
        """Test invalid node index"""
        action = {'node': 5, 'method': 0}
        action_space = {
            'node': ['N1', 'N2'],
            'method': {'N1': ['m1']}
        }
        
        is_valid = validate_node_action(action, action_space)
        
        assert is_valid is False
    
    def test_invalid_method_index(self):
        """Test invalid method index"""
        action = {'node': 0, 'method': 5}
        action_space = {
            'node': ['N1'],
            'method': {'N1': ['m1']}
        }
        
        is_valid = validate_node_action(action, action_space)
        
        assert is_valid is False
    
    def test_node_not_in_methods(self):
        """Test when node not in method dict"""
        action = {'node': 0, 'method': 0}
        action_space = {
            'node': ['N1', 'N2'],
            'method': {'N2': ['m1']}  # N1 not in methods
        }
        
        is_valid = validate_node_action(action, action_space)
        
        assert is_valid is False


@pytest.mark.integration
class TestPPOUtilsIntegration:
    """Integration tests for PPO utils"""
    
    def test_complete_ppo_step(self):
        """Test complete PPO training step"""
        batch_size = 10
        
        # Generate fake data
        rewards = torch.randn(batch_size)
        values = torch.randn(batch_size)
        dones = torch.zeros(batch_size)
        dones[-1] = 1.0
        
        # Compute GAE
        advantages, returns = compute_gae(rewards, values, dones, next_value=0.0)
        
        # Normalize advantages
        advantages_norm = normalize_advantages(advantages)
        
        # Compute losses
        new_log_probs = torch.randn(batch_size)
        old_log_probs = torch.randn(batch_size)
        
        policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages_norm)
        v_loss = value_loss(values, returns)
        ent_loss = entropy_loss(new_log_probs.unsqueeze(-1))
        
        # All should be valid tensors
        assert not torch.isnan(policy_loss)
        assert not torch.isnan(v_loss)
        assert not torch.isnan(ent_loss)
        
        # Compute total loss
        total_loss = policy_loss + 0.5 * v_loss - 0.01 * ent_loss
        assert not torch.isnan(total_loss)


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
