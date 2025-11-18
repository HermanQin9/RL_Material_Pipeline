"""
Tests for ppo/buffer.py module
PPO经验回放缓冲区测试
"""

import pytest
import numpy as np
import torch
from ppo.buffer import RolloutBuffer


@pytest.mark.unit
class TestRolloutBufferInit:
    """Test buffer initialization"""
    
    def test_basic_init(self):
        """Test basic buffer creation"""
        buffer = RolloutBuffer(buffer_size=10, obs_dim=5, action_dim=3)
        
        assert buffer.buffer_size == 10
        assert buffer.obs_dim == 5
        assert buffer.action_dim == 3
        assert buffer.ptr == 0
        assert buffer.size == 0
        assert len(buffer) == 0
    
    def test_array_shapes(self):
        """Test internal array shapes"""
        buffer = RolloutBuffer(buffer_size=100, obs_dim=20, action_dim=10)
        
        assert buffer.observations.shape == (100, 20)
        assert buffer.actions.shape == (100, 10)
        assert buffer.rewards.shape == (100,)
        assert buffer.values.shape == (100,)
        assert buffer.log_probs.shape == (100,)
        assert buffer.dones.shape == (100,)
        assert buffer.advantages.shape == (100,)
        assert buffer.returns.shape == (100,)


@pytest.mark.unit
class TestBufferStore:
    """Test storing transitions"""
    
    def test_store_single_transition(self):
        """Test storing a single transition"""
        buffer = RolloutBuffer(buffer_size=5, obs_dim=3, action_dim=2)
        
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = np.array([0.5, 0.5], dtype=np.float32)
        
        buffer.store(obs, action, reward=1.0, value=0.8, log_prob=-0.5, done=False)
        
        assert buffer.ptr == 1
        assert buffer.size == 1
        assert len(buffer) == 1
        assert np.allclose(buffer.observations[0], obs)
        assert np.allclose(buffer.actions[0], action)
        assert buffer.rewards[0] == 1.0
        assert abs(buffer.values[0] - 0.8) < 1e-6
        assert abs(buffer.log_probs[0] - (-0.5)) < 1e-6
        assert buffer.dones[0] == False
    
    def test_store_multiple_transitions(self):
        """Test storing multiple transitions"""
        buffer = RolloutBuffer(buffer_size=10, obs_dim=2, action_dim=1)
        
        for i in range(5):
            obs = np.array([i, i+1], dtype=np.float32)
            action = np.array([i*0.1], dtype=np.float32)
            buffer.store(obs, action, reward=float(i), value=float(i)*0.9, 
                        log_prob=float(-i), done=(i==4))
        
        assert buffer.size == 5
        assert buffer.ptr == 5
        assert buffer.dones[4] == True
        assert buffer.dones[3] == False
    
    def test_buffer_full(self):
        """Test buffer full property"""
        buffer = RolloutBuffer(buffer_size=3, obs_dim=1, action_dim=1)
        
        assert not buffer.is_full
        
        for i in range(3):
            buffer.store(np.array([i], dtype=np.float32), 
                        np.array([i], dtype=np.float32),
                        reward=0.0, value=0.0, log_prob=0.0, done=False)
        
        assert buffer.is_full
        assert buffer.size == 3


@pytest.mark.unit
class TestComputeReturnsAndAdvantages:
    """Test GAE computation"""
    
    def test_basic_gae(self):
        """Test basic GAE computation"""
        buffer = RolloutBuffer(buffer_size=5, obs_dim=2, action_dim=1)
        
        # Store simple trajectory
        for i in range(5):
            obs = np.array([i, i], dtype=np.float32)
            action = np.array([0.5], dtype=np.float32)
            buffer.store(obs, action, reward=1.0, value=0.5, log_prob=-0.5, done=False)
        
        buffer.compute_returns_and_advantages(last_value=0.5, gamma=0.99, gae_lambda=0.95)
        
        # Check that advantages and returns are computed
        assert buffer.advantages.shape == (5,)
        assert buffer.returns.shape == (5,)
        assert not np.all(buffer.advantages[:buffer.size] == 0)
        assert not np.all(buffer.returns[:buffer.size] == 0)
    
    def test_gae_with_done(self):
        """Test GAE with episode termination"""
        buffer = RolloutBuffer(buffer_size=5, obs_dim=2, action_dim=1)
        
        for i in range(5):
            obs = np.array([i, i], dtype=np.float32)
            action = np.array([0.5], dtype=np.float32)
            done = (i == 2)  # Episode ends at step 2
            buffer.store(obs, action, reward=1.0, value=0.5, log_prob=-0.5, done=done)
        
        buffer.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        
        # Advantages should be normalized
        advantages = buffer.advantages[:buffer.size]
        assert abs(advantages.mean()) < 0.1  # Should be close to 0
        assert abs(advantages.std() - 1.0) < 0.1  # Should be close to 1
    
    def test_gae_empty_buffer(self):
        """Test GAE on empty buffer"""
        buffer = RolloutBuffer(buffer_size=10, obs_dim=2, action_dim=1)
        
        # Should not crash on empty buffer
        buffer.compute_returns_and_advantages(last_value=0.0)
        
        assert buffer.size == 0


@pytest.mark.unit
class TestGetBatch:
    """Test batch sampling"""
    
    def test_get_full_batch(self):
        """Test getting full batch"""
        buffer = RolloutBuffer(buffer_size=5, obs_dim=3, action_dim=2)
        
        # Fill buffer
        for i in range(5):
            buffer.store(
                np.array([i, i+1, i+2], dtype=np.float32),
                np.array([i*0.1, i*0.2], dtype=np.float32),
                reward=float(i), value=float(i)*0.9, log_prob=float(-i), done=False
            )
        
        buffer.compute_returns_and_advantages(last_value=0.0)
        
        batch = buffer.get_batch()
        
        assert isinstance(batch, dict)
        assert 'observations' in batch
        assert 'actions' in batch
        assert 'returns' in batch
        assert 'advantages' in batch
        assert 'log_probs' in batch
        assert 'values' in batch
        
        assert batch['observations'].shape == (5, 3)
        assert batch['actions'].shape == (5, 2)
        assert isinstance(batch['observations'], torch.Tensor)
    
    def test_get_smaller_batch(self):
        """Test getting smaller batch size"""
        buffer = RolloutBuffer(buffer_size=10, obs_dim=2, action_dim=1)
        
        for i in range(10):
            buffer.store(
                np.array([i, i], dtype=np.float32),
                np.array([i], dtype=np.float32),
                reward=0.0, value=0.0, log_prob=0.0, done=False
            )
        
        buffer.compute_returns_and_advantages(last_value=0.0)
        
        batch = buffer.get_batch(batch_size=5)
        
        assert batch['observations'].shape == (5, 2)
        assert batch['actions'].shape == (5, 1)
    
    def test_batch_randomness(self):
        """Test that batches are randomly sampled"""
        buffer = RolloutBuffer(buffer_size=10, obs_dim=1, action_dim=1)
        
        for i in range(10):
            buffer.store(
                np.array([i], dtype=np.float32),
                np.array([i], dtype=np.float32),
                reward=0.0, value=0.0, log_prob=0.0, done=False
            )
        
        buffer.compute_returns_and_advantages(last_value=0.0)
        
        batch1 = buffer.get_batch(batch_size=5)
        batch2 = buffer.get_batch(batch_size=5)
        
        # Batches should likely be different (not guaranteed but very likely)
        # Check at least one difference
        assert not torch.equal(batch1['observations'], batch2['observations']) or \
               not torch.equal(batch1['actions'], batch2['actions'])


@pytest.mark.unit
class TestBufferClear:
    """Test buffer clearing"""
    
    def test_clear_buffer(self):
        """Test clearing buffer"""
        buffer = RolloutBuffer(buffer_size=5, obs_dim=2, action_dim=1)
        
        # Fill buffer
        for i in range(5):
            buffer.store(
                np.array([i, i], dtype=np.float32),
                np.array([i], dtype=np.float32),
                reward=0.0, value=0.0, log_prob=0.0, done=False
            )
        
        assert buffer.size == 5
        assert buffer.ptr == 5
        
        buffer.clear()
        
        assert buffer.size == 0
        assert buffer.ptr == 0
        assert len(buffer) == 0
        assert not buffer.is_full


@pytest.mark.integration
class TestBufferIntegration:
    """Integration tests for buffer"""
    
    def test_complete_episode(self):
        """Test storing and processing complete episode"""
        buffer = RolloutBuffer(buffer_size=20, obs_dim=5, action_dim=3)
        
        # Simulate episode
        episode_length = 15
        for step in range(episode_length):
            obs = np.random.randn(5).astype(np.float32)
            action = np.random.randn(3).astype(np.float32)
            reward = np.random.rand()
            value = np.random.rand()
            log_prob = np.random.randn()
            done = (step == episode_length - 1)
            
            buffer.store(obs, action, reward, value, log_prob, done)
        
        # Compute GAE
        buffer.compute_returns_and_advantages(last_value=0.0)
        
        # Get batch
        batch = buffer.get_batch(batch_size=10)
        
        assert batch['observations'].shape[0] == 10
        assert not torch.isnan(batch['advantages']).any()
        assert not torch.isnan(batch['returns']).any()
        
        # Clear and reuse
        buffer.clear()
        assert buffer.size == 0


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
