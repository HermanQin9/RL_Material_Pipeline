"""
Tests for env/utils.py module
环境工具函数测试
"""

import pytest
import numpy as np
from env.utils import (
    validate_action,
    select_node_by_name,
    get_feature_selection_methods,
    is_feature_selection_node,
    create_node_action,
    compute_action_mask,
    compute_reward,
    get_observation_vector
)


@pytest.mark.unit
class TestValidateAction:
    """Test action validation"""
    
    def test_valid_action(self):
        """Test with valid action"""
        action = {'node': 0, 'method': 0, 'params': [0.5]}
        methods = {'N2': ['method1', 'method2'], 'N1': ['mean'], 'N3': ['none'], 'N4': ['std'], 'N5': ['rf']}
        assert validate_action(action, methods) is True
    
    def test_invalid_action_not_dict(self):
        """Test with non-dict action"""
        assert validate_action("not_a_dict", {}) is False
        assert validate_action(None, {}) is False
        assert validate_action([], {}) is False
    
    def test_missing_node_key(self):
        """Test action missing node key"""
        action = {'method': 0}
        assert validate_action(action, {}) is False
    
    def test_missing_method_key(self):
        """Test action missing method key"""
        action = {'node': 0}
        assert validate_action(action, {}) is False
    
    def test_invalid_node_type(self):
        """Test with non-integer node"""
        action = {'node': 'zero', 'method': 0}
        assert validate_action(action, {}) is False
    
    def test_invalid_method_type(self):
        """Test with non-integer method"""
        action = {'node': 0, 'method': 'zero'}
        assert validate_action(action, {}) is False
    
    def test_node_index_out_of_range(self):
        """Test node index out of range"""
        action = {'node': 5, 'method': 0}  # Max is 4
        methods = {'N2': ['m1']}
        assert validate_action(action, methods) is False
        
        action = {'node': -1, 'method': 0}
        assert validate_action(action, methods) is False
    
    def test_node_not_in_methods(self):
        """Test when node name not in methods dict"""
        action = {'node': 0, 'method': 0}
        methods = {}  # Empty methods
        assert validate_action(action, methods) is False
    
    def test_method_index_out_of_range(self):
        """Test method index out of range"""
        action = {'node': 0, 'method': 2}
        methods = {'N2': ['m1', 'm2']}  # Only 2 methods
        assert validate_action(action, methods) is False
        
        action = {'node': 0, 'method': -1}
        assert validate_action(action, methods) is False


@pytest.mark.unit
class TestSelectNodeByName:
    """Test node selection by name"""
    
    def test_existing_node(self):
        """Test selecting existing node"""
        pipeline = ['N0', 'N2', 'N1', 'N3', 'N4', 'N5', 'N8', 'N9']
        assert select_node_by_name('N0', pipeline) == 0
        assert select_node_by_name('N2', pipeline) == 1
        assert select_node_by_name('N9', pipeline) == 7
    
    def test_non_existing_node(self):
        """Test selecting non-existing node"""
        pipeline = ['N0', 'N2']
        assert select_node_by_name('N10', pipeline) == -1
        assert select_node_by_name('', pipeline) == -1


@pytest.mark.unit
class TestFeatureSelectionMethods:
    """Test feature selection utilities"""
    
    def test_get_methods(self):
        """Test getting feature selection methods"""
        methods = get_feature_selection_methods()
        assert isinstance(methods, list)
        assert 'none' in methods
        assert 'variance' in methods
        assert 'univariate' in methods
        assert 'pca' in methods
        assert len(methods) == 4
    
    def test_is_feature_selection_node(self):
        """Test identifying feature selection node"""
        assert is_feature_selection_node('N3') is True
        assert is_feature_selection_node('N0') is False
        assert is_feature_selection_node('N2') is False
        assert is_feature_selection_node('N4') is False


@pytest.mark.unit
class TestCreateNodeAction:
    """Test node action creation"""
    
    def test_basic_action(self):
        """Test creating basic action"""
        action = create_node_action(0, 1, 0.5)
        assert action['node'] == 0
        assert action['method'] == 1
        assert action['params'] == [0.5]
    
    def test_default_param(self):
        """Test with default parameter"""
        action = create_node_action(2, 3)
        assert action['node'] == 2
        assert action['method'] == 3
        assert action['params'] == [0.5]
    
    def test_custom_param(self):
        """Test with custom parameter"""
        action = create_node_action(1, 0, 0.8)
        assert action['params'] == [0.8]


@pytest.mark.unit
class TestComputeActionMask:
    """Test action mask computation"""
    
    def test_first_step_mask(self):
        """Test mask at first step - only N2 allowed"""
        mask = compute_action_mask([False] * 5, current_step=0, num_nodes=5)
        assert mask['node'][0] == True  # N2
        assert all(not mask['node'][i] for i in range(1, 5))
    
    def test_last_step_mask(self):
        """Test mask at last step - only N5 allowed"""
        mask = compute_action_mask([True, True, True, True, False], current_step=4, num_nodes=5)
        assert mask['node'][-1] == True  # N5
        assert all(not mask['node'][i] for i in range(4))
    
    def test_middle_step_mask(self):
        """Test mask in middle steps"""
        visited = [True, False, True, False, False]
        mask = compute_action_mask(visited, current_step=2, num_nodes=5)
        assert mask['node'][0] == False  # Visited
        assert mask['node'][1] == True   # Not visited
        assert mask['node'][2] == False  # Visited
        assert mask['node'][3] == True   # Not visited


@pytest.mark.unit
class TestComputeReward:
    """Test reward computation"""
    
    def test_positive_reward(self):
        """Test positive reward with good performance"""
        reward = compute_reward(mae=0.1, r2=0.9, num_features=10)
        assert isinstance(reward, float)
        assert reward > 0
    
    def test_negative_reward(self):
        """Test negative reward with poor performance"""
        reward = compute_reward(mae=1.0, r2=0.1, num_features=100)
        assert isinstance(reward, float)
        assert reward < 0
    
    def test_complexity_penalty(self):
        """Test complexity penalty effect"""
        reward1 = compute_reward(mae=0.1, r2=0.9, num_features=10)
        reward2 = compute_reward(mae=0.1, r2=0.9, num_features=100)
        assert reward1 > reward2  # More features = lower reward
    
    def test_custom_penalty(self):
        """Test custom complexity penalty"""
        reward1 = compute_reward(mae=0.1, r2=0.9, num_features=50, complexity_penalty=0.01)
        reward2 = compute_reward(mae=0.1, r2=0.9, num_features=50, complexity_penalty=0.1)
        assert reward1 > reward2


@pytest.mark.unit
class TestGetObservationVector:
    """Test observation vector construction"""
    
    def test_basic_observation(self):
        """Test basic observation vector"""
        fingerprint = np.array([0.1, 0.9, 50], dtype=np.float32)
        node_visited = [True, False, True, False, False]
        method_calls = {'method1': 2, 'method2': 1, 'method3': 0}
        
        obs = get_observation_vector(fingerprint, node_visited, method_calls, current_step=2)
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        # fingerprint (3) + node_visited (5) + method_calls (3) + step (1) = 12
        assert obs.shape[0] == 12
    
    def test_observation_normalization(self):
        """Test observation normalization"""
        fingerprint = np.array([0.5, 0.5, 100], dtype=np.float32)
        node_visited = [False] * 5
        method_calls = {'m1': 10, 'm2': 20, 'm3': 30}
        
        obs = get_observation_vector(fingerprint, node_visited, method_calls, current_step=3)
        
        # Check step normalization (3/5.0 = 0.6)
        assert abs(obs[-1] - 0.6) < 1e-5
    
    def test_zero_method_calls(self):
        """Test with zero method calls"""
        fingerprint = np.array([0.0, 1.0, 0], dtype=np.float32)
        node_visited = [False] * 3
        method_calls = {'m1': 0, 'm2': 0}
        
        obs = get_observation_vector(fingerprint, node_visited, method_calls, current_step=0)
        
        # Should handle division by zero
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
