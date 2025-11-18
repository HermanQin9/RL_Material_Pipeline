# -*- coding: utf-8 -*-
"""
Quick System Test - Pytest Version
Tests all components using pytest framework
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import warnings
warnings.filterwarnings('ignore')

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.quick
@pytest.mark.unit
def test_configuration():
    """Test configuration parameters"""
    from config import N_TOTAL, BATCH_SIZE, TARGET_PROP
    
    assert N_TOTAL == 400, f"Expected N_TOTAL=400, got {N_TOTAL}"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert TARGET_PROP == 'formation_energy_per_atom', "Unexpected TARGET_PROP"


@pytest.mark.quick
@pytest.mark.unit
def test_module_imports():
    """Test all critical module imports"""
    from env.pipeline_env import PipelineEnv
    from ppo.trainer import PPOTrainer
    from methods.data_methods import fetch_data, split_by_fe
    from methods.data.preprocessing import kg_process
    from nodes import (
        DataFetchNode, FeatureMatrixNode, ImputeNode,
        FeatureSelectionNode, ScalingNode, ModelTrainingNode
    )
    
    # All imports successful if no exception raised
    assert PipelineEnv is not None
    assert PPOTrainer is not None


@pytest.mark.quick
@pytest.mark.unit
def test_n5_knowledge_graph():
    """Test N5 Knowledge Graph with all strategies"""
    from methods.data.preprocessing import kg_process
    
    test_data = {
        'X_train': np.random.rand(20, 10),
        'y_train': np.random.rand(20),
        'X_val': np.random.rand(5, 10),
        'y_val': np.random.rand(5),
        'feature_names': [f'f{i}' for i in range(10)]
    }
    
    strategies = ['entity', 'relation', 'none']
    for strategy in strategies:
        result = kg_process(test_data, strategy=strategy)
        
        assert 'X_train' in result, f"X_train missing for {strategy}"
        assert 'X_val' in result, f"X_val missing for {strategy}"
        assert result['X_train'].shape[0] == 20, f"Wrong train size for {strategy}"
        assert result['X_val'].shape[0] == 5, f"Wrong val size for {strategy}"
        # KG adds 10 features to original 10
        assert result['X_train'].shape[1] >= test_data['X_train'].shape[1], f"KG should maintain or add features for {strategy}"


@pytest.mark.quick
@pytest.mark.integration
def test_environment_flexibility():
    """Test flexible pipeline environment"""
    from env.pipeline_env import PipelineEnv
    
    env = PipelineEnv()
    
    assert len(env.pipeline_nodes) == 10, "Should have 10 nodes"
    
    obs = env.reset()
    assert obs is not None, "Reset should return observation"
    assert 'node_visited' in obs or 'node_visits' in obs, "Observation should contain node tracking"
    
    action_mask = env._compute_action_mask()
    assert action_mask.sum() > 0, "Should have available actions"


@pytest.mark.quick
@pytest.mark.unit
def test_visualization_files():
    """Test existence of visualization files"""
    files = [
        'scripts/validate_rl_best_practices.py',
        'dashboard/app.py',
        'dash_app/plotly_dashboard.py'
    ]
    
    for path in files:
        full_path = project_root / path
        assert full_path.exists(), f"Missing file: {path}"
        
        # Check file is not empty
        content = full_path.read_text(encoding='utf-8')
        assert len(content) > 100, f"File {path} seems too small"


@pytest.fixture
def sample_pipeline_data():
    """Fixture providing sample data for pipeline tests"""
    return {
        'X_train': np.random.rand(50, 10),
        'y_train': np.random.rand(50),
        'X_val': np.random.rand(10, 10),
        'y_val': np.random.rand(10)
    }


if __name__ == "__main__":
    # Allow running directly with: python quick_test.py
    pytest.main([__file__, '-v', '--tb=short'])
