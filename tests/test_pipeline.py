#!/usr/bin/env python3
"""
Pipeline Testing Script

Tests basic pipeline execution, including node execution, state management, and data flow.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import logging
from typing import Dict, Any

from pipeline import run_pipeline
from nodes import (
    DataFetchNode,
    FeatureMatrixNode,
    ImputeNode,
    FeatureSelectionNode,
    ScalingNode,
    ModelTrainingNode
)
from env.pipeline_env import PipelineEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.slow
def test_basic_pipeline():
    """Test basic pipeline execution"""
    result = run_pipeline(
        cache=True,
        impute_strategy='mean',
        nan_thresh=0.5,
        train_val_ratio=0.8,
        selection_strategy='none',
        scaling_strategy='standard',
        model_strategy='rf',
        model_params={'n_estimators': 10}
    )

    # Verify results
    assert result is not None, "Pipeline should return result"
    assert 'model' in result, "Result should contain model"
    assert 'y_val_pred' in result, "Result should contain predictions"

    # Check prediction accuracy
    mae = result.get('mae')
    r2 = result.get('r2')
    
    if mae:
        assert mae >= 0, "MAE should be non-negative"
    if r2:
        assert -1 <= r2 <= 1, "R2 should be between -1 and 1"


@pytest.mark.integration
@pytest.mark.slow
def test_node_execution():
    """Test individual node execution"""
    # DataFetchNode
    n0 = DataFetchNode()
    data = n0.execute('api', {'cache': True}, {})
    assert 'X_train' in data, "DataFetchNode should return X_train"
    assert 'y_train' in data, "DataFetchNode should return y_train"

    # FeatureMatrixNode
    n2 = FeatureMatrixNode()
    features = n2.execute('construct', {
        'nan_thresh': 0.5,
        'train_val_ratio': 0.8,
        'verbose': False
    }, data)
    assert 'X_train' in features, "FeatureMatrixNode should return X_train"
    assert 'feature_names' in features, "FeatureMatrixNode should return feature_names"
    assert features['X_train'].shape[1] > 0, "Should have features"

    # ImputeNode
    n1 = ImputeNode()
    imputed = n1.execute('impute', {'strategy': 'mean'}, features)
    assert 'X_train' in imputed, "ImputeNode should return X_train"
    assert not np.isnan(imputed['X_train']).any(), "Should not have NaN values"

    # ScalingNode
    n4 = ScalingNode()
    scaled = n4.execute('scale', {'strategy': 'standard'}, imputed)
    assert 'X_train' in scaled, "ScalingNode should return X_train"
    assert 'scaler' in scaled, "ScalingNode should return scaler"
    
    # Check scaling worked
    mean = np.mean(scaled['X_train'])
    assert abs(mean) < 0.1, "Scaled data should have mean near 0"

    # ModelTrainingNode
    n5 = ModelTrainingNode()
    trained = n5.execute('train_rf', {'n_estimators': 10}, scaled)
    assert 'model' in trained, "ModelTrainingNode should return model"
    assert 'y_val_pred' in trained, "ModelTrainingNode should return predictions"


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_sequence():
    """Test flexible pipeline node sequence execution"""
    # Test with environment to verify flexible ordering
    env = PipelineEnv()
    
    assert env is not None
    assert len(env.pipeline_nodes) == 10
    
    # Verify nodes can be executed in flexible order
    obs = env.reset()
    assert obs is not None


@pytest.mark.integration
@pytest.mark.slow
def test_different_strategies():
    """Test different node strategies"""
    strategies = [
        {'impute': 'mean', 'scale': 'standard', 'model': 'rf'},
        {'impute': 'median', 'scale': 'robust', 'model': 'gbr'},
    ]
    
    for strategy in strategies:
        result = run_pipeline(
            cache=True,
            impute_strategy=strategy['impute'],
            scaling_strategy=strategy['scale'],
            model_strategy=strategy['model'],
            model_params={'n_estimators': 5}
        )
        
        assert result is not None, f"Pipeline should work with {strategy}"
        assert 'model' in result


@pytest.mark.integration
def test_pipeline_with_ppo_env():
    """Test pipeline integration with PPO environment"""
    env = PipelineEnv()
    assert env is not None, "Should create PipelineEnv"
    
    obs = env.reset()
    assert obs is not None, "Should return observation"
    assert 'node_visited' in obs, "Observation should contain node_visited"  # Changed from node_visits
    
    # Test that we have 10 nodes
    assert len(env.pipeline_nodes) == 10, "Should have 10 nodes"


@pytest.mark.integration
def test_error_handling():
    """Test error handling in pipeline"""
    # Test with invalid strategy
    with pytest.raises(Exception):
        run_pipeline(
            cache=True,
            impute_strategy='invalid_strategy',
            model_strategy='rf'
        )


@pytest.fixture
def sample_pipeline_data():
    """Fixture for sample pipeline data"""
    return {
        'X_train': np.random.rand(100, 10),
        'y_train': np.random.rand(100),
        'X_val': np.random.rand(20, 10),
        'y_val': np.random.rand(20)
    }


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
