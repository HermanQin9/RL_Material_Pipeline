#!/usr/bin/env python3
"""
Test Data Methods

Tests for data fetching, preprocessing, and helper functions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pickle
import pandas as pd
import pytest
import numpy as np

from methods.data_methods import split_by_fe
from methods.model_methods import train_rf
from config import PROC_DIR, TARGET_PROP


class DummyComposition:
    """Picklable dummy composition for testing"""
    def __init__(self, fe: int):
        self._fe = int(fe)
    
    def as_dict(self):
        return {'Fe': self._fe}


@pytest.fixture
def dummy_dataframe():
    """Generate dummy DataFrame for testing"""
    n = 10
    data = {
        'composition': [DummyComposition(i % 2) for i in range(n)],
        'structure': [None] * n,
        TARGET_PROP: np.arange(n, dtype=float)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data():
    """Generate sample training data"""
    return {
        'X_train': np.random.rand(50, 10),
        'y_train': np.random.rand(50),
        'X_val': np.random.rand(10, 10),
        'y_val': np.random.rand(10),
        'feature_names': [f'feature_{i}' for i in range(10)]
    }


@pytest.mark.unit
def test_split_by_fe(dummy_dataframe):
    """Test split by Fe content - now uses configurable splitting strategy"""
    import os
    
    # Temporarily set environment variables for predictable testing
    old_n_in = os.environ.get('N_IN_DIST')
    old_n_out = os.environ.get('N_OUT_DIST')
    old_strategy = os.environ.get('SPLIT_STRATEGY')
    
    try:
        # Set test configuration: small dataset (10 samples)
        # Use 7 in-dist, 3 out-dist to fit available data
        os.environ['N_IN_DIST'] = '7'
        os.environ['N_OUT_DIST'] = '3'
        os.environ['SPLIT_STRATEGY'] = 'element_based'
        
        train, test = split_by_fe(dummy_dataframe)
        
        # Verify split sizes (may be adjusted for small dataset)
        assert len(train) > 0, "Train set should not be empty"
        assert len(test) > 0, "Test set should not be empty"
        assert len(train) + len(test) == len(dummy_dataframe), "All samples should be used"
        
        # Verify no overlap in actual data (check using formation energy values)
        # Since indices are reset, we check the actual data values
        train_fe_values = set(train[TARGET_PROP].values)
        test_fe_values = set(test[TARGET_PROP].values)
        # There should be no duplicate formation energy values in our test data
        assert len(train_fe_values.intersection(test_fe_values)) == 0, "Train and test should not share same formation energy values"
        
    finally:
        # Restore original environment variables
        if old_n_in is not None:
            os.environ['N_IN_DIST'] = old_n_in
        elif 'N_IN_DIST' in os.environ:
            del os.environ['N_IN_DIST']
            
        if old_n_out is not None:
            os.environ['N_OUT_DIST'] = old_n_out
        elif 'N_OUT_DIST' in os.environ:
            del os.environ['N_OUT_DIST']
            
        if old_strategy is not None:
            os.environ['SPLIT_STRATEGY'] = old_strategy
        elif 'SPLIT_STRATEGY' in os.environ:
            del os.environ['SPLIT_STRATEGY']


@pytest.mark.unit  
def test_train_rf(sample_data):
    """Test random forest training"""
    from nodes import ModelTrainingNode
    
    # Add required X_test and y_test for ModelTrainingNode
    sample_data_complete = sample_data.copy()
    sample_data_complete['X_test'] = np.random.rand(5, 10)
    sample_data_complete['y_test'] = np.random.rand(5)
    
    node = ModelTrainingNode()
    result = node.execute('train_rf', {'n_estimators': 10}, sample_data_complete)
    
    assert 'model' in result
    assert 'y_val_pred' in result
    assert len(result['y_val_pred']) == len(sample_data['y_val'])


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
