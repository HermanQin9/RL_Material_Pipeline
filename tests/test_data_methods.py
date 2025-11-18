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
    """Test split by Fe content"""
    train, test = split_by_fe(dummy_dataframe)
    
    # Train should have Fe=0
    train_fe = [c.as_dict()['Fe'] for c in train['composition']]
    assert all(fe == 0 for fe in train_fe), "Train set should only contain Fe=0"
    
    # Test should have Fe=1
    test_fe = [c.as_dict()['Fe'] for c in test['composition']]
    assert all(fe == 1 for fe in test_fe), "Test set should only contain Fe=1"


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
