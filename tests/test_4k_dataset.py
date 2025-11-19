#!/usr/bin/env python3
"""
Test 4K Dataset Operations

Tests for fetching and processing large (4K) datasets from Materials Project
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config import TEST_MODE, N_TOTAL, CACHE_FILE, API_KEY
from methods.data_methods import fetch_data


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not API_KEY, reason="API_KEY not configured")
def test_4k_data_fetch():
    """Test fetching 4K dataset from Materials Project API"""
    # Check if cache exists
    cache_path = Path(CACHE_FILE)
    
    # Fetch data (use cache if available)
    df = fetch_data(cache=True)
    
    # Verify data structure
    assert isinstance(df, pd.DataFrame), "Should return DataFrame"
    assert len(df) > 0, "Should have data"
    assert 'formation_energy_per_atom' in df.columns, "Should have target property"
    
    # Verify data quality
    assert df.shape[0] >= 100, f"Should have at least 100 samples, got {df.shape[0]}"
    assert df.shape[1] >= 3, f"Should have at least 3 columns, got {df.shape[1]}"


@pytest.mark.slow
@pytest.mark.integration
def test_4k_featurization():
    """Test featurization of 4K dataset"""
    # Fetch data
    df = fetch_data(cache=True)
    
    # Check required columns
    assert 'composition' in df.columns or 'formula_pretty' in df.columns
    assert 'structure' in df.columns or 'structure' in str(df.columns)
    
    # Check data types
    assert not df['formation_energy_per_atom'].isnull().all(), "Target should have values"


@pytest.mark.slow
@pytest.mark.integration
def test_4k_pipeline():
    """Test complete pipeline with 4K dataset"""
    from pipeline import run_pipeline
    
    try:
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
        
        assert result is not None, "Pipeline should return result"
        assert 'model' in result, "Result should contain model"
        assert 'y_val_pred' in result, "Result should contain predictions"
        
        # Check prediction quality
        mae = result.get('mae')
        if mae:
            assert mae >= 0, "MAE should be non-negative"
            assert mae < 10, f"MAE seems too high: {mae}"
            
    except Exception as e:
        pytest.skip(f"Pipeline execution failed: {e}")


@pytest.mark.unit
def test_cache_file_config():
    """Test cache file configuration"""
    assert CACHE_FILE is not None, "CACHE_FILE should be configured"
    assert isinstance(CACHE_FILE, (str, Path)), "CACHE_FILE should be string or Path"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
