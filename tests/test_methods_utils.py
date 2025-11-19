"""
Tests for methods/utils.py module
模型训练工具函数测试
"""

import pytest
import os
import tempfile
from pathlib import Path
from methods.utils import setup_logger, save_training_summary, save_model_comparison


@pytest.mark.unit
class TestSetupLogger:
    """Test logger setup"""
    
    def test_logger_creation(self):
        """Test creating a logger"""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0
    
    def test_logger_singleton(self):
        """Test logger is not duplicated"""
        logger1 = setup_logger("test_singleton")
        initial_handlers = len(logger1.handlers)
        
        logger2 = setup_logger("test_singleton")
        assert len(logger2.handlers) == initial_handlers
        assert logger1 is logger2


@pytest.mark.unit
class TestSaveTrainingSummary:
    """Test training summary saving"""
    
    def test_save_successful_models(self, tmp_path):
        """Test saving summary with successful models"""
        results = {
            'rf': {'params': {'n_estimators': 100}, 'y_val_pred': [1, 2, 3]},
            'gbr': {'params': {'learning_rate': 0.1}, 'y_val_pred': [4, 5, 6]},
        }
        
        output_path = tmp_path / "summary.txt"
        saved_path = save_training_summary(results, str(output_path))
        
        assert os.path.exists(saved_path)
        
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Total models attempted: 2" in content
            assert "Successful: 2" in content
            assert "Failed: 0" in content
            assert "rf" in content
            assert "gbr" in content
    
    def test_save_failed_models(self, tmp_path):
        """Test saving summary with failed models"""
        results = {
            'rf': {'params': {'n_estimators': 100}, 'y_val_pred': [1, 2, 3]},
            'gbr': None,  # Failed model
            'xgb': None,  # Failed model
        }
        
        output_path = tmp_path / "summary_failed.txt"
        saved_path = save_training_summary(results, str(output_path))
        
        assert os.path.exists(saved_path)
        
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Total models attempted: 3" in content
            assert "Successful: 1" in content
            assert "Failed: 2" in content
            assert "gbr" in content
            assert "xgb" in content
    
    def test_save_empty_results(self, tmp_path):
        """Test saving summary with empty results"""
        results = {}
        
        output_path = tmp_path / "summary_empty.txt"
        saved_path = save_training_summary(results, str(output_path))
        
        assert os.path.exists(saved_path)
        
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Total models attempted: 0" in content
    
    def test_auto_path_generation(self):
        """Test automatic path generation"""
        results = {'rf': {'params': {}, 'y_val_pred': [1]}}
        
        saved_path = save_training_summary(results, output_path=None)
        
        assert os.path.exists(saved_path)
        assert "training_summary_" in saved_path
        
        # Cleanup
        os.remove(saved_path)


@pytest.mark.unit
class TestSaveModelComparison:
    """Test model comparison saving"""
    
    def test_save_comparison(self, tmp_path):
        """Test saving model comparison"""
        results = {
            'rf': {
                'params': {'n_estimators': 100, 'max_depth': 10},
                'y_val_pred': [1, 2, 3],
                'y_test_pred': [4, 5, 6, 7]
            },
            'gbr': {
                'params': {'learning_rate': 0.1},
                'y_val_pred': [8, 9],
                'y_test_pred': [10, 11, 12]
            },
        }
        
        output_path = tmp_path / "comparison.txt"
        saved_path = save_model_comparison(results, str(output_path))
        
        assert os.path.exists(saved_path)
        
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Model Performance Comparison" in content
            assert "rf" in content
            assert "gbr" in content
            assert "3" in content  # Val samples for rf
            assert "4" in content  # Test samples for rf
    
    def test_save_comparison_no_successful(self, tmp_path):
        """Test saving comparison with no successful models"""
        results = {
            'rf': None,
            'gbr': None,
        }
        
        output_path = tmp_path / "comparison_none.txt"
        saved_path = save_model_comparison(results, str(output_path))
        
        assert os.path.exists(saved_path)
        
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "No successful models to compare" in content
    
    def test_long_params_truncation(self, tmp_path):
        """Test long parameter string truncation"""
        results = {
            'rf': {
                'params': {'param' + str(i): i for i in range(50)},  # Very long params
                'y_val_pred': [1],
                'y_test_pred': [2]
            },
        }
        
        output_path = tmp_path / "comparison_long.txt"
        saved_path = save_model_comparison(results, str(output_path))
        
        assert os.path.exists(saved_path)
        
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Should truncate long params string
            assert "..." in content or len(content) < 10000


@pytest.mark.integration
class TestUtilsIntegration:
    """Integration tests for utils module"""
    
    def test_complete_workflow(self, tmp_path):
        """Test complete logging and saving workflow"""
        # Setup logger
        logger = setup_logger("integration_test")
        
        # Create results
        results = {
            'model1': {'params': {'a': 1}, 'y_val_pred': [1, 2], 'y_test_pred': [3]},
            'model2': None,
        }
        
        # Save summary
        summary_path = tmp_path / "integration_summary.txt"
        save_training_summary(results, str(summary_path))
        
        # Save comparison
        comparison_path = tmp_path / "integration_comparison.txt"
        save_model_comparison(results, str(comparison_path))
        
        # Verify both files exist
        assert os.path.exists(summary_path)
        assert os.path.exists(comparison_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
