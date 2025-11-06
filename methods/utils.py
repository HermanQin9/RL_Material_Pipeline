# Utility functions for model training and experiment management
import os
import logging
import traceback
from datetime import datetime
from config import LOG_DIR

def setup_logger(name="model_training", level=logging.INFO):
 """Setup logging configuration for model training."""
 logger = logging.getLogger(name)
 if not logger.handlers: # Avoid adding handlers multiple times
 handler = logging.StreamHandler()
 formatter = logging.Formatter(
 '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
 )
 handler.setFormatter(formatter)
 logger.addHandler(handler)
 logger.setLevel(level)

 # Also log to file
 os.makedirs(LOG_DIR, exist_ok=True)
 file_handler = logging.FileHandler(
 os.path.join(LOG_DIR, f'model_training_{datetime.now().strftime("%Y%m%d")}.log')
 )
 file_handler.setFormatter(formatter)
 logger.addHandler(file_handler)

 return logger

def save_training_summary(results, output_path=None):
 """
 Save a summary of training results to a file.

 Args:
 results: Dictionary of training results from train_multiple_models
 output_path: Path to save the summary (optional)

 Returns:
 str: Path to the saved summary file
 """
 logger = logging.getLogger("model_training")

 try:
 if output_path is None:
 output_path = os.path.join(LOG_DIR, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

 os.makedirs(os.path.dirname(output_path), exist_ok=True)

 with open(output_path, 'w', encoding='utf-8') as f:
 f.write(f"Model Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
 f.write("=" * 60 + "\n\n")

 successful_models = [k for k, v in results.items() if v is not None]
 failed_models = [k for k, v in results.items() if v is None]

 f.write(f"Total models attempted: {len(results)}\n")
 f.write(f"Successful: {len(successful_models)}\n")
 f.write(f"Failed: {len(failed_models)}\n\n")

 if successful_models:
 f.write("Successful Models:\n")
 for model in successful_models:
 f.write(f" - {model}\n")
 f.write("\n")

 if failed_models:
 f.write("Failed Models:\n")
 for model in failed_models:
 f.write(f" - {model}\n")
 f.write("\n")

 f.write("Detailed Results:\n")
 for model_key, result in results.items():
 f.write(f"\n{model_key}:\n")
 if result is not None:
 f.write(f" Status: SUCCESS\n")
 f.write(f" Parameters: {result.get('params', 'N/A')}\n")
 if 'y_val_pred' in result and result['y_val_pred'] is not None:
 f.write(f" Validation predictions: {len(result['y_val_pred'])} samples\n")
 else:
 f.write(f" Status: FAILED\n")

 logger.info(f"Training summary saved to: {output_path}")
 return output_path

 except Exception as e:
 logger.error(f"Failed to save training summary: {e}")
 logger.error(f"Traceback: {traceback.format_exc()}")
 raise

def save_model_comparison(results, output_path=None):
 """
 Save a comparison of model performance to a file.

 Args:
 results: Dictionary of training results from train_multiple_models
 output_path: Path to save the comparison (optional)

 Returns:
 str: Path to the saved comparison file
 """
 logger = logging.getLogger("model_training")

 try:
 if output_path is None:
 output_path = os.path.join(LOG_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

 os.makedirs(os.path.dirname(output_path), exist_ok=True)

 with open(output_path, 'w', encoding='utf-8') as f:
 f.write(f"Model Performance Comparison - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
 f.write("=" * 70 + "\n\n")

 successful_results = {k: v for k, v in results.items() if v is not None}

 if not successful_results:
 f.write("No successful models to compare.\n")
 return output_path

 f.write(f"{'Model':<10} {'Parameters':<50} {'Val Samples':<12} {'Test Samples':<12}\n")
 f.write("-" * 86 + "\n")

 for model_key, result in successful_results.items():
 params_str = str(result.get('params', 'N/A'))[:47] + "..." if len(str(result.get('params', 'N/A'))) > 50 else str(result.get('params', 'N/A'))
 val_samples = len(result['y_val_pred']) if result.get('y_val_pred') is not None else 0
 test_samples = len(result['y_test_pred']) if result.get('y_test_pred') is not None else 0

 f.write(f"{model_key:<10} {params_str:<50} {val_samples:<12} {test_samples:<12}\n")

 logger.info(f"Model comparison saved to: {output_path}")
 return output_path

 except Exception as e:
 logger.error(f"Failed to save model comparison: {e}")
 logger.error(f"Traceback: {traceback.format_exc()}")
 raise
