"""
Pipeline utilities for reinforcement learning and advanced pipeline operations.

"""

from typing import Dict, Any, Optional
from pipeline import run_pipeline # Updated import for new structure


class PipelineAPI:
 """
 API
 API Interface for RL to search pipeline structures
 """

 def __init__(self, config: Optional[Dict[str, Any]] = None):
 self.config = config or {}
 self.last_results = None

 def reset(self, config: Optional[Dict[str, Any]] = None) -> None:
 """
 pipeline 
 Reset or override parts of the pipeline config
 """
 if config:
 self.config.update(config)
 self.last_results = None

 def step(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
 """
 pipeline 
 Execute one pipeline run and return results (for RL step)
 """
 if config:
 self.config.update(config)
 self.last_results = run_pipeline(**self.config)
 return self.last_results

 def get_pipeline_state(self) -> Dict[str, Any]:
 """
 pipeline 
 Query the full results of the last pipeline run, including all core variables
 """
 results = self.last_results or {}
 keys_needed = [
 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
 'feature_names', 'model', 'y_val_pred', 'y_test_pred'
 ]
 # key KeyError
 full_results = {k: results.get(k, None) for k in keys_needed}
 # 
 full_results['pipeline_config'] = self.config
 return full_results
