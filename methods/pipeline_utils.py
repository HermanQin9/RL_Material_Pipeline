"""
Pipeline utilities for reinforcement learning and advanced pipeline operations.
流水线工具模块，用于强化学习和高级流水线操作。
"""

from typing import Dict, Any, Optional
from pipeline import run_pipeline  # Updated import for new structure


class PipelineAPI:
    """
    为后续强化学习实验自动搜索和评估模型性能设计的API接口
    API Interface for RL to search pipeline structures
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.last_results = None

    def reset(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        重置或覆盖部分 pipeline 参数
        Reset or override parts of the pipeline config
        """
        if config:
            self.config.update(config)
        self.last_results = None

    def step(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行一步 pipeline 并返回新结果
        Execute one pipeline run and return results (for RL step)
        """
        if config:
            self.config.update(config)
        self.last_results = run_pipeline(**self.config)
        return self.last_results

    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        查询最近一次 pipeline 运行的全部结果，包括所有核心变量
        Query the full results of the last pipeline run, including all core variables
        """
        results = self.last_results or {}
        keys_needed = [
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
            'feature_names', 'model', 'y_val_pred', 'y_test_pred'
        ]
        # 动态补齐每个 key，避免 KeyError
        full_results = {k: results.get(k, None) for k in keys_needed}
        # 其他原有内容保留
        full_results['pipeline_config'] = self.config
        return full_results
