"""
自定义异常类 - 用于机器学习流水线的错误处理
Custom exception classes for machine learning pipeline error handling
"""

class PipelineError(Exception):
    """流水线基础异常类"""
    pass

class DataValidationError(PipelineError):
    """数据验证异常"""
    pass

class ModelTrainingError(PipelineError):
    """模型训练异常"""
    pass

class ModelConfigurationError(PipelineError):
    """模型配置异常"""
    pass

class FeatureSelectionError(PipelineError):
    """特征选择异常"""
    pass

class DataProcessingError(PipelineError):
    """数据处理异常"""
    pass

class ImputationError(PipelineError):
    """数据填补异常"""
    pass

class ScalingError(PipelineError):
    """数据缩放异常"""
    pass

__all__ = [
    'PipelineError',
    'DataValidationError', 
    'ModelTrainingError',
    'ModelConfigurationError',
    'FeatureSelectionError',
    'DataProcessingError',
    'ImputationError',
    'ScalingError'
]