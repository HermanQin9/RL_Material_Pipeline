"""
 - 
Custom exception classes for machine learning pipeline error handling
"""

class PipelineError(Exception):
 """"""
 pass

class DataValidationError(PipelineError):
 """"""
 pass

class ModelTrainingError(PipelineError):
 """"""
 pass

class ModelConfigurationError(PipelineError):
 """"""
 pass

class FeatureSelectionError(PipelineError):
 """"""
 pass

class DataProcessingError(PipelineError):
 """"""
 pass

class ImputationError(PipelineError):
 """"""
 pass

class ScalingError(PipelineError):
 """"""
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