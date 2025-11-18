"""
Methods package for the MatFormPPO pipeline.
Contains data processing and model training functions.

"""

# Data processing functions
from .data_methods import (
 fetch_and_featurize,
 impute_data,
 feature_matrix,
 feature_selection,
 scale_features,
 # Individual functions for specific strategies
 impute_mean,
 impute_median,
 impute_knn,
 no_selection,
 scale_standard,
 scale_robust,
 scale_minmax,
 scale_none,
 # Helper functions
 prepare_node_input,
 validate_state_keys,
 split_labels,
 update_state,
)

# Model training functions
from .model_methods import (
 train_rf,
 train_gbr,
 train_lgbm,
 train_xgb,
 train_cat,
 extract_search_param,
 fit_and_predict,
 # Analysis functions
 compute_metrics_and_sizes,
 print_results,
 save_pipeline_outputs,
)

__all__ = [
 # Data processing
 'fetch_and_featurize',
 'impute_data',
 'feature_matrix', 
 'feature_selection',
 'scale_features',
 'gnn_process', # N4 GNN
 'impute_mean',
 'impute_median',
 'impute_knn',
 'no_selection',
 'scale_standard',
 'scale_robust',
 'scale_minmax',
 'scale_none',
 # GNN related
 'structure_to_graph',
 'extract_gnn_features',
 'SimpleGCN',
 'SimpleGAT',
 'SimpleGraphSAGE',
 # Helper functions
 'prepare_node_input',
 'validate_state_keys',
 'split_labels',
 'update_state',
 # Model training
 'train_rf',
 'train_gbr',
 'train_lgbm',
 'train_xgb',
 'train_cat',
 'extract_search_param',
 'fit_and_predict',
 # Analysis functions
 'compute_metrics_and_sizes',
 'print_results',
 'save_pipeline_outputs',
]
