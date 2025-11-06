"""
 / Pipeline Node Definition Module

This module defines the node classes for the machine learning pipeline.


Each node represents a step in the ML pipeline: data fetch, feature matrix construction,
imputation, feature selection, scaling, and model training.

"""

from typing import Dict, Any
from methods.data_methods import (
 fetch_and_featurize,
 impute_data,
 feature_matrix,
 feature_selection,
 scale_features
)
from methods.model_methods import train_rf, train_gbr, train_lgbm, train_xgb, train_cat
from methods.data.preprocessing import (
 clean_data,
 gnn_process,
 kg_process,
 terminate,
)
# GNN / Optional import of full GNN module
try:
 from methods.gnn_processing import gnn_process as gnn_process_full
 HAS_GNN_MODULE = True
except ImportError:
 HAS_GNN_MODULE = False

class Node:
 """ / Base pipeline node class"""
 def __init__(self, node_id, name, type_label, methods):
 """
 / Initialize node

 Args:
 node_id: ID / Node ID
 name: / Node name
 type_label: / Node type label
 methods: / Available methods dictionary
 """
 self.id = node_id
 self.name = name
 self.type = type_label
 self.methods = methods

 def execute(self, method, params, data):
 """
 dict{'error': ...}
 """
 import logging
 logger = logging.getLogger(__name__)
 try:
 logger.info(f"[Node:{self.name}] : {method}, keys: {list(data.keys())}")
 result = self.methods[method](data, **params)
 if not isinstance(result, dict):
 logger.warning(f"[Node:{self.name}] dict")
 result = {'output': result}
 logger.info(f"[Node:{self.name}] keys: {list(result.keys())}")
 return result
 except Exception as e:
 logger.error(f"[Node:{self.name}] {method}: {e}")
 return {'error': f'{self.name}.{method} failed: {e}'}

# ================= Core Nodes (N0-N2): Fixed Pipeline Start =================
# (N0-N2): / Core nodes: Fixed pipeline initialization

class DataFetchNode(Node):
 """
 (N0) / Data Fetch Node (N0)

 Fetches materials data from Materials Project API and performs initial featurization.
 Materials Project API

 Available Methods / :
 - 'api': Fetch data from Materials Project and extract basic features
 Materials Project

 Functionality / :
 - Downloads crystal structure and formation energy data

 - Caches data locally for faster subsequent runs

 - Splits into train/validation sets
 /

 Output / :
 - structures: Crystal structure objects / 
 - y_formation_energy: Target formation energies / 
 - material_ids: Materials Project IDs / Materials Project

 Note / :
 This is the first node in the 10-node architecture (always executed first).
 10
 """
 def __init__(self):
 """ / Initialize data fetch node"""
 methods = {'api': fetch_and_featurize}
 super().__init__('N0', 'DataFetch', 'FeatureEngineering', methods)


class ImputeNode(Node):
 """
 (N1) / Missing Data Imputation Node (N1)

 Handles missing values in the feature matrix using various imputation strategies.

 
 Available Methods / :
 - 'impute': Execute imputation strategy

 
 Imputation Strategies / :
 - 'mean': Replace missing values with column mean / 
 - 'median': Replace with column median (robust to outliers) / 
 - 'knn': K-Nearest Neighbors imputation / K

 Hyperparameters / :
 - param (float): Strategy-specific parameter [0.0-1.0]
 [0.0-1.0]
 - For 'knn': Controls number of neighbors (scaled to k=1-10)
 'knn': k=1-10

 Input / :
 - X_train, X_val: Feature matrices with potential missing values

 
 Output / :
 - X_train, X_val: Complete feature matrices (no missing values)

 """
 def __init__(self):
 """ / Initialize imputation node"""
 methods = {'impute': impute_data}
 super().__init__('N1', 'Impute', 'DataProcessing', methods)


class FeatureMatrixNode(Node):
 """
 (N2) / Feature Matrix Construction Node (N2)

 Constructs comprehensive feature matrix from crystal structures using materials informatics.

 
 Available Methods / :
 - 'construct': Build feature matrix from crystal structures

 
 Feature Engineering / :
 - Composition features (element ratios, atomic properties)

 - Structure features (lattice parameters, symmetry)

 - Electronic features (band gap estimates, oxidation states)

 
 Input / :
 - structures: Crystal structure objects from N0
 N0
 - y_formation_energy: Target labels

 
 Output / :
 - X_train, y_train: Training feature matrix and labels

 - X_val, y_val: Validation feature matrix and labels

 
 Note / :
 This is the second node in the 10-node architecture (always executed second).
 10
 Feature matrix MUST be constructed before imputation or other preprocessing.

 """
 def __init__(self):
 """ / Initialize feature matrix node"""
 methods = {'construct': feature_matrix}
 super().__init__('N2', 'FeatureMatrix', 'FeatureEngineering', methods)

# ================= Legacy Nodes (N3-N5): Old 6-Node Architecture =================
# (N3-N5): 6 / Legacy nodes from old 6-node architecture
# NOTE: These are kept for backward compatibility but replaced by new nodes in 10-node architecture
# : 10

class FeatureSelectionNode(Node):
 """
 (N3) / Feature Selection Node (Old N3)

 Legacy node for feature selection (replaced by N6 SelectionNode in 10-node architecture).
 10N6 SelectionNode

 Available Methods / :
 - 'select': Execute feature selection strategy

 
 Note / :
 Use SelectionNode (N6) for 10-node architecture pipelines.
 10SelectionNode (N6)
 """
 def __init__(self):
 """ / Initialize feature selection node"""
 methods = {'select': feature_selection}
 super().__init__('N3', 'FeatureSelection', 'FeatureEngineering', methods)


class ScalingNode(Node):
 """
 (N4) / Feature Scaling Node (Old N4)

 Legacy node for feature scaling (replaced by N7 ScalingNodeB in 10-node architecture).
 10N7 ScalingNodeB

 Available Methods / :
 - 'scale': Execute scaling strategy

 
 Note / :
 Use ScalingNodeB (N7) for 10-node architecture pipelines.
 10ScalingNodeB (N7)
 """
 def __init__(self):
 """ / Initialize scaling node"""
 methods = {'scale': scale_features}
 super().__init__('N4', 'Scaling', 'Preprocessing', methods)


class ModelTrainingNode(Node):
 """
 (N5) / Model Training Node (Old N5)

 Legacy node for model training (replaced by N8 ModelTrainingNodeB in 10-node architecture).
 10N8 ModelTrainingNodeB

 Available Methods / :
 - 'train_rf': Random Forest Regressor / 
 - 'train_gbr': Gradient Boosting Regressor / 
 - 'train_lgbm': LightGBM Regressor / LightGBM
 - 'train_xgb': XGBoost Regressor / XGBoost
 - 'train_cat': CatBoost Regressor / CatBoost

 Note / :
 Use ModelTrainingNodeB (N8) for 10-node architecture pipelines.
 10ModelTrainingNodeB (N8)
 """
 def __init__(self):
 """ / Initialize model training node"""
 methods = {'train_rf': train_rf,
 'train_gbr': train_gbr,
 'train_lgbm': train_lgbm,
 'train_xgb': train_xgb,
 'train_cat': train_cat}
 self.default_algorithm = 'train_rf' # / Default algorithm
 super().__init__('N5', 'ModelTraining', 'Training', methods)

 def execute(self, method, params, data):
 """

 """
 import logging
 logger = logging.getLogger(__name__)
 try:
 if method == 'train':
 algorithm = params.pop('algorithm', self.default_algorithm)
 method = algorithm
 logger.info(f"[Node:{self.name}] : {method}, keys: {list(data.keys())}")
 result = super().execute(method, params, data)
 if not isinstance(result, dict):
 logger.warning(f"[Node:{self.name}] dict")
 result = {'output': result}
 logger.info(f"[Node:{self.name}] keys: {list(result.keys())}")
 return result
 except Exception as e:
 logger.error(f"[Node:{self.name}] {method}: {e}")
 return {'error': f'{self.name}.{method} failed: {e}'}


# ================= Additional Nodes for 10-node architecture =================
# 10 / Extended nodes for 10-node flexible architecture
# These nodes enable PPO to explore millions of pipeline combinations
# PPO

class CleaningNode(Node):
 """
 (N3) / Data Cleaning Node (N3)

 Removes outliers and noise from the dataset before feature engineering.

 
 Available Methods / :
 - 'clean': Applies data cleaning based on strategy parameter

 
 Strategies / :
 - 'outlier': Remove statistical outliers / 
 - 'noise': Denoise using smoothing / 
 - 'none': Skip cleaning / 

 Hyperparameters / :
 - param (float): Threshold for outlier detection [0.0-1.0]
 [0.0-1.0]
 """
 def __init__(self):
 """ / Initialize cleaning node"""
 methods = {'clean': clean_data}
 super().__init__('N3', 'Cleaning', 'DataProcessing', methods)


class GNNNode(Node):
 """
 (N4) / Graph Neural Network Processing Node (N4)

 Applies graph neural networks to extract structural features from crystal graphs.

 
 Available Methods / :
 - 'process': Execute GNN-based feature extraction
 GNN

 GNN Architectures / GNN:
 - 'gcn': Graph Convolutional Network () / 
 - 'gat': Graph Attention Network / 
 - 'sage': GraphSAGE Network / GraphSAGE

 Hyperparameters / :
 - param (float): Controls GNN output dimension [0.0-1.0]
 GNN
 - 0.0-0.33: 8-dim embeddings ()
 - 0.33-0.67: 16-dim embeddings ()
 - 0.67-1.0: 32-dim embeddings ()

 Implementation / :
 GNN (methods/gnn_processing.py) 
 - SimpleGCN: 2
 - SimpleGAT: 
 - SimpleGraphSAGE: 
 - PyTorch

 Features / :

 GNN
 GPU

 

 Performance / :
 - GCN: (~50ms/)
 - GAT: (~80ms/)
 - GraphSAGE: (~40ms/)
 - : (<1ms/)

 Documentation / :
 : docs/GNN_IMPLEMENTATION.md
 """
 def __init__(self):
 """ / Initialize GNN node"""
 methods = {'process': gnn_process}
 super().__init__('N4', 'GNN', 'FeatureEngineering', methods)


class KGNode(Node):
 """
 (N5) / Knowledge Graph Processing Node (N5)

 Enriches features with domain knowledge from materials science knowledge graphs.

 
 Available Methods / :
 - 'process': Execute knowledge graph enrichment

 
 KG Strategies / :
 - 'entity': Entity-based knowledge extraction / 
 - 'relation': Relation-based knowledge extraction / 
 - 'none': Skip knowledge graph enrichment / 

 Note / :
 Currently placeholder implementation. Full KG integration to be developed.

 """
 def __init__(self):
 """ / Initialize knowledge graph node"""
 methods = {'process': kg_process}
 super().__init__('N5', 'KnowledgeGraph', 'FeatureEngineering', methods)


class SelectionNode(Node):
 """
 (N6) / Feature Selection Node (N6)

 Selects the most relevant features for model training to reduce dimensionality.

 
 Available Methods / :
 - 'select': Execute feature selection strategy

 
 Selection Strategies / :
 - 'variance': Variance threshold-based selection / 
 - 'univariate': Univariate statistical tests / 
 - 'pca': Principal Component Analysis / 

 Hyperparameters / :
 - param (float): Selection threshold or component ratio [0.0-1.0]
 [0.0-1.0]
 """
 def __init__(self):
 """ / Initialize feature selection node"""
 methods = {'select': feature_selection}
 super().__init__('N6', 'FeatureSelection', 'FeatureEngineering', methods)


class ScalingNodeB(Node):
 """
 (N7) / Feature Scaling Node (N7)

 Normalizes feature distributions to improve model convergence and performance.

 
 Available Methods / :
 - 'scale': Execute scaling strategy

 
 Scaling Strategies / :
 - 'std': StandardScaler (zero mean, unit variance) / 
 - 'robust': RobustScaler (median, IQR) / IQR
 - 'minmax': MinMaxScaler [0,1] normalization / [0,1] 

 Hyperparameters / :
 - param (float): Scaling parameter (strategy-dependent) [0.0-1.0]
 [0.0-1.0]
 """
 def __init__(self):
 """ / Initialize scaling node"""
 methods = {'scale': scale_features}
 super().__init__('N7', 'Scaling', 'Preprocessing', methods)


class ModelTrainingNodeB(Node):
 """
 (N8) / Model Training Node (N8)

 Trains machine learning models for formation energy prediction.

 
 Available Methods / :
 - 'train_rf': Random Forest Regressor / 
 - 'train_gbr': Gradient Boosting Regressor / 
 - 'train_lgbm': LightGBM Regressor / LightGBM
 - 'train_xgb': XGBoost Regressor / XGBoost
 - 'train_cat': CatBoost Regressor / CatBoost

 Hyperparameters / :
 - param (float): Model complexity parameter [0.0-1.0]
 [0.0-1.0]
 - algorithm (str): Specific ML algorithm to use

 
 Note / :
 This is the pre-end node in the 10-node architecture (always executed before N9).
 10N9
 """
 def __init__(self):
 """ / Initialize model training node"""
 methods = {'train_rf': train_rf,
 'train_gbr': train_gbr,
 'train_lgbm': train_lgbm,
 'train_xgb': train_xgb,
 'train_cat': train_cat}
 self.default_algorithm = 'train_rf' # / Default algorithm
 super().__init__('N8', 'ModelTraining', 'Training', methods)

 def execute(self, method, params, data):
 """
 / Execute model training with error handling and logging

 Supports dynamic algorithm selection through 'train' method dispatch.
 'train'

 Args:
 method (str): Training method name or 'train' for dynamic selection
 'train'
 params (dict): Training parameters including optional 'algorithm' key
 'algorithm'
 data (dict): Input data with X_train, y_train, X_val, y_val
 X_train, y_train, X_val, y_val

 Returns:
 dict: Training results including model, metrics, and predictions

 """
 import logging
 logger = logging.getLogger(__name__)
 try:
 # Handle 'train' method by dispatching to specific algorithm
 # 'train'
 if method == 'train':
 algorithm = params.pop('algorithm', self.default_algorithm)
 method = algorithm

 logger.info(f"[Node:{self.name}] / Executing method: {method}, keys / Input keys: {list(data.keys())}")
 result = super().execute(method, params, data)

 # Ensure output is dictionary format / 
 if not isinstance(result, dict):
 logger.warning(f"[Node:{self.name}] dict / Output not dict, auto-wrapping")
 result = {'output': result}

 logger.info(f"[Node:{self.name}] keys / Output keys: {list(result.keys())}")
 return result
 except Exception as e:
 logger.error(f"[Node:{self.name}] / Executing {method} / exception: {e}")
 return {'error': f'{self.name}.{method} failed: {e}'}


class EndNode(Node):
 """
 (N9) / Termination Node (N9)

 Marks the end of the pipeline execution and triggers reward computation.

 
 Available Methods / :
 - 'terminate': Finalize pipeline and compute final metrics

 
 Behavior / :
 - Signals episode termination to the RL environment
 RL
 - Triggers MAE, R² calculation for reward function
 MAER²
 - Updates fingerprint with final pipeline performance

 
 Note / :
 This is the final node in the 10-node architecture (always executed last).
 10
 """
 def __init__(self):
 """ / Initialize end node"""
 methods = {'terminate': terminate}
 super().__init__('N9', 'End', 'Control', methods)