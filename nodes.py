"""
流水线节点定义模块 / Pipeline Node Definition Module

This module defines the node classes for the machine learning pipeline.
本模块定义机器学习流水线的节点类。

Each node represents a step in the ML pipeline: data fetch, feature matrix construction,
imputation, feature selection, scaling, and model training.
每个节点代表机器学习流水线中的一个步骤：数据获取、特征矩阵构建、缺失值填充、特征选择、数据缩放和模型训练。
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

class Node:
    """流水线节点基础类 / Base pipeline node class"""
    def __init__(self, node_id, name, type_label, methods):
        """
        初始化节点 / Initialize node
        
        Args:
            node_id: 节点ID / Node ID
            name: 节点名称 / Node name
            type_label: 节点类型标签 / Node type label
            methods: 节点可用方法字典 / Available methods dictionary
        """
        self.id = node_id
        self.name = name
        self.type = type_label
        self.methods = methods

    def execute(self, method, params, data):
        """
        执行节点方法，增加异常捕获和日志，所有输出为dict，异常时返回{'error': ...}
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"[Node:{self.name}] 执行方法: {method}, 输入keys: {list(data.keys())}")
            result = self.methods[method](data, **params)
            if not isinstance(result, dict):
                logger.warning(f"[Node:{self.name}] 输出非dict，自动包装")
                result = {'output': result}
            logger.info(f"[Node:{self.name}] 输出keys: {list(result.keys())}")
            return result
        except Exception as e:
            logger.error(f"[Node:{self.name}] 执行{method}异常: {e}")
            return {'error': f'{self.name}.{method} failed: {e}'}

# Node 0: 获取数据、特征化并划分训练/测试集 / Node 0: Fetch data, featurize, and split into train/test sets
class DataFetchNode(Node):
    """数据获取节点 / Data fetch node"""
    def __init__(self):
        """初始化数据获取节点 / Initialize data fetch node"""
        methods = {'api': fetch_and_featurize}
        super().__init__('N0', 'DataFetch', 'FeatureEngineering', methods)

# Node 1: Impute missing data | Node 1: 缺失值填充

class ImputeNode(Node):
    """缺失值填充节点 / Missing data imputation node"""
    def __init__(self):
        """初始化缺失值填充节点 / Initialize imputation node"""
        methods = {'impute': impute_data}
        super().__init__('N1', 'Impute', 'DataProcessing', methods)

# Node 2 : Construct feature matrix | Node 2: 构建特征矩阵

class FeatureMatrixNode(Node):
    """特征矩阵构建节点 / Feature matrix construction node"""
    def __init__(self):
        """初始化特征矩阵节点 / Initialize feature matrix node"""
        methods = {'construct': feature_matrix}
        super().__init__('N2', 'FeatureMatrix', 'FeatureEngineering', methods)

# Node 3: Feature Selection / 特征选择

class FeatureSelectionNode(Node):
    """特征选择节点 / Feature selection node"""
    def __init__(self):
        """初始化特征选择节点 / Initialize feature selection node"""
        methods = {'select': feature_selection}
        super().__init__('N3', 'FeatureSelection', 'FeatureEngineering', methods)

# Node 4 : Scaling features / 特征缩放

class ScalingNode(Node):
    """特征缩放节点 / Feature scaling node"""
    def __init__(self):
        """初始化特征缩放节点 / Initialize scaling node"""
        methods = {'scale': scale_features}
        super().__init__('N4', 'Scaling', 'Preprocessing', methods)

# Node 5: Model Training / 模型训练

class ModelTrainingNode(Node):
    """模型训练节点 / Model training node"""
    def __init__(self):
        """初始化模型训练节点 / Initialize model training node"""
        methods = {'train_rf': train_rf,
                   'train_gbr': train_gbr,
                   'train_lgbm': train_lgbm,
                   'train_xgb': train_xgb,
                   'train_cat': train_cat}
        self.default_algorithm = 'train_rf'  # 默认算法 / Default algorithm
        super().__init__('N5', 'ModelTraining', 'Training', methods)

    def execute(self, method, params, data):
        """
        执行模型训练，增加异常捕获和日志。
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            if method == 'train':
                algorithm = params.pop('algorithm', self.default_algorithm)
                method = algorithm
            logger.info(f"[Node:{self.name}] 执行方法: {method}, 输入keys: {list(data.keys())}")
            result = super().execute(method, params, data)
            if not isinstance(result, dict):
                logger.warning(f"[Node:{self.name}] 输出非dict，自动包装")
                result = {'output': result}
            logger.info(f"[Node:{self.name}] 输出keys: {list(result.keys())}")
            return result
        except Exception as e:
            logger.error(f"[Node:{self.name}] 执行{method}异常: {e}")
            return {'error': f'{self.name}.{method} failed: {e}'}


# ================= Additional Nodes for 10-node architecture =================

class CleaningNode(Node):
    """N3 Cleaning: outlier/noise/none"""
    def __init__(self):
        methods = {'clean': clean_data}
        super().__init__('N3', 'Cleaning', 'DataProcessing', methods)


class GNNNode(Node):
    """N4 GNN processing (placeholder)"""
    def __init__(self):
        methods = {'process': gnn_process}
        super().__init__('N4', 'GNN', 'FeatureEngineering', methods)


class KGNode(Node):
    """N5 Knowledge Graph processing (placeholder)"""
    def __init__(self):
        methods = {'process': kg_process}
        super().__init__('N5', 'KnowledgeGraph', 'FeatureEngineering', methods)


class SelectionNode(Node):
    """N6 Feature Selection (variance/univariate/pca)"""
    def __init__(self):
        methods = {'select': feature_selection}
        super().__init__('N6', 'FeatureSelection', 'FeatureEngineering', methods)


class ScalingNodeB(Node):
    """N7 Scaling (std/robust/minmax)"""
    def __init__(self):
        methods = {'scale': scale_features}
        super().__init__('N7', 'Scaling', 'Preprocessing', methods)


class ModelTrainingNodeB(Node):
    """N8 Model Training (rf/gbr/xgb/cat)"""
    def __init__(self):
        methods = {'train_rf': train_rf,
                   'train_gbr': train_gbr,
                   'train_lgbm': train_lgbm,
                   'train_xgb': train_xgb,
                   'train_cat': train_cat}
        self.default_algorithm = 'train_rf'
        super().__init__('N8', 'ModelTraining', 'Training', methods)

    def execute(self, method, params, data):
        import logging
        logger = logging.getLogger(__name__)
        try:
            if method == 'train':
                algorithm = params.pop('algorithm', self.default_algorithm)
                method = algorithm
            logger.info(f"[Node:{self.name}] 执行方法: {method}, 输入keys: {list(data.keys())}")
            result = super().execute(method, params, data)
            if not isinstance(result, dict):
                logger.warning(f"[Node:{self.name}] 输出非dict，自动包装")
                result = {'output': result}
            logger.info(f"[Node:{self.name}] 输出keys: {list(result.keys())}")
            return result
        except Exception as e:
            logger.error(f"[Node:{self.name}] 执行{method}异常: {e}")
            return {'error': f'{self.name}.{method} failed: {e}'}


class EndNode(Node):
    """N9 Termination (no-op)"""
    def __init__(self):
        methods = {'terminate': terminate}
        super().__init__('N9', 'End', 'Control', methods)