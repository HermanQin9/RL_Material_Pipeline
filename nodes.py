"""
流水线节点定义模块 / Pipeline Node Definition Module (10-Node Architecture)

This module defines the node classes for the 10-node ML pipeline.
本模块定义10节点机器学习流水线的节点类。

10-Node Pipeline:
N0: Data Fetch | N1: Imputation | N2: Feature Matrix | N3: Cleaning
N4: GNN Processing | N5: Knowledge Graph | N6: Feature Selection
N7: Scaling | N8: Model Training | N9: Termination
"""

from typing import Dict, Any
from methods.data_methods import (
    fetch_and_featurize,
    impute_data,
    feature_matrix,
    feature_selection,
    scale_features
)
from methods.data.preprocessing import clean_data, gnn_process, kg_process, terminate
from methods.model_methods import train_rf, train_gbr, train_lgbm, train_xgb, train_cat

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
        执行节点方法 / Execute node method
        
        Args:
            method: 方法名称 / Method name
            params: 方法参数 / Method parameters
            data: 输入数据 / Input data
            
        Returns:
            处理后的数据 / Processed data
        """
        return self.methods[method](data, **params)

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

# Node 3: Data Cleaning / 数据清洗

class CleaningNode(Node):
    """数据清洗节点 / Data cleaning node"""
    def __init__(self):
        """初始化数据清洗节点 / Initialize cleaning node"""
        methods = {'clean': clean_data}
        super().__init__('N3', 'Cleaning', 'DataProcessing', methods)

# Node 4: GNN Processing / 图神经网络处理

class GNNNode(Node):
    """GNN图神经网络节点 / GNN processing node"""
    def __init__(self):
        """初始化GNN节点 / Initialize GNN node"""
        methods = {'gnn': gnn_process}
        super().__init__('N4', 'GNN', 'GraphProcessing', methods)

# Node 5: Knowledge Graph / 知识图谱

class KnowledgeGraphNode(Node):
    """知识图谱节点 / Knowledge graph node"""
    def __init__(self):
        """初始化知识图谱节点 / Initialize KG node"""
        methods = {'kg': kg_process}
        super().__init__('N5', 'KnowledgeGraph', 'KnowledgeProcessing', methods)

# Node 6: Feature Selection / 特征选择

class FeatureSelectionNode(Node):
    """特征选择节点 / Feature selection node"""
    def __init__(self):
        """初始化特征选择节点 / Initialize feature selection node"""
        methods = {'select': feature_selection}
        super().__init__('N6', 'FeatureSelection', 'FeatureEngineering', methods)

# Node 7: Scaling features / 特征缩放

class ScalingNode(Node):
    """特征缩放节点 / Feature scaling node"""
    def __init__(self):
        """初始化特征缩放节点 / Initialize scaling node"""
        methods = {'scale': scale_features}
        super().__init__('N7', 'Scaling', 'Preprocessing', methods)

# Node 8: Model Training / 模型训练

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
        super().__init__('N8', 'ModelTraining', 'Training', methods)

    def execute(self, method, params, data):
        """
        执行模型训练 / Execute model training
        
        Map 'train' to the default algorithm or a user-specified algorithm.
        将'train'映射到默认算法或用户指定的算法。
        """
        # 将'train'映射到默认算法或用户指定算法 / Map 'train' to default or user-specified algorithm
        if method == 'train':
            algorithm = params.pop('algorithm', self.default_algorithm)
            method = algorithm
        return super().execute(method, params, data)

# Node 9: Termination / 终止节点

class TerminationNode(Node):
    """终止节点 / Termination node"""
    def __init__(self):
        """初始化终止节点 / Initialize termination node"""
        methods = {'terminate': terminate}
        super().__init__('N9', 'Termination', 'Control', methods)