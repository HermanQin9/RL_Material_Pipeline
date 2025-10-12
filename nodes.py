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

# ================= Core Nodes (N0-N2): Fixed Pipeline Start =================
# 核心节点 (N0-N2): 固定的流水线起始部分 / Core nodes: Fixed pipeline initialization

class DataFetchNode(Node):
    """
    数据获取节点 (N0) / Data Fetch Node (N0)
    
    Fetches materials data from Materials Project API and performs initial featurization.
    从Materials Project API获取材料数据并执行初始特征化。
    
    Available Methods / 可用方法:
        - 'api': Fetch data from Materials Project and extract basic features
          从Materials Project获取数据并提取基本特征
    
    Functionality / 功能:
        - Downloads crystal structure and formation energy data
          下载晶体结构和形成能数据
        - Caches data locally for faster subsequent runs
          本地缓存数据以加快后续运行
        - Splits into train/validation sets
          划分为训练/验证集
    
    Output / 输出:
        - structures: Crystal structure objects / 晶体结构对象
        - y_formation_energy: Target formation energies / 目标形成能
        - material_ids: Materials Project IDs / Materials Project标识符
    
    Note / 注意:
        This is the first node in the 10-node architecture (always executed first).
        这是10节点架构中的第一个节点（总是首先执行）。
    """
    def __init__(self):
        """初始化数据获取节点 / Initialize data fetch node"""
        methods = {'api': fetch_and_featurize}
        super().__init__('N0', 'DataFetch', 'FeatureEngineering', methods)


class ImputeNode(Node):
    """
    缺失值填充节点 (N1) / Missing Data Imputation Node (N1)
    
    Handles missing values in the feature matrix using various imputation strategies.
    使用各种填充策略处理特征矩阵中的缺失值。
    
    Available Methods / 可用方法:
        - 'impute': Execute imputation strategy
          执行填充策略
    
    Imputation Strategies / 填充策略:
        - 'mean': Replace missing values with column mean / 用列均值替换缺失值
        - 'median': Replace with column median (robust to outliers) / 用列中位数替换（对异常值稳健）
        - 'knn': K-Nearest Neighbors imputation / K近邻填充
    
    Hyperparameters / 超参数:
        - param (float): Strategy-specific parameter [0.0-1.0]
          策略特定参数 [0.0-1.0]
          - For 'knn': Controls number of neighbors (scaled to k=1-10)
            对于'knn': 控制邻居数量（缩放到k=1-10）
    
    Input / 输入:
        - X_train, X_val: Feature matrices with potential missing values
          可能包含缺失值的特征矩阵
    
    Output / 输出:
        - X_train, X_val: Complete feature matrices (no missing values)
          完整的特征矩阵（无缺失值）
    """
    def __init__(self):
        """初始化缺失值填充节点 / Initialize imputation node"""
        methods = {'impute': impute_data}
        super().__init__('N1', 'Impute', 'DataProcessing', methods)


class FeatureMatrixNode(Node):
    """
    特征矩阵构建节点 (N2) / Feature Matrix Construction Node (N2)
    
    Constructs comprehensive feature matrix from crystal structures using materials informatics.
    使用材料信息学从晶体结构构建综合特征矩阵。
    
    Available Methods / 可用方法:
        - 'construct': Build feature matrix from crystal structures
          从晶体结构构建特征矩阵
    
    Feature Engineering / 特征工程:
        - Composition features (element ratios, atomic properties)
          组成特征（元素比例，原子性质）
        - Structure features (lattice parameters, symmetry)
          结构特征（晶格参数，对称性）
        - Electronic features (band gap estimates, oxidation states)
          电子特征（带隙估计，氧化态）
    
    Input / 输入:
        - structures: Crystal structure objects from N0
          来自N0的晶体结构对象
        - y_formation_energy: Target labels
          目标标签
    
    Output / 输出:
        - X_train, y_train: Training feature matrix and labels
          训练特征矩阵和标签
        - X_val, y_val: Validation feature matrix and labels
          验证特征矩阵和标签
    
    Note / 注意:
        This is the second node in the 10-node architecture (always executed second).
        这是10节点架构中的第二个节点（总是第二个执行）。
        Feature matrix MUST be constructed before imputation or other preprocessing.
        特征矩阵必须在填充或其他预处理之前构建。
    """
    def __init__(self):
        """初始化特征矩阵节点 / Initialize feature matrix node"""
        methods = {'construct': feature_matrix}
        super().__init__('N2', 'FeatureMatrix', 'FeatureEngineering', methods)

# ================= Legacy Nodes (N3-N5): Old 6-Node Architecture =================
# 遗留节点 (N3-N5): 旧的6节点架构 / Legacy nodes from old 6-node architecture
# NOTE: These are kept for backward compatibility but replaced by new nodes in 10-node architecture
# 注意: 保留这些节点以实现向后兼容，但在10节点架构中已被新节点替代

class FeatureSelectionNode(Node):
    """
    特征选择节点 (旧N3) / Feature Selection Node (Old N3)
    
    Legacy node for feature selection (replaced by N6 SelectionNode in 10-node architecture).
    特征选择的遗留节点（在10节点架构中被N6 SelectionNode替代）。
    
    Available Methods / 可用方法:
        - 'select': Execute feature selection strategy
          执行特征选择策略
    
    Note / 注意:
        Use SelectionNode (N6) for 10-node architecture pipelines.
        对于10节点架构流水线请使用SelectionNode (N6)。
    """
    def __init__(self):
        """初始化特征选择节点 / Initialize feature selection node"""
        methods = {'select': feature_selection}
        super().__init__('N3', 'FeatureSelection', 'FeatureEngineering', methods)


class ScalingNode(Node):
    """
    特征缩放节点 (旧N4) / Feature Scaling Node (Old N4)
    
    Legacy node for feature scaling (replaced by N7 ScalingNodeB in 10-node architecture).
    特征缩放的遗留节点（在10节点架构中被N7 ScalingNodeB替代）。
    
    Available Methods / 可用方法:
        - 'scale': Execute scaling strategy
          执行缩放策略
    
    Note / 注意:
        Use ScalingNodeB (N7) for 10-node architecture pipelines.
        对于10节点架构流水线请使用ScalingNodeB (N7)。
    """
    def __init__(self):
        """初始化特征缩放节点 / Initialize scaling node"""
        methods = {'scale': scale_features}
        super().__init__('N4', 'Scaling', 'Preprocessing', methods)


class ModelTrainingNode(Node):
    """
    模型训练节点 (旧N5) / Model Training Node (Old N5)
    
    Legacy node for model training (replaced by N8 ModelTrainingNodeB in 10-node architecture).
    模型训练的遗留节点（在10节点架构中被N8 ModelTrainingNodeB替代）。
    
    Available Methods / 可用方法:
        - 'train_rf': Random Forest Regressor / 随机森林回归器
        - 'train_gbr': Gradient Boosting Regressor / 梯度提升回归器
        - 'train_lgbm': LightGBM Regressor / LightGBM回归器
        - 'train_xgb': XGBoost Regressor / XGBoost回归器
        - 'train_cat': CatBoost Regressor / CatBoost回归器
    
    Note / 注意:
        Use ModelTrainingNodeB (N8) for 10-node architecture pipelines.
        对于10节点架构流水线请使用ModelTrainingNodeB (N8)。
    """
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
# 10节点架构扩展节点 / Extended nodes for 10-node flexible architecture
# These nodes enable PPO to explore millions of pipeline combinations
# 这些节点使PPO能够探索数百万种流水线组合

class CleaningNode(Node):
    """
    数据清洗节点 (N3) / Data Cleaning Node (N3)
    
    Removes outliers and noise from the dataset before feature engineering.
    在特征工程之前从数据集中去除异常值和噪声。
    
    Available Methods / 可用方法:
        - 'clean': Applies data cleaning based on strategy parameter
          应用基于策略参数的数据清洗
          
    Strategies / 策略:
        - 'outlier': Remove statistical outliers / 去除统计异常值
        - 'noise': Denoise using smoothing / 使用平滑去噪
        - 'none': Skip cleaning / 跳过清洗
    
    Hyperparameters / 超参数:
        - param (float): Threshold for outlier detection [0.0-1.0]
          异常值检测阈值 [0.0-1.0]
    """
    def __init__(self):
        """初始化数据清洗节点 / Initialize cleaning node"""
        methods = {'clean': clean_data}
        super().__init__('N3', 'Cleaning', 'DataProcessing', methods)


class GNNNode(Node):
    """
    图神经网络处理节点 (N4) / Graph Neural Network Processing Node (N4)
    
    Applies graph neural networks to extract structural features from crystal graphs.
    应用图神经网络从晶体图中提取结构特征。
    
    Available Methods / 可用方法:
        - 'process': Execute GNN-based feature extraction
          执行基于GNN的特征提取
    
    GNN Architectures / GNN架构:
        - 'gcn': Graph Convolutional Network / 图卷积网络
        - 'gat': Graph Attention Network / 图注意力网络  
        - 'sage': GraphSAGE Network / GraphSAGE网络
    
    Note / 注意:
        Currently placeholder implementation. Full GNN processing to be integrated.
        当前为占位符实现。完整的GNN处理将被集成。
    """
    def __init__(self):
        """初始化图神经网络节点 / Initialize GNN node"""
        methods = {'process': gnn_process}
        super().__init__('N4', 'GNN', 'FeatureEngineering', methods)


class KGNode(Node):
    """
    知识图谱处理节点 (N5) / Knowledge Graph Processing Node (N5)
    
    Enriches features with domain knowledge from materials science knowledge graphs.
    使用材料科学知识图谱中的领域知识丰富特征。
    
    Available Methods / 可用方法:
        - 'process': Execute knowledge graph enrichment
          执行知识图谱富集
    
    KG Strategies / 知识图谱策略:
        - 'entity': Entity-based knowledge extraction / 基于实体的知识提取
        - 'relation': Relation-based knowledge extraction / 基于关系的知识提取
        - 'none': Skip knowledge graph enrichment / 跳过知识图谱富集
    
    Note / 注意:
        Currently placeholder implementation. Full KG integration to be developed.
        当前为占位符实现。完整的知识图谱集成将被开发。
    """
    def __init__(self):
        """初始化知识图谱节点 / Initialize knowledge graph node"""
        methods = {'process': kg_process}
        super().__init__('N5', 'KnowledgeGraph', 'FeatureEngineering', methods)


class SelectionNode(Node):
    """
    特征选择节点 (N6) / Feature Selection Node (N6)
    
    Selects the most relevant features for model training to reduce dimensionality.
    选择最相关的特征进行模型训练以降低维度。
    
    Available Methods / 可用方法:
        - 'select': Execute feature selection strategy
          执行特征选择策略
    
    Selection Strategies / 选择策略:
        - 'variance': Variance threshold-based selection / 基于方差阈值的选择
        - 'univariate': Univariate statistical tests / 单变量统计检验
        - 'pca': Principal Component Analysis / 主成分分析
    
    Hyperparameters / 超参数:
        - param (float): Selection threshold or component ratio [0.0-1.0]
          选择阈值或成分比率 [0.0-1.0]
    """
    def __init__(self):
        """初始化特征选择节点 / Initialize feature selection node"""
        methods = {'select': feature_selection}
        super().__init__('N6', 'FeatureSelection', 'FeatureEngineering', methods)


class ScalingNodeB(Node):
    """
    特征缩放节点 (N7) / Feature Scaling Node (N7)
    
    Normalizes feature distributions to improve model convergence and performance.
    归一化特征分布以提高模型收敛性和性能。
    
    Available Methods / 可用方法:
        - 'scale': Execute scaling strategy
          执行缩放策略
    
    Scaling Strategies / 缩放策略:
        - 'std': StandardScaler (zero mean, unit variance) / 标准缩放器（零均值，单位方差）
        - 'robust': RobustScaler (median, IQR) / 鲁棒缩放器（中位数，IQR）
        - 'minmax': MinMaxScaler [0,1] normalization / 最小最大缩放器 [0,1] 归一化
    
    Hyperparameters / 超参数:
        - param (float): Scaling parameter (strategy-dependent) [0.0-1.0]
          缩放参数（取决于策略）[0.0-1.0]
    """
    def __init__(self):
        """初始化特征缩放节点 / Initialize scaling node"""
        methods = {'scale': scale_features}
        super().__init__('N7', 'Scaling', 'Preprocessing', methods)


class ModelTrainingNodeB(Node):
    """
    模型训练节点 (N8) / Model Training Node (N8)
    
    Trains machine learning models for formation energy prediction.
    训练用于形成能预测的机器学习模型。
    
    Available Methods / 可用方法:
        - 'train_rf': Random Forest Regressor / 随机森林回归器
        - 'train_gbr': Gradient Boosting Regressor / 梯度提升回归器
        - 'train_lgbm': LightGBM Regressor / LightGBM回归器
        - 'train_xgb': XGBoost Regressor / XGBoost回归器
        - 'train_cat': CatBoost Regressor / CatBoost回归器
    
    Hyperparameters / 超参数:
        - param (float): Model complexity parameter [0.0-1.0]
          模型复杂度参数 [0.0-1.0]
        - algorithm (str): Specific ML algorithm to use
          使用的特定机器学习算法
    
    Note / 注意:
        This is the pre-end node in the 10-node architecture (always executed before N9).
        这是10节点架构中的倒数第二个节点（总是在N9之前执行）。
    """
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
        执行模型训练，增加异常捕获和日志 / Execute model training with error handling and logging
        
        Supports dynamic algorithm selection through 'train' method dispatch.
        支持通过'train'方法调度动态算法选择。
        
        Args:
            method (str): Training method name or 'train' for dynamic selection
                         训练方法名称或'train'用于动态选择
            params (dict): Training parameters including optional 'algorithm' key
                          训练参数，包括可选的'algorithm'键
            data (dict): Input data with X_train, y_train, X_val, y_val
                        输入数据，包含X_train, y_train, X_val, y_val
        
        Returns:
            dict: Training results including model, metrics, and predictions
                  训练结果，包括模型、指标和预测
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            # Handle 'train' method by dispatching to specific algorithm
            # 处理'train'方法，调度到特定算法
            if method == 'train':
                algorithm = params.pop('algorithm', self.default_algorithm)
                method = algorithm
            
            logger.info(f"[Node:{self.name}] 执行方法 / Executing method: {method}, 输入keys / Input keys: {list(data.keys())}")
            result = super().execute(method, params, data)
            
            # Ensure output is dictionary format / 确保输出为字典格式
            if not isinstance(result, dict):
                logger.warning(f"[Node:{self.name}] 输出非dict，自动包装 / Output not dict, auto-wrapping")
                result = {'output': result}
            
            logger.info(f"[Node:{self.name}] 输出keys / Output keys: {list(result.keys())}")
            return result
        except Exception as e:
            logger.error(f"[Node:{self.name}] 执行 / Executing {method} 异常 / exception: {e}")
            return {'error': f'{self.name}.{method} failed: {e}'}


class EndNode(Node):
    """
    终止节点 (N9) / Termination Node (N9)
    
    Marks the end of the pipeline execution and triggers reward computation.
    标记流水线执行的结束并触发奖励计算。
    
    Available Methods / 可用方法:
        - 'terminate': Finalize pipeline and compute final metrics
          完成流水线并计算最终指标
    
    Behavior / 行为:
        - Signals episode termination to the RL environment
          向RL环境发出回合终止信号
        - Triggers MAE, R² calculation for reward function
          触发MAE、R²计算用于奖励函数
        - Updates fingerprint with final pipeline performance
          使用最终流水线性能更新指纹
    
    Note / 注意:
        This is the final node in the 10-node architecture (always executed last).
        这是10节点架构中的最后一个节点（总是最后执行）。
    """
    def __init__(self):
        """初始化终止节点 / Initialize end node"""
        methods = {'terminate': terminate}
        super().__init__('N9', 'End', 'Control', methods)