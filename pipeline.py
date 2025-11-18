"""
完整流水线模块 / Complete Pipeline Module

This module implements two pipeline execution functions:
本模块实现两个流水线执行函数：

1. run_pipeline() - Legacy 6-node pipeline (N0→N2→N1→N3→N4→N5)
   旧的6节点流水线，用于向后兼容
   
2. run_pipeline_config() - Flexible 10-node pipeline (N0→N2→[flexible]→N8→N9)
   灵活的10节点流水线，支持PPO控制的节点选择和排序

10-Node Architecture / 10节点架构:
    N0: DataFetch (固定首位 / Fixed start)
    N2: FeatureMatrix (固定第二 / Fixed second)
    N1: Impute (灵活 / Flexible)
    N3: Cleaning (灵活 / Flexible) 
    N4: GNN (灵活 / Flexible)
    N5: KnowledgeGraph (灵活 / Flexible)
    N6: FeatureSelection (灵活 / Flexible)
    N7: Scaling (灵活 / Flexible)
    N8: ModelTraining (固定倒二 / Fixed pre-end)
    N9: End (固定最后 / Fixed end)

PPO controls the order and selection of flexible middle nodes (N1,N3,N4,N5,N6,N7)
PPO控制灵活中间节点的顺序和选择 (N1,N3,N4,N5,N6,N7)
"""
from nodes import (
    DataFetchNode,
    ImputeNode,
    FeatureMatrixNode,
    FeatureSelectionNode,
    ScalingNode,
    ModelTrainingNode,
    CleaningNode,
    GNNNode,
    KGNode,
    SelectionNode,
    ScalingNodeB,
    ModelTrainingNodeB,
    EndNode,
)
from config import PROC_DIR, MODEL_DIR, LOG_DIR, TARGET_PROP
from methods.data_methods import prepare_node_input, validate_state_keys, split_labels, update_state
from methods.model_methods import compute_metrics_and_sizes, print_results, save_pipeline_outputs
# from utils.pipeline_utils import PipelineAPI  # Moved to avoid circular import
import logging
from pathlib import Path
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, r2_score

# 日志配置 / Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_pipeline(
    cache: bool = True,
    impute_strategy: str = 'mean',
    impute_params: Optional[dict] = None,
    nan_thresh: float = 0.5,
    train_val_ratio: float = 0.8,
    selection_strategy: str = 'none',
    selection_params: Optional[dict] = None,
    scaling_strategy: str = 'standard',
    scaling_params: Optional[dict] = None,
    model_strategy: str = 'rf',
    model_params: Optional[dict] = None
) -> dict:
    """
    完整流水线调度函数 (旧版6节点) / Full Pipeline Runner (Legacy 6-Node)
    
    Executes a fixed-sequence pipeline: N0→N2→N1→N3→N4→N5
    执行固定序列流水线：N0→N2→N1→N3→N4→N5
    
    This function is kept for backward compatibility. For new 10-node architecture,
    use run_pipeline_config() instead.
    此函数保留用于向后兼容。对于新的10节点架构，请使用run_pipeline_config()。
    
    Args:
        cache (bool): Whether to use cached data / 是否使用缓存数据
        impute_strategy (str): Imputation strategy ('mean'/'median'/'knn')
                              缺失值填充策略
        impute_params (dict): Additional imputation parameters / 额外填充参数
        nan_thresh (float): NaN threshold for feature filtering / NaN阈值
        train_val_ratio (float): Train/validation split ratio / 训练/验证划分比例
        selection_strategy (str): Feature selection strategy / 特征选择策略
        selection_params (dict): Additional selection parameters / 额外选择参数
        scaling_strategy (str): Scaling strategy ('standard'/'robust'/'minmax')
                               缩放策略
        scaling_params (dict): Additional scaling parameters / 额外缩放参数
        model_strategy (str): Model type ('rf'/'gbr'/'xgb'/'cat')
                             模型类型
        model_params (dict): Model hyperparameters / 模型超参数
    
    Returns:
        dict: Pipeline results including trained model and metrics
              流水线结果，包括训练好的模型和指标
    
    Note:
        This uses legacy node IDs (N3=FeatureSelection, N4=Scaling, N5=ModelTraining)
        which differ from 10-node architecture.
        使用旧的节点ID（N3=特征选择，N4=缩放，N5=模型训练），与10节点架构不同。
    """
    
    # N0 数据获取特征化 / N0 Data fetch & featurization
    data_node = DataFetchNode()
    logger.info("Running N0 (DataFetchNode.api), cache=%s", cache)
    fetched = data_node.execute(method='api', params={'cache': cache}, data={})

    # N2 特征矩阵生成 / N2 Feature matrix construction (MUST come before imputation)
    feature_node = FeatureMatrixNode()
    logger.info("Running N2 (FeatureMatrixNode.construct), nan_thresh=%.2f, train_val_ratio=%.2f", nan_thresh, train_val_ratio)
    features = feature_node.execute(
        method='construct',
        params={'nan_thresh': nan_thresh, 'train_val_ratio': train_val_ratio, 'verbose': False},
        data=fetched
    )

    # N1 缺失填补 / N1 Imputation (operates on X_train, X_val, X_test)
    impute_node = ImputeNode()
    logger.info("Running N1 (ImputeNode.impute), strategy=%s, params=%s", impute_strategy, impute_params)
    imputed = impute_node.execute(
        method='impute', 
        params={'strategy': impute_strategy, 'params': impute_params or {}}, 
        data=features
    )

    # N3 特征选择 / N3 Feature selection
    select_node = FeatureSelectionNode()
    logger.info("Running N3 (FeatureSelectionNode.select), strategy=%s, params=%s", selection_strategy, selection_params)
    selected = select_node.execute(
        method='select',
        params={'strategy': selection_strategy, 'params': selection_params or {}},
        data=imputed
    )

    # N4 标准化 / N4 Scaling
    scaling_node = ScalingNode()
    logger.info("Running N4 (ScalingNode.scale), strategy=%s, params=%s", scaling_strategy, scaling_params)
    scaled = scaling_node.execute(
        method='scale',
        params={'strategy': scaling_strategy, 'params': scaling_params or {}},
        data=selected
    )

    # N5 模型训练 / N5 Model training
    train_node = ModelTrainingNode()
    logger.info("Running N5 (ModelTrainingNode.train), strategy=%s, params=%s", model_strategy, model_params)
    algorithm_name = f"train_{model_strategy}"  # Add 'train_' prefix
    trained = train_node.execute(
        method='train',
        params={'algorithm': algorithm_name, **(model_params or {})},
        data=scaled
    )
    
    return trained


def run_pipeline_config(**config) -> dict:
    """
    灵活的10节点流水线执行器 / Flexible 10-Node Pipeline Runner
    
    Executes configurable pipeline sequences with PPO-controlled node selection.
    执行可配置的流水线序列，支持PPO控制的节点选择。
    
    10-Node Architecture / 10节点架构:
        - Fixed nodes: N0 (start), N2 (second), N8 (pre-end), N9 (end)
          固定节点：N0 (起始), N2 (第二), N8 (倒二), N9 (终止)
        - Flexible nodes: N1, N3, N4, N5, N6, N7 (PPO controls order)
          灵活节点：N1, N3, N4, N5, N6, N7 (PPO控制顺序)
    
    Config Format / 配置格式:
        sequence (list): Node execution order / 节点执行顺序
                        Example: ['N0','N2','N1','N6','N7','N8','N9']
        
        For each node Nx (except N0, N2, N9):
        对于每个节点Nx（除了N0, N2, N9）:
            Nx_method (str): Method name for node / 节点的方法名
            Nx_params (dict): Optional method parameters / 可选方法参数
        
        Global parameters / 全局参数:
            cache (bool): Use cached data (default: True) / 使用缓存数据
            nan_thresh (float): NaN threshold (default: 0.5) / NaN阈值
            train_val_ratio (float): Split ratio (default: 0.8) / 划分比例
    
    Node Methods / 节点方法:
        N1 (Impute): 'mean', 'median', 'knn'
        N3 (Cleaning): 'outlier', 'noise', 'none'
        N4 (GNN): 'gcn', 'gat', 'sage'
        N5 (KG): 'entity', 'relation', 'none'
        N6 (Selection): 'variance', 'univariate', 'pca'
        N7 (Scaling): 'std', 'robust', 'minmax'
        N8 (ModelTraining): 'rf', 'gbr', 'xgb', 'cat'
    
    Args:
        **config: Pipeline configuration dictionary / 流水线配置字典
    
    Returns:
        dict: Pipeline results / 流水线结果
            - metrics: Performance metrics (MAE, R2, etc.) / 性能指标
            - sizes: Data size information / 数据大小信息
            - feature_names: Selected feature names / 选择的特征名称
            - model: Trained model object / 训练好的模型对象
            - outputs_dir: Directory with saved outputs / 输出保存目录
    
    Example / 示例:
        >>> config = {
        ...     'sequence': ['N0','N2','N1','N6','N7','N8','N9'],
        ...     'N1_method': 'median',
        ...     'N6_method': 'pca',
        ...     'N7_method': 'std',
        ...     'N8_method': 'xgb',
        ...     'cache': True
        ... }
        >>> result = run_pipeline_config(**config)
    
    Note:
        This function is designed for PPO reinforcement learning optimization.
        PPO agent automatically generates optimal sequences and method selections.
        此函数设计用于PPO强化学习优化。PPO代理自动生成最优序列和方法选择。
    """
    sequence = config.get('sequence', [])
    if not sequence:
        raise ValueError("Missing sequence in pipeline config")

    # Defaults
    cache = config.get('cache', True)
    nan_thresh = config.get('nan_thresh', 0.5)
    train_val_ratio = config.get('train_val_ratio', 0.8)

    state: Dict[str, Any] = {}
    exec_times: Dict[str, float] = {}
    import time
    start_time = time.time()

    # Node instances
    n0 = DataFetchNode()
    n2 = FeatureMatrixNode()
    n1 = ImputeNode()
    n3c = CleaningNode()
    n4g = GNNNode()
    n5k = KGNode()
    n6s = SelectionNode()
    n7b = ScalingNodeB()
    n8t = ModelTrainingNodeB()
    n9e = EndNode()

    def step_timer(key, fn):
        t0 = time.time()
        out = fn()
        exec_times[key] = time.time() - t0
        return out

    # Always run N0 then N2 first regardless of explicit sequence assertion here
    if sequence[0] != 'N0' or sequence[1] != 'N2':
        # still honor with forced start
        pass

    # N0
    out0 = step_timer('N0', lambda: n0.execute('api', {'cache': cache}, {}))
    update_state('N0', out0, state)

    # N2
    out2 = step_timer('N2', lambda: n2.execute('construct', {'nan_thresh': nan_thresh, 'train_val_ratio': train_val_ratio, 'verbose': False}, state))
    update_state('N2', out2, state)

    # Middle nodes
    middle_nodes = [n for n in sequence if n in {'N1','N3','N4','N5','N6','N7'}]
    for nid in middle_nodes:
        method = config.get(f'{nid}_method')
        params = config.get(f'{nid}_params', {}) or {}
        if nid == 'N1':
            # Imputation
            out = step_timer('N1', lambda: n1.execute('impute', {'strategy': method, 'params': params}, state))
            update_state('N1', out, state)
        elif nid == 'N3':
            out = step_timer('N3', lambda: n3c.execute('clean', {'strategy': method, 'params': params}, state))
            update_state('N3', out, state)
        elif nid == 'N4':
            out = step_timer('N4', lambda: n4g.execute('process', {'strategy': method, 'params': params}, state))
            update_state('N4', out, state)
        elif nid == 'N5':
            out = step_timer('N5', lambda: n5k.execute('process', {'strategy': method, 'params': params}, state))
            update_state('N5', out, state)
        elif nid == 'N6':
            out = step_timer('N6', lambda: n6s.execute('select', {'strategy': method, 'params': params}, state))
            update_state('N6', out, state)
        elif nid == 'N7':
            # map std -> standard
            strat = 'standard' if method == 'std' else method
            out = step_timer('N7', lambda: n7b.execute('scale', {'strategy': strat, 'params': params}, state))
            update_state('N7', out, state)

    # N8 Training
    meth8 = config.get('N8_method', 'rf')
    params8 = config.get('N8_params', {}) or {}
    algo = f"train_{meth8}"
    out8 = step_timer('N8', lambda: n8t.execute('train', {'algorithm': algo, **params8}, state))
    update_state('N8', out8, state)

    # N9 End (no-op)
    _ = step_timer('N9', lambda: n9e.execute('terminate', {}, state))

    # Metrics & outputs
    res = compute_metrics_and_sizes(state, start_time, sequence, exec_times)
    if isinstance(res, tuple) and len(res) == 2:
        metrics, sizes = res
    elif isinstance(res, dict):
        metrics, sizes = res, {}
    else:
        # Fallback: unexpected return, wrap into metrics
        metrics, sizes = {'error': 'metrics_failed', 'value': res}, {}
    save_dir = save_pipeline_outputs(state, metrics, verbose=False)

    outputs = {
        'metrics': metrics,
        'sizes': sizes,
        'feature_names': state.get('feature_names'),
        'model': state.get('model'),
        'outputs_dir': save_dir,
    }
    return outputs

# 主函数入口 / Main entry point
if __name__ == '__main__':
    result = run_pipeline(
        cache=True,
        impute_strategy='mean',
        selection_strategy='none',
        scaling_strategy='standard',
        model_strategy='rf',
        model_params={'n_estimators': 50}
    )
    print("Pipeline completed successfully!")
    print(f"Final result keys: {list(result.keys())}")
