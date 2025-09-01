"""
完整流水线：N0 → N2 → N1 → N3 → N4 → N5
Full pipeline: N0 (data fetch) → N2 (feature matrix) → N1 (imputation) → N3 (feature selection) → N4 (scaling) → N5 (model training)
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
    完整流水线调度函数。
    Full pipeline runner.
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
    Flexible runner for 10-node Option 2 sequences.

    Expects keys like:
      sequence: [ 'N0','N2', <perm of N1,N3,N4,N5,N6,N7>, 'N8','N9' ]
      For each Nx in sequence (except N0,N2,N9), expects Nx_method and optional Nx_params.
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
