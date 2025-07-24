"""
完整流水线：N0 → N2 → N1 → N3 → N4 → N5
Full pipeline: N0 (data fetch) → N2 (feature matrix) → N1 (imputation) → N3 (feature selection) → N4 (scaling) → N5 (model training)
"""
from nodes import DataFetchNode, ImputeNode, FeatureMatrixNode, FeatureSelectionNode, ScalingNode, ModelTrainingNode
from config import PROC_DIR, MODEL_DIR, LOG_DIR, TARGET_PROP
from methods.data_methods import prepare_node_input, validate_state_keys, split_labels, update_state
from methods.model_methods import compute_metrics_and_sizes, print_results, save_pipeline_outputs
# from pipeline_utils import PipelineAPI  # Moved to avoid circular import
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
        params={'nan_thresh': nan_thresh, 'train_val_ratio': train_val_ratio},
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
