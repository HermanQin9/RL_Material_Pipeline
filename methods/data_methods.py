# methods/data_methods.py
"""
数据处理方法模块 / Data Processing Methods Module

This module contains all data processing methods for the ML pipeline including
data fetching, feature matrix construction, imputation, feature selection, and scaling.
本模块包含机器学习流水线的所有数据处理方法，包括数据获取、特征矩阵构建、缺失值填充、特征选择和数据缩放。

Main functions:
- fetch_and_featurize: 获取并特征化数据 / Fetch and featurize data
- feature_matrix: 构建特征矩阵 / Construct feature matrix  
- impute_data: 缺失值填充 / Missing data imputation
- feature_selection: 特征选择 / Feature selection
- scale_features: 特征缩放 / Feature scaling
"""
from __future__ import annotations
import logging
import os
import joblib
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from mp_api.client import MPRester
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from matminer.featurizers.composition import ElementProperty
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from config import API_KEY, CACHE_FILE, PROC_DIR, BATCH_SIZE, N_TOTAL, TARGET_PROP, MODEL_DIR

# 初始化 Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

__all__ = [
    "fetch_and_featurize",
    "prepare_node_input",
    "validate_state_keys", 
    "split_labels",
    "update_state",
    # other data‑node functions …
]

# Features will be used
FEATURE_METHODS = [
    (ElementProperty.from_preset("magpie"), "composition"),
    (DensityFeatures(), "structure"),
    (GlobalSymmetryFeatures(), "structure"),
]

# Set the cache and processed data paths
CACHE_PATH = Path(CACHE_FILE)
PROC_PATH = Path(PROC_DIR)

# Change to the public function
def split_by_fe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按是否含 Fe 将数据切分为 train/test"""
    mask = df["composition"].apply(lambda c: c is not None and c.as_dict().get("Fe", 0) > 0)
    train_df = df[~mask].reset_index(drop=True)
    test_df = df[mask].reset_index(drop=True)
    return train_df, test_df


def safe_featurize(feat_obj, x):
    """安全的特征化函数，处理可能的错误"""
    if x is None:
        return [np.nan] * len(feat_obj.feature_labels())
    try:
        return feat_obj.featurize(x)
    except (TypeError, ValueError, AttributeError) as e:
        # 当遇到 NoneType 错误或其他特征化错误时，返回 NaN
        logger.warning(f"Featurization failed for {type(feat_obj).__name__}: {str(e)[:100]}")
        return [np.nan] * len(feat_obj.feature_labels())

def apply_featurizers(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 批量应用 featurizers 并拼接结果"""
    parts: list[pd.DataFrame] = [df]
    for feat_obj, col in FEATURE_METHODS:
        labels = feat_obj.feature_labels()
        logger.info(f"Applying {type(feat_obj).__name__} to column '{col}' ({len(df)} samples)")
        # 使用安全的特征化函数
        array = Parallel(n_jobs=-1)(delayed(safe_featurize)(feat_obj, v) for v in df[col])
        parts.append(pd.DataFrame(array, columns=labels)) # type: ignore
        logger.info(f"  Generated {len(labels)} features")
    return pd.concat(parts, axis=1)

# Node 0: 获取数据、特征化并划分训练/测试集
def fetch_and_featurize(_: Any = None, cache: bool = True) -> Dict[str, Any]:
    """N0: 获取数据、特征化并划分训练/测试集"""
    logger.info("N0 start: cache=%s", cache)

    full_df = fetch_data(cache=cache)
    featurized_df = featurize_data(full_df)
    train_df, test_df = split_data(featurized_df)

    logger.info("N0 complete: train=%d, test=%d", len(train_df), len(test_df))
    return {
        "train_data": train_df,
        "test_data": test_df,
        "full_data": featurized_df,
        "X_train": None, "X_val": None, "X_test": None,
        "y_train": None, "y_val": None, "y_test": None,
    }

def get_value(d, key, default=None):
    # 支持 dict、对象（如 dataclass 或一般有属性的对象）
    if isinstance(d, dict):
        return d.get(key, default)
    elif hasattr(d, key):
        return getattr(d, key, default)
    else:
        return default

# Make the N0 into 3 functions:
def fetch_data(cache: bool = True) -> pd.DataFrame:
    if cache and CACHE_PATH.exists():
        logger.info("Loading cache from %s", CACHE_PATH)
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict):
            return cached["full_data"]
        elif isinstance(cached, pd.DataFrame):
            return cached
        else:
            raise TypeError(f"Unsupported cache type: {type(cached)}")

    logger.info("Fetching data from MP API")
    dfs = []
    fetched = 0
    with MPRester(API_KEY) as mpr:
        docs_iter = mpr.materials.summary.search(
            fields=["material_id", "structure", "elements", "formula_pretty", TARGET_PROP],
            chunk_size=BATCH_SIZE,
            num_chunks=N_TOTAL // BATCH_SIZE + 1,
        )
        for docs in tqdm(docs_iter, desc="MP fetch", ncols=80):
            docs = docs if isinstance(docs, list) else [docs]
            valid = [d for d in docs if getattr(d, TARGET_PROP, None) is not None]
            if not valid:
                continue
            df_b = pd.DataFrame([
                {
                    "material_id": get_value(d, "material_id"),
                    "structure": get_value(d, "structure"),
                    "elements": get_value(d, "elements"),
                    "formula_pretty": get_value(d, "formula_pretty"),
                    TARGET_PROP: get_value(d, TARGET_PROP),
                } for d in valid
            ]).dropna(subset=["structure"]).reset_index(drop=True)
            df_b["composition"] = df_b["structure"].apply(lambda s: getattr(s, 'composition', None) if s is not None else None)
            dfs.append(df_b)
            fetched += len(df_b)
            if fetched >= N_TOTAL:
                break

    if not dfs:
        raise RuntimeError("No data fetched from API")

    full_df = pd.concat(dfs, ignore_index=True)[:N_TOTAL]

    # 缓存
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(full_df, f)

    return full_df

def featurize_data(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = apply_featurizers(df)
    df_feat.to_csv(PROC_PATH / "all_data_feat.csv", index=False)
    return df_feat

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = split_by_fe(df)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/Test split resulted in empty dataframe!")
    return train_df, test_df



# N1: Impution Node
# Incase of the missinf values, we will use the imputer to fill them.

def impute_data(data, strategy='mean', params=None):
    strategies = {
        'mean': impute_mean,
        'median': impute_median,
        'knn': impute_knn,
        'none': impute_none
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    return strategies[strategy](data, **(params or {}))

def drop_allnan(df):
    return df.dropna(axis=1, how='all') if isinstance(df, pd.DataFrame) else df

def apply_imputer(imp, data):
    """For（X_train, X_val, X_test）use the same imputer，format in dict"""
    X_train = imp.fit_transform(drop_allnan(data['X_train']))
    X_val   = imp.transform(drop_allnan(data['X_val']))
    X_test_raw = data.get('X_test')
    X_test  = imp.transform(drop_allnan(X_test_raw)) if X_test_raw is not None else None
    assert not np.isnan(X_train).any(), "NaN remains after impute!"
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'imputer': imp
    }

# Imputation methods
def impute_mean(data):
    logger.info("Imputing using mean")
    imp = SimpleImputer(strategy='mean')
    return apply_imputer(imp, data)

def impute_median(data):
    logger.info("Imputing using median")
    imp = SimpleImputer(strategy='median')
    return apply_imputer(imp, data)

def impute_knn(data, n_neighbors=5, **params):
    """
    KNN填充缺失值 / KNN imputation for missing values
    
    Args:
        data: 输入数据 / Input data
        n_neighbors: 邻居数量 / Number of neighbors
        **params: 包含'param' [0,1]映射到n_neighbors [3,10] / Contains 'param' [0,1] mapped to n_neighbors [3,10]
    """
    # Map param [0,1] to n_neighbors [3, 10] / 映射param到邻居数
    if 'param' in params:
        p = params['param']
        n_neighbors = int(3 + p * 7)
    logger.info("Imputing using KNN, neighbors=%d", n_neighbors)
    imp = KNNImputer(n_neighbors=n_neighbors)
    return apply_imputer(imp, data)

def impute_none(data):
    logger.info("No imputation performed")
    return {
        'X_train': data['X_train'], 
        'X_val': data['X_val'], 
        'X_test': data.get('X_test'),
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test')
    }



# N2: Feature Matrix Node 需要设置verbose开关
def select_feature_columns(df, nan_thresh=0.5):
    exclude = ['material_id', 'structure', 'elements', 'formula_pretty', 'composition', TARGET_PROP]
    return [
        c for c in df.columns if c not in exclude and df[c].isna().mean() < nan_thresh
    ]

def numeric_feature_matrix(df, cols):
    return df[cols].apply(pd.to_numeric, errors='coerce')

def split_train_val(X, y, ratio=0.8):
    split_idx = int(ratio * len(X))
    return X.iloc[:split_idx], X.iloc[split_idx:], y[:split_idx], y[split_idx:]

def select_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns

def feature_matrix(data, nan_thresh=0.5, train_val_ratio=0.8, verbose=True):
    train_df = data['train_data']
    test_df = data['test_data']

    feat_cols = select_feature_columns(train_df, nan_thresh)
    X_train_full = numeric_feature_matrix(train_df, feat_cols)
    X_test_full = numeric_feature_matrix(test_df, feat_cols) if len(test_df) > 0 else None

    y_full = train_df[TARGET_PROP].values
    X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_full, train_val_ratio)
    X_test = X_test_full

    num_cols = select_numeric_cols(X_train)
    X_train = X_train[num_cols]
    X_val   = X_val[num_cols]
    X_test  = X_test[num_cols] if X_test is not None else None
    y_test  = test_df[TARGET_PROP].values if TARGET_PROP in test_df.columns else None

    if verbose:
        print(f"[N2] X_train shape: {X_train_full.shape}, NaN count: {X_train_full.isna().sum().sum()}")
        print("[DEBUG] feat_cols:", feat_cols)
        print("[DEBUG] X_train shape:", X_train.shape)
        print("[DEBUG] X_val shape:", X_val.shape)
        print("[DEBUG] X_train head:\n", X_train.head())
        print("[DEBUG] X_val head:\n", X_val.head())

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'feature_names': list(num_cols),
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


# Node 3: Feature selection | Node 3: Feature selection

def feature_selection(data, strategy='none', params=None):
    """
    特征选择入口函数。
    Feature selection entry function.
    """
    strategies = {
        'none': no_selection,
        'variance': variance_selection,
        'univariate': univariate_selection,
        'pca': pca_selection
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    return strategies[strategy](data, **(params or {}))

def apply_selector(selector, data, y_required=False):
    """
    用于 N3/N4：统一作用于 train/val/test，适配 selector/scaler。
    Apply sklearn selector/scaler/transformer to train/val/test sets.

    Args:
        selector: 拟合对象 (selector/scaler/transformer)
        data: 输入数据 dict
        y_required: 是否强制传 y（比如特征选择）

    Returns:
        X_train, X_val, X_test: 各 split 处理后矩阵
    """
    if y_required and data.get('y_train') is not None:
        X_train = selector.fit_transform(data['X_train'], data['y_train'])
    else:
        X_train = selector.fit_transform(data['X_train'])
    X_val   = selector.transform(data['X_val'])
    X_test  = selector.transform(data['X_test']) if data.get('X_test') is not None else None
    return X_train, X_val, X_test

def filter_feature_names(data, selector, prefix=''):
    """
    根据支持向量或主成分自动生成新的特征名。
    Auto-generate new feature names according to support mask or PCA components.
    """
    if hasattr(selector, 'get_support') and 'feature_names' in data:
        mask = selector.get_support()
        return [fname for fname, m in zip(data['feature_names'], mask) if m]
    elif prefix:
        num_comps = selector.n_components_ if hasattr(selector, 'n_components_') else selector.n_components
        return [f"{prefix}{i+1}" for i in range(num_comps)]
    else:
        return data.get('feature_names')

def no_selection(data, **params):
    """
    不进行任何特征选择，直接返回。
    No feature selection, just return the original features.
    """
    return {
        'X_train': data['X_train'],
        'X_val': data['X_val'],
        'X_test': data.get('X_test'),
        'feature_names': data.get('feature_names'),
        'y_train': data.get('y_train'), 'y_val': data.get('y_val'), 'y_test': data.get('y_test')
    }

def variance_selection(data, var_ratio=0.01, **params):
    """
    方差阈值特征选择。
    Feature selection by variance threshold.
    """
    if np.isnan(data['X_train']).any():
        raise RuntimeError('X_train contains NaN before feature selection!')
    threshold = params.get('param', var_ratio)
    if isinstance(threshold, float) and threshold < 0:
        threshold = 0.0
    selector = VarianceThreshold(threshold=threshold)
    X_train, X_val, X_test = apply_selector(selector, data)
    np.save(os.path.join(PROC_DIR, "selector_mask.npy"), selector.get_support())
    new_feature_names = filter_feature_names(data, selector)
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': data.get('y_train'), 'y_val': data.get('y_val'), 'y_test': data.get('y_test'),
        'selector': selector, 'feature_names': new_feature_names
    }

# 有监督学习，需要用到y | Supervised learning, requires the use of y
def univariate_selection(data, k=20, **params):
    """
    单变量特征选择 (F检验, 相关性评分)。
    Univariate feature selection (F-regression, correlation scores).
    """
    if np.isnan(data['X_train']).any():
        raise RuntimeError('X_train contains NaN before feature selection!')
    X_tr = data['X_train']
    X_tr_arr = X_tr if isinstance(X_tr, np.ndarray) else np.array(X_tr)
    y_tr = data.get('y_train')
    if 'param' in params:
        p = params['param']
        num_feats = X_tr_arr.shape[1]
        k_val = max(1, int(round(p * (num_feats - 1))) + 1)
    else:
        k_val = k
    selector = SelectKBest(score_func=f_regression, k=min(k_val, X_tr_arr.shape[1]))
    # 统一通过 apply_selector 调用，并指定 y_required=True
    X_train, X_val, X_test = apply_selector(
        selector,
        {
            'X_train': X_tr_arr,
            'X_val': np.array(data['X_val']),
            'X_test': np.array(data['X_test']) if data.get('X_test') is not None else None,
            'y_train': y_tr
        },
        y_required=True  # 关键！select_univariate 必须传 y
    )
    new_feature_names = filter_feature_names(data, selector)
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': data.get('y_train'), 'y_val': data.get('y_val'), 'y_test': data.get('y_test'),
        'selector': selector, 'feature_names': new_feature_names
    }

def pca_selection(data, n_components=0.95, **params):
    """
    主成分分析 (PCA) 特征降维。
    Principal component analysis (PCA) for feature reduction.
    """
    if np.isnan(data['X_train']).any():
        raise RuntimeError('X_train contains NaN before feature selection!')
    X_tr = data['X_train']
    X_tr_arr = X_tr if isinstance(X_tr, np.ndarray) else np.array(X_tr)
    if 'param' in params:
        p = params['param']
        if p <= 0:
            comp_val = 0.01
        elif p <= 1:
            comp_val = p
        else:
            comp_val = int(p)
    else:
        comp_val = n_components
    pca = PCA(n_components=comp_val)
    X_train = pca.fit_transform(X_tr_arr)
    X_val   = pca.transform(np.array(data['X_val']))
    X_test  = pca.transform(np.array(data['X_test'])) if data.get('X_test') is not None else None
    new_feature_names = filter_feature_names(data, pca, prefix='PC')
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': data.get('y_train'), 'y_val': data.get('y_val'), 'y_test': data.get('y_test'),
        'selector': pca, 'feature_names': new_feature_names
    }

# Node 4 : Feature Scaling

# 全局唯一 apply_selector 已在此文件定义
# def apply_selector(selector, data, y_required=False): ...

def scale_features(data, strategy='standard', params=None):
    """
    特征缩放/标准化统一入口。
    Unified scaling interface for pipeline.
    """
    strategies = {
        'none': scale_none,
        'standard': scale_standard,
        'robust': scale_robust,
        'minmax': scale_minmax
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown scaling strategy: {strategy}")
    return strategies[strategy](data, **(params or {}))

def scale_standard(data, **params):
    """
    标准化（StandardScaler）
    Standardization (StandardScaler)
    """
    scaler = StandardScaler()
    X_train, X_val, X_test = apply_selector(scaler, data)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'scaler': scaler
    }

def scale_robust(data, **params):
    """
    稳健缩放（RobustScaler）
    Robust scaling (RobustScaler)
    """
    X_train, X_val, X_test = apply_selector(RobustScaler(), data)
    joblib.dump(RobustScaler(), os.path.join(MODEL_DIR, "scaler_robust.joblib"))
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'scaler': RobustScaler()
    }

def scale_minmax(data, **params):
    """
    MinMax标准化（MinMaxScaler）
    Min-Max scaling (MinMaxScaler)
    """
    X_train, X_val, X_test = apply_selector(MinMaxScaler(), data)
    joblib.dump(MinMaxScaler(), os.path.join(MODEL_DIR, "scaler_minmax.joblib"))
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test'),
        'scaler': MinMaxScaler()
    }

def scale_none(data, **params):
    """
    不进行缩放。
    No scaling (identity transform)
    """
    return {
        'X_train': data['X_train'],
        'X_val': data['X_val'],
        'X_test': data.get('X_test'),
        'y_train': data.get('y_train'),
        'y_val': data.get('y_val'),
        'y_test': data.get('y_test')
    }


# ========================= 辅助函数 / Helper Functions =========================

def prepare_node_input(node_key: str, state: dict, verbose: bool = False) -> dict:
    """
    为指定节点准备输入数据。
    Prepare input data for the specified node.
    """
    if verbose:
        print(f"\n[节点 {node_key}] 输入 keys: {list(state.keys())}")
        if state.get('X_train') is not None:
            print(f"[节点 {node_key}] X_train shape: {state['X_train'].shape}")
        if state.get('y_train') is not None:
            print(f"[节点 {node_key}] y_train preview: {state['y_train'][:5]}")

    if node_key == 'N0':
        # N0 数据获取节点不需要输入 / N0 data fetch node needs no input
        return {}
    
    elif node_key == 'N2':
        # N2 特征矩阵节点需要原始数据 / N2 feature matrix node needs raw data
        required_keys = ['train_df', 'test_df']
        validate_state_keys(state, required_keys, node_key)
        return {
            'train_data': state['train_df'],
            'test_data': state['test_df']
        }

    elif node_key == 'N1':
        # N1 缺失值处理节点需要特征矩阵和标签
        required_keys = ['X_train', 'X_val', 'y_train', 'y_val']
        validate_state_keys(state, required_keys, node_key)
        return {
            'X_train': state['X_train'],
            'X_val': state['X_val'],
            'X_test': state.get('X_test'),
            'y_train': state.get('y_train'),
            'y_val': state.get('y_val'),
            'y_test': state.get('y_test')
        }

    elif node_key in ('N3', 'N4'):
        # N3/N4 特征选择与缩放，传递全部 X/y
        required_keys = ['X_train', 'X_val','X_test', 'y_train', 'y_val','y_test']
        validate_state_keys(state, required_keys, node_key)
        X_train = state['X_train']
        X_val = state['X_val']
        y_train = state['y_train']
        y_val = state['y_val']
        # 处理缺失值样本（如有）
        bad_train = np.any(pd.isna(X_train), axis=1)
        bad_val = np.any(pd.isna(X_val), axis=1)
        train_idx = np.where(bad_train)[0]
        val_idx = np.where(bad_val)[0]
        if len(train_idx) > 0 or len(val_idx) > 0:
            if verbose:
                print(f" before {node_key}: drop {len(train_idx)} train / {len(val_idx)} val samples due to missing values")
            X_train = np.delete(X_train, train_idx, axis=0)
            y_train = np.delete(y_train, train_idx, axis=0)
            X_val = np.delete(X_val, val_idx, axis=0)
            y_val = np.delete(y_val, val_idx, axis=0)
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': state.get('X_test'),
            'y_train': y_train,
            'y_val': y_val,
            'y_test': state.get('y_test'),
            'feature_names': state.get('feature_names'),
        }

    elif node_key == 'N5':
        # N5 模型训练节点：确保标签已分割 / N5 training node: ensure labels are split
        if 'y_train' not in state:
            if verbose:
                print("    [INFO] 分割标签数据 / Splitting label data...")
            split_labels(state)
        return {
            'X_train': state.get('X_train'),
            'X_val': state.get('X_val'),
            'X_test': state.get('X_test'),
            'y_train': state.get('y_train'),
            'y_val': state.get('y_val'),
            'y_test': state.get('y_test'),
            'feature_names': state.get('feature_names')
        }

    else:
        # 其他节点使用完整状态 / Other nodes use full state
        return state


def validate_state_keys(state: Dict[str, Any], required_keys: list, node_key: str) -> None:
    """
    验证状态字典中是否包含必需的键。
    Validate that the state dict contains required keys.
    """
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
        raise KeyError(f"节点 {node_key} 缺少必需的状态键: {missing_keys} / Node {node_key} missing required state keys: {missing_keys}")


def split_labels(state: Dict[str, Any]) -> None:
    """
    分割标签数据为训练集和验证集。
    Split label data into train and validation sets.
    """
    if 'train_df' not in state:
        raise KeyError("State中缺少train_df / Missing train_df in state")
    y_full = state['train_df'][TARGET_PROP].values
    split_idx = int(0.8 * len(y_full))
    state['y_train'], state['y_val'] = y_full[:split_idx], y_full[split_idx:]
    # 如果有测试集，也提取测试标签 / If test set exists, extract test labels too
    if state.get('test_df') is not None:
        state['y_test'] = state['test_df'][TARGET_PROP].values


def update_state(node_key: str,
                 node_output: Dict[str, Any],
                 state: Dict[str, Any],
                 verbose: bool = False) -> None:
    """Merge node output and safeguard state consistency."""
    # ---------- 0) 通用：合并输出 ----------
    state.update(node_output or {})

    # ---------- 1) N0: 数据抓取节点 ----------
    if node_key == 'N0':
        # 1.1 取出三份 DataFrame
        train_df = node_output.get('train_data')
        test_df  = node_output.get('test_data')
        full_df  = node_output.get('full_data')

        # 1.2 严格校验
        for name, df_ in [('train_data', train_df),
                          ('test_data',  test_df),
                          ('full_data',  full_df)]:
            if df_ is None or not isinstance(df_, pd.DataFrame):
                raise RuntimeError(f"N0 输出缺少有效的 {name}")

        # 1.3 写回 state
        state['train_df'] = train_df
        state['test_df']  = test_df
        state['full_df']  = full_df

        # 1.4 ⚠️ 清理占位 y_*（避免 len(None)）
        for k in ('y_train', 'y_val', 'y_test',
                  'X_train', 'X_val', 'X_test'):   # ← 加上这三项
            if state.get(k) is None:
                state.pop(k, None)
    # ---------- 2) N2: 特征矩阵节点 ----------
    elif node_key == 'N2':
        if node_output.get('y_train') is None or node_output.get('y_val') is None:
            raise RuntimeError('N2 输出缺少 y_train 或 y_val')
    
    elif node_key in ('N1', 'N3', 'N4'):
        #  确认中间节点输出包含完整的训练/验证特征和标签 / Validate intermediate node outputs
        if node_output.get('X_train') is None or node_output.get('X_val') is None:
            raise RuntimeError(f"{node_key} 输出缺少 X_train 或 X_val")
        if node_output.get('y_train') is None or node_output.get('y_val') is None:
            raise RuntimeError(f"{node_key} 输出缺少 y_train 或 y_val")

    # ---------- 3) N5: 训练节点 ----------
    elif node_key == 'N5':
        if 'y_val_pred' not in node_output:
            raise RuntimeError('N5 输出缺少/ is lack of  y_val_pred')
 

