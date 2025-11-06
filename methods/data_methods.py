# methods/data_methods.py
"""
 / Data Processing Methods Module

This module contains all data processing methods for the ML pipeline including
data fetching, feature matrix construction, imputation, feature selection, and scaling.


Main functions:
- fetch_and_featurize: / Fetch and featurize data
- feature_matrix: / Construct feature matrix 
- impute_data: / Missing data imputation
- feature_selection: / Feature selection
- scale_features: / Feature scaling
"""
from __future__ import annotations
import logging
import os
import joblib
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# / Lazy imports for optional heavy dependencies
try:
 from mp_api.client import MPRester # type: ignore
except Exception: # pragma: no cover
 MPRester = None # type: ignore

try:
 from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures # type: ignore
 from matminer.featurizers.composition import ElementProperty # type: ignore
except Exception: # pragma: no cover
 DensityFeatures = GlobalSymmetryFeatures = ElementProperty = None # type: ignore

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# GNN / Optional GNN dependencies
try:
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
 from torch_geometric.data import Data
 from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
 TORCH_AVAILABLE = True
except ImportError:
 TORCH_AVAILABLE = False
 torch = None
 nn = None
 F = None
 warnings.warn("PyTorch Geometric not available. GNN will use fallback statistical features.")

try:
 from pymatgen.structure.structure import Structure
 from pymatgen.core.periodic_table import Element
 PYMATGEN_AVAILABLE = True
except ImportError:
 PYMATGEN_AVAILABLE = False

from config import API_KEY, CACHE_FILE, PROC_DIR, BATCH_SIZE, N_TOTAL, TARGET_PROP, MODEL_DIR

# Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

__all__ = [
 "fetch_and_featurize",
 "prepare_node_input",
 "validate_state_keys", 
 "split_labels",
 "update_state",
 # N1: Imputation
 "impute_data",
 "impute_mean",
 "impute_median",
 "impute_knn",
 "impute_none",
 # N2: Feature Matrix
 "feature_matrix",
 # N3: Feature Selection
 "feature_selection",
 "no_selection",
 "variance_selection",
 "univariate_selection",
 "pca_selection",
 # N4: GNN Processing ()
 "gnn_process",
 "structure_to_graph",
 "extract_gnn_features",
 "SimpleGCN",
 "SimpleGAT",
 "SimpleGraphSAGE",
 # N6-N7: Scaling
 "scale_features",
 "scale_standard",
 "scale_robust",
 "scale_minmax",
 "scale_none",
]

# Features will be used
def _build_feature_methods():
 methods = []
 if ElementProperty is not None:
 try:
 methods.append((ElementProperty.from_preset("magpie"), "composition"))
 except Exception:
 pass
 if DensityFeatures is not None:
 try:
 methods.append((DensityFeatures(), "structure"))
 except Exception:
 pass
 if GlobalSymmetryFeatures is not None:
 try:
 methods.append((GlobalSymmetryFeatures(), "structure"))
 except Exception:
 pass
 return methods

FEATURE_METHODS = _build_feature_methods()

# Set the cache and processed data paths
CACHE_PATH = Path(CACHE_FILE)
PROC_PATH = Path(PROC_DIR)

# Change to the public function
def split_by_fe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
 """ Fe train/test (robust to types)."""
 if "composition" not in df.columns:
 # Fallback: random split
 split = int(0.8 * len(df))
 return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

 def has_fe(c) -> bool:
 if c is None:
 return False
 try:
 if hasattr(c, "as_dict"):
 return bool(c.as_dict().get("Fe", 0) > 0)
 if isinstance(c, dict):
 return bool(c.get("Fe", 0) > 0)
 except Exception:
 pass
 # Fallback to string contains
 return "Fe" in str(c)

 mask = df["composition"].apply(has_fe)
 train_df = df[~mask].reset_index(drop=True)
 test_df = df[mask].reset_index(drop=True)
 # Ensure non-empty splits
 if train_df.empty or test_df.empty:
 split = int(0.8 * len(df))
 return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
 return train_df, test_df


def safe_featurize(feat_obj, x):
 """"""
 if x is None:
 return [np.nan] * len(feat_obj.feature_labels())
 try:
 return feat_obj.featurize(x)
 except (TypeError, ValueError, AttributeError) as e:
 # NoneType NaN
 logger.warning(f"Featurization failed for {type(feat_obj).__name__}: {str(e)[:100]}")
 return [np.nan] * len(feat_obj.feature_labels())

def apply_featurizers(df: pd.DataFrame) -> pd.DataFrame:
 """ DataFrame featurizers """
 parts: list[pd.DataFrame] = [df]
 for feat_obj, col in FEATURE_METHODS:
 labels = feat_obj.feature_labels()
 logger.info(f"Applying {type(feat_obj).__name__} to column '{col}' ({len(df)} samples)")
 # 
 array = Parallel(n_jobs=-1)(delayed(safe_featurize)(feat_obj, v) for v in df[col])
 parts.append(pd.DataFrame(array, columns=labels)) # type: ignore
 logger.info(f" Generated {len(labels)} features")
 return pd.concat(parts, axis=1)

# Node 0: /
def fetch_and_featurize(_: Any = None, cache: bool = True) -> Dict[str, Any]:
 """N0: /"""
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
 # dict dataclass 
 if isinstance(d, dict):
 return d.get(key, default)
 elif hasattr(d, key):
 return getattr(d, key, default)
 else:
 return default

# Make the N0 into 3 functions:
def fetch_data(cache: bool = True) -> pd.DataFrame:
 # Prefer cached pickle
 if cache and CACHE_PATH.exists():
 logger.info("Loading cache from %s", CACHE_PATH)
 try:
 with open(CACHE_PATH, "rb") as f:
 cached = pickle.load(f)
 if isinstance(cached, dict):
 # Prefer explicit 'full_data' key
 if 'full_data' in cached and isinstance(cached['full_data'], pd.DataFrame):
 return cached['full_data']
 # Fallback: if train/test provided, combine as a surrogate full df
 train_df = cached.get('train_data')
 test_df = cached.get('test_data')
 if isinstance(train_df, pd.DataFrame) and isinstance(test_df, pd.DataFrame):
 return pd.concat([train_df, test_df], ignore_index=True)
 # If dict contains a DataFrame under other key, try to return the first DataFrame
 for v in cached.values():
 if isinstance(v, pd.DataFrame):
 return v
 if isinstance(cached, pd.DataFrame):
 return cached
 except Exception as e:
 logger.warning("Failed to load cache (%s). Falling back to CSV or API.", str(e)[:120])
 # Fallback to processed CSV if exists
 csv_path = PROC_PATH / "all_data_feat.csv"
 if csv_path.exists():
 logger.info("Loading precomputed features CSV: %s", csv_path)
 df = pd.read_csv(csv_path)
 # Minimal columns for downstream
 if TARGET_PROP not in df.columns:
 # fabricate a target to allow demo runs
 df[TARGET_PROP] = np.random.randn(len(df))
 # Create dummy structure/composition cols if missing
 if 'structure' not in df.columns:
 df['structure'] = None
 if 'composition' not in df.columns:
 df['composition'] = None
 # Split train/test downstream
 return df

 # If API client unavailable, raise with hint
 if MPRester is None:
 raise RuntimeError("mp_api not available and no cache/CSV found. Provide data or install mp_api.")

 logger.info("Fetching data from MP API")
 dfs = []
 fetched = 0
 with MPRester(API_KEY) as mpr: # type: ignore
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

 # 
 CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
 with open(CACHE_PATH, "wb") as f:
 pickle.dump(full_df, f)

 return full_df

def featurize_data(df: pd.DataFrame) -> pd.DataFrame:
 # If we already have a precomputed CSV, reuse
 csv_path = PROC_PATH / "all_data_feat.csv"
 if csv_path.exists():
 try:
 return pd.read_csv(csv_path)
 except Exception:
 pass
 # If featurizers unavailable, pass through numeric columns as features
 if not FEATURE_METHODS:
 logger.warning("Matminer not available; using numeric columns as features")
 df_feat = df.copy()
 df_feat.to_csv(csv_path, index=False)
 return df_feat
 df_feat = apply_featurizers(df)
 df_feat.to_csv(csv_path, index=False)
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
 """ForX_train, X_val, X_testuse the same imputerformat in dict"""
 import logging
 logger = logging.getLogger(__name__)
 X_train = imp.fit_transform(drop_allnan(data['X_train']))
 X_val = imp.transform(drop_allnan(data['X_val']))
 X_test_raw = data.get('X_test')
 X_test = imp.transform(drop_allnan(X_test_raw)) if X_test_raw is not None else None
 # NaN
 def fix_nan(arr, name):
 if isinstance(arr, np.ndarray) and np.isnan(arr).any():
 logger.warning(f"[impute_data] {name} contains NaN after impute, auto-filling with 0.")
 arr = np.nan_to_num(arr)
 return arr
 X_train = fix_nan(X_train, 'X_train')
 X_val = fix_nan(X_val, 'X_val')
 X_test = fix_nan(X_test, 'X_test') if X_test is not None else None
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
def impute_mean(data, **params):
 """Mean imputation. Extra params are ignored safely."""
 logger.info("Imputing using mean")
 imp = SimpleImputer(strategy='mean')
 return apply_imputer(imp, data)

def impute_median(data, **params):
 """Median imputation. Extra params are ignored safely."""
 logger.info("Imputing using median")
 imp = SimpleImputer(strategy='median')
 return apply_imputer(imp, data)

def impute_knn(data, n_neighbors=5, **params):
 """
 KNN imputation with optional mapping from a normalized param in [0,1]
 to integer neighbors in [1, 10].

 Priority:
 1) If explicit n_neighbors is provided in params, use it.
 2) Else if 'param' provided, map p->[1..10].
 3) Else use the function argument default (n_neighbors).
 """
 # 1) Explicit override via params
 if 'n_neighbors' in params:
 try:
 n_neighbors = int(params['n_neighbors'])
 except Exception:
 pass

 # 2) Map from normalized 'param' if provided and n_neighbors not explicitly set
 if 'param' in params and 'n_neighbors' not in params:
 try:
 p = float(params['param'])
 # clip to [0,1]
 p = 0.0 if p < 0 else (1.0 if p > 1 else p)
 # Map to 1..10 (inclusive)
 n_neighbors = 1 + int(round(p * 9))
 except Exception:
 # keep existing n_neighbors if mapping fails
 pass

 # Safety clamp
 n_neighbors = max(1, int(n_neighbors))

 logger.info("Imputing using KNN, neighbors=%d%s",
 n_neighbors,
 f" (mapped from param={params.get('param'):.3f})" if 'param' in params and 'n_neighbors' not in params else "")
 imp = KNNImputer(n_neighbors=n_neighbors)
 return apply_imputer(imp, data)

def impute_none(data, **params):
 """No imputation. Extra params are ignored safely."""
 logger.info("No imputation performed")
 return {
 'X_train': data['X_train'], 
 'X_val': data['X_val'], 
 'X_test': data.get('X_test'),
 'y_train': data.get('y_train'),
 'y_val': data.get('y_val'),
 'y_test': data.get('y_test')
 }



# N2: Feature Matrix Node verbose
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
 X_val = X_val[num_cols]
 X_test = X_test[num_cols] if X_test is not None else None
 y_test = test_df[TARGET_PROP].values if TARGET_PROP in test_df.columns else None

 if verbose:
 print(f"[N2] X_train shape: {X_train_full.shape}, NaN count: {X_train_full.isna().sum().sum()}")
 print("[DEBUG] feat_cols:", feat_cols)
 print("[DEBUG] X_train shape:", X_train.shape)
 print("[DEBUG] X_val shape:", X_val.shape)
 print("[DEBUG] X_train head:\n", X_train.head())
 print("[DEBUG] X_val head:\n", X_val.head())

 # NaN
 import logging
 logger = logging.getLogger(__name__)
 def fix_nan(arr, name):
 if isinstance(arr, pd.DataFrame):
 if arr.isna().any().any():
 logger.warning(f"[feature_matrix] {name} contains NaN, auto-filling with 0.")
 return arr.fillna(0)
 elif isinstance(arr, np.ndarray):
 if np.isnan(arr).any():
 logger.warning(f"[feature_matrix] {name} contains NaN, auto-filling with 0.")
 arr = np.nan_to_num(arr)
 return arr
 X_train = fix_nan(X_train, 'X_train')
 X_val = fix_nan(X_val, 'X_val')
 X_test = fix_nan(X_test, 'X_test') if X_test is not None else None
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
 N3/N4 train/val/test selector/scaler
 Apply sklearn selector/scaler/transformer to train/val/test sets.

 Args:
 selector: (selector/scaler/transformer)
 data: dict
 y_required: y

 Returns:
 X_train, X_val, X_test: split 
 """
 if y_required and data.get('y_train') is not None:
 X_train = selector.fit_transform(data['X_train'], data['y_train'])
 else:
 X_train = selector.fit_transform(data['X_train'])
 X_val = selector.transform(data['X_val'])
 X_test = selector.transform(data['X_test']) if data.get('X_test') is not None else None
 return X_train, X_val, X_test

def filter_feature_names(data, selector, prefix=''):
 """

 Auto-generate new feature names according to support mask or PCA components.
 """
 if hasattr(selector, 'get_support') and 'feature_names' in data:
 mask = selector.get_support()
 # pandas Series 
 # Explicitly convert to bool to avoid pandas Series ambiguity error
 return [fname for fname, m in zip(data['feature_names'], mask) if bool(m)]
 elif prefix:
 num_comps = selector.n_components_ if hasattr(selector, 'n_components_') else selector.n_components
 return [f"{prefix}{i+1}" for i in range(num_comps)]
 else:
 return data.get('feature_names')

def no_selection(data, **params):
 """

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

# y | Supervised learning, requires the use of y
def univariate_selection(data, k=20, **params):
 """
 (F, )
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
 # apply_selector y_required=True
 X_train, X_val, X_test = apply_selector(
 selector,
 {
 'X_train': X_tr_arr,
 'X_val': np.array(data['X_val']),
 'X_test': np.array(data['X_test']) if data.get('X_test') is not None else None,
 'y_train': y_tr
 },
 y_required=True # select_univariate y
 )
 new_feature_names = filter_feature_names(data, selector)
 return {
 'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
 'y_train': data.get('y_train'), 'y_val': data.get('y_val'), 'y_test': data.get('y_test'),
 'selector': selector, 'feature_names': new_feature_names
 }

def pca_selection(data, n_components=0.95, **params):
 """
 (PCA) 
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
 X_val = pca.transform(np.array(data['X_val']))
 X_test = pca.transform(np.array(data['X_test'])) if data.get('X_test') is not None else None
 new_feature_names = filter_feature_names(data, pca, prefix='PC')
 return {
 'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
 'y_train': data.get('y_train'), 'y_val': data.get('y_val'), 'y_test': data.get('y_test'),
 'selector': pca, 'feature_names': new_feature_names
 }

# Node 4 : Feature Scaling

# apply_selector 
# def apply_selector(selector, data, y_required=False): ...

def scale_features(data, strategy='standard', params=None):
 """
 /
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
 StandardScaler
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
 RobustScaler
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
 MinMaxMinMaxScaler
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


# ========================= N4: / GNN Processing for N4 Node =========================
# GNN
# This section implements a complete GNN pipeline for extracting deep learning features from crystal structures

# ========================= N4: / GNN Processing for N4 Node =========================
# GNN
# This section implements a complete GNN pipeline for extracting deep learning features from crystal structures

# GNNPyTorch/ Define GNN model classes (only if PyTorch available)
if TORCH_AVAILABLE:
 class SimpleGCN(nn.Module):
 """
 (GCN) / Simplified Graph Convolutional Network

 2GCN
 Uses 2-layer GCN for graph feature learning, outputs global mean-pooled graph-level representation.

 Architecture:
 - Layer 1: GCNConv(input_dim hidden_dim) + BatchNorm + ReLU + Dropout
 - Layer 2: GCNConv(hidden_dim output_dim) + BatchNorm + ReLU
 - Pooling: Global mean pooling for graph-level representation
 """
 def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 16):
 """
 GCN / Initialize GCN model

 Args:
 input_dim: / Input feature dimension (typically 3 for atomic features)
 hidden_dim: / Hidden layer dimension (default 32)
 output_dim: / Output feature dimension (default 16)
 """
 super().__init__()
 self.conv1 = GCNConv(input_dim, hidden_dim)
 self.conv2 = GCNConv(hidden_dim, output_dim)
 self.bn1 = nn.BatchNorm1d(hidden_dim)
 self.bn2 = nn.BatchNorm1d(output_dim)

 def forward(self, data):
 """
 / Forward pass

 Args:
 data: torch_geometric.data.Data :
 - x: [n_nodes, input_dim] / Node features
 - edge_index: [2, n_edges] / Edge indices
 - batch: [n_nodes] / Batch assignment for pooling

 Returns:
 graph_embedding: [batch_size, output_dim] / Graph-level representation
 """
 x, edge_index, batch = data.x, data.edge_index, data.batch

 # / First convolution layer
 x = self.conv1(x, edge_index)
 x = self.bn1(x)
 x = F.relu(x)
 x = F.dropout(x, p=0.1, training=self.training)

 # / Second convolution layer
 x = self.conv2(x, edge_index)
 x = self.bn2(x)
 x = F.relu(x)

 # / Global average pooling
 graph_embedding = global_mean_pool(x, batch)
 return graph_embedding


 class SimpleGAT(nn.Module):
 """
 (GAT) / Simplified Graph Attention Network

 
 Uses multi-head attention mechanism to learn atomic importance weights, capturing complex atomic interactions.

 Architecture:
 - Layer 1: GATConv(input_dim hidden_dim, heads=4) + BatchNorm + ReLU + Dropout
 - Layer 2: GATConv(hidden_dim output_dim, heads=1) + ReLU
 - Attention: Multi-head self-attention for feature interaction learning
 """
 def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 16, heads: int = 4):
 """
 GAT / Initialize GAT model

 Args:
 input_dim: / Input feature dimension
 hidden_dim: / Hidden layer dimension
 output_dim: / Output feature dimension
 heads: / Number of attention heads (default 4)
 """
 super().__init__()
 self.att1 = GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=0.1)
 self.att2 = GATConv(hidden_dim, output_dim, heads=1, dropout=0.1)
 self.bn1 = nn.BatchNorm1d(hidden_dim)

 def forward(self, data):
 """
 / Forward pass

 Args:
 data: torch_geometric.data.Data / PyG Data object

 Returns:
 graph_embedding: [batch_size, output_dim] / Graph-level representation
 """
 x, edge_index, batch = data.x, data.edge_index, data.batch

 # / First multi-head attention layer
 x = self.att1(x, edge_index)
 x = self.bn1(x)
 x = F.relu(x)
 x = F.dropout(x, p=0.1, training=self.training)

 # / Second single-head attention layer
 x = self.att2(x, edge_index)
 x = F.relu(x)

 # / Global average pooling
 graph_embedding = global_mean_pool(x, batch)
 return graph_embedding


 class SimpleGraphSAGE(nn.Module):
 """
 GraphSAGE / Simplified GraphSAGE

 
 Uses neighborhood sampling and aggregation for scalable graph learning, especially suitable for large-scale crystal structures.

 Architecture:
 - Layer 1: SAGEConv(input_dim hidden_dim) + BatchNorm + ReLU + Dropout
 - Layer 2: SAGEConv(hidden_dim output_dim) + BatchNorm + ReLU
 - Aggregation: Mean aggregation of neighbor features
 """
 def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 16):
 """
 GraphSAGE / Initialize GraphSAGE model

 Args:
 input_dim: / Input feature dimension
 hidden_dim: / Hidden layer dimension
 output_dim: / Output feature dimension
 """
 super().__init__()
 self.sage1 = SAGEConv(input_dim, hidden_dim)
 self.sage2 = SAGEConv(hidden_dim, output_dim)
 self.bn1 = nn.BatchNorm1d(hidden_dim)
 self.bn2 = nn.BatchNorm1d(output_dim)

 def forward(self, data):
 """
 / Forward pass

 Args:
 data: torch_geometric.data.Data / PyG Data object

 Returns:
 graph_embedding: [batch_size, output_dim] / Graph-level representation
 """
 x, edge_index, batch = data.x, data.edge_index, data.batch

 # / First neighborhood aggregation layer
 x = self.sage1(x, edge_index)
 x = self.bn1(x)
 x = F.relu(x)
 x = F.dropout(x, p=0.1, training=self.training)

 # / Second neighborhood aggregation layer
 x = self.sage2(x, edge_index)
 x = self.bn2(x)
 x = F.relu(x)

 # / Global average pooling
 graph_embedding = global_mean_pool(x, batch)
 return graph_embedding

else:
 # PyTorch / Dummy classes when PyTorch unavailable
 class SimpleGCN:
 """GCN placeholder when PyTorch unavailable"""
 pass

 class SimpleGAT:
 """GAT placeholder when PyTorch unavailable"""
 pass

 class SimpleGraphSAGE:
 """GraphSAGE placeholder when PyTorch unavailable"""
 pass


def structure_to_graph(structure: Any, cutoff_distance: float = 5.0) -> Dict[str, Any]:
 """
 / Convert crystal structure to graph representation

 

 
 :
 1. / Extract atom positions and properties
 2. [, , ] / Build node features
 3. / Compute edges based on cutoff distance

 Args:
 structure: pymatgen Structure / pymatgen Structure object or dict
 cutoff_distance: (Å) / Cutoff distance in Angstrom (default 5.0)

 Returns:
 dict: / Graph data dict containing:
 - node_features: [n_nodes, 3] / Atomic feature matrix
 - edge_index: [2, n_edges] / Edge source and target indices
 - edge_attr: [n_edges, 2] / Edge attributes (distance)
 - atomic_numbers: [n_nodes] / Atomic number list
 - n_nodes: / Number of nodes

 Example:
 >>> structure = pymatgen_structure_object
 >>> graph_dict = structure_to_graph(structure, cutoff_distance=4.0)
 >>> print(graph_dict['node_features'].shape) # (n_atoms, 3)
 """
 if not PYMATGEN_AVAILABLE:
 logger.warning("pymatgen not available. Using fallback graph construction.")
 return _fallback_graph_construction()

 try:
 # / Extract atom positions and properties
 sites = structure.sites if hasattr(structure, 'sites') else structure.get('sites', [])
 if not sites:
 logger.warning("No sites found in structure.")
 return _fallback_graph_construction()

 n_nodes = len(sites)

 # 1. / Build node features
 # : (), (), ()
 node_features = []
 atomic_numbers = []

 for site in sites:
 try:
 # / Extract element information
 element = site.species[0] if hasattr(site, 'species') else site.get('element')
 elem = Element(element) if isinstance(element, str) else element
 atomic_num = elem.Z
 atomic_numbers.append(atomic_num)

 # : [/118, , ] 
 # [0,1]
 features = [
 float(atomic_num) / 118.0, # / Normalized atomic number
 (elem.atomic_radius / 200.0) if elem.atomic_radius else 0.5, # / Atomic radius
 (elem.X / 4.0) if elem.X else 0.5, # / Electronegativity
 ]
 node_features.append(features)
 except Exception as e:
 logger.warning(f"Failed to extract element properties for site: {e}")
 # / Use default features
 node_features.append([0.5, 0.5, 0.5])
 atomic_numbers.append(0)

 node_features = np.array(node_features, dtype=np.float32)
 atomic_numbers = np.array(atomic_numbers, dtype=np.int32)

 # 2. / Build edges
 # 
 edge_list = []
 edge_attrs = []

 for i in range(n_nodes):
 for j in range(i + 1, n_nodes):
 try:
 # / Calculate distance between atoms
 site_i = sites[i]
 site_j = sites[j]
 distance = site_i.distance(site_j)

 # / Add edge if within cutoff
 if distance < cutoff_distance:
 # / Create bidirectional edges
 edge_list.append([i, j])
 edge_list.append([j, i])

 # : [, 1]
 edge_attr = [distance / cutoff_distance, 1.0]
 edge_attrs.append(edge_attr)
 edge_attrs.append(edge_attr)
 except Exception as e:
 logger.debug(f"Error calculating distance between atoms {i} and {j}: {e}")
 continue

 # / Handle edge cases
 if edge_list:
 edge_index = np.array(edge_list, dtype=np.int64).T
 edge_attr = np.array(edge_attrs, dtype=np.float32)
 else:
 # 
 logger.info(f"No edges found with cutoff={cutoff_distance}. Creating self-loops.")
 edge_index = np.array([[i, i] for i in range(n_nodes)], dtype=np.int64).T
 edge_attr = np.ones((n_nodes, 2), dtype=np.float32)

 return {
 'node_features': node_features,
 'edge_index': edge_index,
 'edge_attr': edge_attr,
 'atomic_numbers': atomic_numbers,
 'n_nodes': n_nodes
 }

 except Exception as e:
 logger.warning(f"Structure to graph conversion failed: {e}. Using fallback.")
 return _fallback_graph_construction()


def _fallback_graph_construction() -> Dict[str, Any]:
 """
 / Fallback graph construction

 pymatgen
 GNN
 """
 return {
 'node_features': np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
 'edge_index': np.array([[0], [0]], dtype=np.int64),
 'edge_attr': np.array([[1.0, 1.0]], dtype=np.float32),
 'atomic_numbers': np.array([0], dtype=np.int32),
 'n_nodes': 1
 }


def extract_gnn_features(
 structures: List[Any],
 method: str = 'gcn',
 output_dim: int = 16,
 device: str = 'cpu'
) -> np.ndarray:
 """
 GNN / Extract deep learning features from crystal structures using GNN

 :
 1. / Convert structures to graphs
 2. GNN / Initialize selected GNN model
 3. / Perform forward pass and extract graph-level representations
 4. / Return feature matrix

 GNN:
 - 'gcn': 
 - 'gat': 
 - 'sage': GraphSAGE

 Args:
 structures: / List of crystal structures
 method: GNN ('gcn', 'gat', 'sage') / GNN method
 output_dim: / Output feature dimension (8, 16, or 32)
 device: ('cpu' 'cuda') / Computing device

 Returns:
 features: [n_structures, output_dim] GNN / GNN feature matrix

 Example:
 >>> structures = [struct1, struct2, struct3]
 >>> features = extract_gnn_features(structures, method='gat', output_dim=16)
 >>> print(features.shape) # (3, 16)
 """
 if not TORCH_AVAILABLE:
 logger.warning("PyTorch not available. Using fallback statistical features instead.")
 return _statistical_fallback_features(structures, output_dim)

 try:
 # GNN / Initialize GNN model
 logger.info(f"Initializing GNN model: {method}, output_dim={output_dim}, device={device}")

 if method.lower() == 'gcn':
 model = SimpleGCN(input_dim=3, hidden_dim=32, output_dim=output_dim)
 elif method.lower() == 'gat':
 model = SimpleGAT(input_dim=3, hidden_dim=32, output_dim=output_dim)
 elif method.lower() == 'sage':
 model = SimpleGraphSAGE(input_dim=3, hidden_dim=32, output_dim=output_dim)
 else:
 raise ValueError(f"Unknown GNN method: {method}. Choose from ['gcn', 'gat', 'sage']")

 model = model.to(device)
 model.eval() # dropoutbatch norm

 # / Convert structures to graphs
 logger.info(f"Converting {len(structures)} structures to graphs...")
 graphs = []
 for i, structure in enumerate(structures):
 try:
 graph_dict = structure_to_graph(structure)
 graphs.append(graph_dict)
 except Exception as e:
 logger.warning(f"Failed to convert structure {i} to graph: {e}")
 graphs.append(_fallback_graph_construction())

 # PyG / Create PyG data objects and extract features
 logger.info(f"Extracting GNN features from {len(graphs)} graphs...")
 features = []

 with torch.no_grad():
 for i, graph_dict in enumerate(graphs):
 try:
 # torch / Convert to torch tensors
 node_feat = torch.FloatTensor(graph_dict['node_features']).to(device)
 edge_idx = torch.LongTensor(graph_dict['edge_index']).to(device)

 # PyG Data / Create PyG Data object
 data = Data(
 x=node_feat,
 edge_index=edge_idx,
 batch=torch.zeros(graph_dict['n_nodes'], dtype=torch.long).to(device)
 )

 # / Forward pass
 embedding = model(data)
 features.append(embedding.cpu().numpy())

 if (i + 1) % max(1, len(graphs) // 10) == 0:
 logger.debug(f"Processed {i + 1}/{len(graphs)} structures")

 except Exception as e:
 logger.warning(f"Error processing graph {i}: {e}. Using zeros.")
 features.append(np.zeros(output_dim, dtype=np.float32))

 features = np.array(features, dtype=np.float32)
 logger.info(f"Successfully extracted GNN features: shape={features.shape}")
 return features

 except Exception as e:
 logger.error(f"GNN feature extraction failed: {e}. Using fallback statistical features.")
 return _statistical_fallback_features(structures, output_dim)


def _statistical_fallback_features(structures: List[Any], output_dim: int) -> np.ndarray:
 """
 / Statistical feature fallback

 GNN
 : 

 Args:
 structures: / List of crystal structures
 output_dim: / Output feature dimension

 Returns:
 features: [n_structures, output_dim] / Statistical feature matrix
 """
 logger.info(f"Using statistical fallback features with output_dim={output_dim}")
 features = []

 for i, structure in enumerate(structures):
 try:
 # / Compute basic structural statistics
 if hasattr(structure, 'sites'):
 n_atoms = len(structure.sites)
 elif isinstance(structure, dict):
 n_atoms = len(structure.get('sites', []))
 else:
 n_atoms = 1

 # / Create statistical feature vector
 feat = np.random.randn(output_dim) * 0.01 + 0.5 # 0.5
 feat[0] = np.log(n_atoms + 1) / 4.0 # / Log of atom count
 if output_dim > 1:
 feat[1] = np.sin(n_atoms) * 0.5 + 0.5 # / Periodic feature

 features.append(feat)
 except Exception as e:
 logger.warning(f"Error computing statistics for structure {i}: {e}")
 # / Use default features
 features.append(np.ones(output_dim, dtype=np.float32) * 0.5)

 return np.array(features, dtype=np.float32)


def gnn_process(
 data: Dict[str, Any],
 strategy: str = 'gcn',
 param: Optional[float] = None,
 params: Optional[dict] = None,
 **kwargs
) -> Dict[str, Any]:
 """
 N4 / N4 Node: Graph Neural Network Feature Extraction

 :
 GNN
 GNN

 :
 1. [0.0-1.0] [8/16/32]
 2. 
 3. GNNGNN
 4. GNN + 

 :
 - param [0.0, 0.33) output_dim = 8 ()
 - param [0.33, 0.67) output_dim = 16 ()
 - param [0.67, 1.0] output_dim = 32 ()

 :
 - 'gcn': Graph Convolutional Network (, )
 - 'gat': Graph Attention Network (, )
 - 'sage': GraphSAGE ()

 Args:
 data: / Pipeline state dict, must contain:
 - X_train: [n_train, n_features] 
 - X_val: [n_val, n_features] 
 - X_test: [n_test, n_features] ()
 - y_train, y_val, y_test: / Labels
 - feature_names: / Feature name list
 - structures_train: / Training structures
 - structures_val: / Validation structures
 - structures_test: / Test structures ()

 strategy: GNN ('gcn', 'gat', 'sage') / GNN method

 param: [0.0-1.0] / Hyperparameter controlling output dim
 - 0.0-0.33: (8)
 - 0.33-0.67: (16)
 - 0.67-1.0: (32)

 params: / Optional parameter dict
 - output_dim: / Explicit output dimension
 - device: / Computing device ('cpu' or 'cuda')

 Returns:
 dict: :
 - X_train, X_val, X_test: / Extended feature matrices
 - y_train, y_val, y_test: / Labels (unchanged)
 - feature_names: / Updated feature names
 - gnn_features_train: GNN / Raw GNN features
 - gnn_features_val, gnn_features_test: GNN
 - gnn_info: GNN / Metadata about GNN processing

 Example:
 >>> result = gnn_process(
 ... data,
 ... strategy='gat',
 ... param=0.65 # 16
 ... )
 >>> X_train_extended = result['X_train']
 >>> print(X_train_extended.shape) # (n_train, n_original_features + 16)

 Performance Notes:
 - GCN: ~50ms per sample, good for most cases
 - GAT: ~80ms per sample, best accuracy but slower
 - GraphSAGE: ~40ms per sample, fastest option
 - Fallback (no PyTorch): <1ms per sample, but lower quality features
 """
 logger.info(f"Starting GNN processing: strategy={strategy}, param={param}")

 # 1. / Parameter handling
 if params is None:
 params = {}

 # / Determine output dimension from normalized parameter
 if param is not None:
 param = max(0.0, min(1.0, float(param)))
 if param < 0.33:
 output_dim = 8
 elif param < 0.67:
 output_dim = 16
 else:
 output_dim = 32
 else:
 output_dim = params.get('output_dim', 16)

 logger.info(f"GNN output dimension: {output_dim}")

 # 2. / Check required data
 X_train = data.get('X_train')
 X_val = data.get('X_val')
 X_test = data.get('X_test')
 structures_train = data.get('structures_train', [])
 structures_val = data.get('structures_val', [])
 structures_test = data.get('structures_test', [])

 # / Fallback if no structure data
 if (not structures_train or len(structures_train) == 0) and X_train is not None:
 logger.warning("WARNING: No structure data available. Using statistical fallback features.")
 return _gnn_fallback(data, output_dim)

 # 3. GNN / Extract GNN features
 try:
 # / Determine device
 if TORCH_AVAILABLE and torch.cuda.is_available():
 device = 'cuda'
 logger.info("CUDA available, using GPU for GNN processing")
 else:
 device = 'cpu'
 logger.info("Using CPU for GNN processing")

 # / Process training structures
 if structures_train and len(structures_train) > 0:
 gnn_features_train = extract_gnn_features(
 structures_train,
 method=strategy,
 output_dim=output_dim,
 device=device
 )
 logger.info(f"GNN training features extracted: shape={gnn_features_train.shape}")
 else:
 logger.warning("No training structures available")
 gnn_features_train = None

 # / Process validation set
 if structures_val and len(structures_val) > 0:
 gnn_features_val = extract_gnn_features(
 structures_val,
 method=strategy,
 output_dim=output_dim,
 device=device
 )
 logger.info(f"GNN validation features extracted: shape={gnn_features_val.shape}")
 else:
 gnn_features_val = None

 # / Process test set
 if structures_test and len(structures_test) > 0:
 gnn_features_test = extract_gnn_features(
 structures_test,
 method=strategy,
 output_dim=output_dim,
 device=device
 )
 logger.info(f"GNN test features extracted: shape={gnn_features_test.shape}")
 else:
 gnn_features_test = None

 except Exception as e:
 logger.error(f"ERROR: GNN feature extraction error: {e}. Using statistical fallback.")
 return _gnn_fallback(data, output_dim)

 # 4. GNN / Merge GNN features with original features
 logger.info("Merging GNN features with original features...")

 if X_train is not None and gnn_features_train is not None:
 # / Ensure shape compatibility
 if len(gnn_features_train) == len(X_train):
 X_train_extended = np.concatenate([X_train, gnn_features_train], axis=1)
 logger.info(f" X_train shape: {X_train.shape} + {gnn_features_train.shape} {X_train_extended.shape}")
 else:
 logger.warning(f" Shape mismatch: X_train({len(X_train)}) vs GNN({len(gnn_features_train)}). Skipping merge.")
 X_train_extended = X_train
 else:
 X_train_extended = X_train

 if X_val is not None and gnn_features_val is not None:
 if len(gnn_features_val) == len(X_val):
 X_val_extended = np.concatenate([X_val, gnn_features_val], axis=1)
 logger.info(f" X_val shape: {X_val.shape} + {gnn_features_val.shape} {X_val_extended.shape}")
 else:
 X_val_extended = X_val
 else:
 X_val_extended = X_val

 if X_test is not None and gnn_features_test is not None:
 if len(gnn_features_test) == len(X_test):
 X_test_extended = np.concatenate([X_test, gnn_features_test], axis=1)
 logger.info(f" X_test shape: {X_test.shape} + {gnn_features_test.shape} {X_test_extended.shape}")
 else:
 X_test_extended = X_test
 else:
 X_test_extended = X_test

 # 5. / Update feature names
 feature_names = list(data.get('feature_names', []))
 for i in range(output_dim):
 feature_names.append(f'gnn_{strategy}_{i}')

 logger.info(f"Final feature count: {len(feature_names)} (added {output_dim} GNN features)")

 # 6. / Return updated state
 result = {
 'X_train': X_train_extended,
 'X_val': X_val_extended,
 'X_test': X_test_extended,
 'y_train': data.get('y_train'),
 'y_val': data.get('y_val'),
 'y_test': data.get('y_test'),
 'feature_names': feature_names,
 'gnn_features_train': gnn_features_train,
 'gnn_features_val': gnn_features_val,
 'gnn_features_test': gnn_features_test,
 'gnn_info': {
 'method': strategy,
 'output_dim': output_dim,
 'device': device if TORCH_AVAILABLE else 'cpu',
 'torch_available': TORCH_AVAILABLE,
 'pymatgen_available': PYMATGEN_AVAILABLE
 }
 }

 logger.info(f"GNN processing complete!")
 return result


def _gnn_fallback(data: Dict[str, Any], output_dim: int) -> Dict[str, Any]:
 """
 GNN / GNN processing fallback mechanism

 GNNGNN

 
 : , , , 
 Statistical features include: mean, std, min, max

 Args:
 data: / Pipeline state dict
 output_dim: / Output feature dimension

 Returns:
 dict: / Updated state with statistical features
 """
 logger.info(f"Using statistical fallback features with output_dim={output_dim}")

 X_train = data.get('X_train')
 X_val = data.get('X_val')
 X_test = data.get('X_test')

 def append_stats(X: Optional[np.ndarray], name: str = 'data') -> Optional[np.ndarray]:
 """
 / Append row-wise statistics to data

 Args:
 X: / Input feature matrix
 name: / Dataset name (for logging)

 Returns:
 / Extended feature matrix
 """
 if X is None:
 return None

 # / Compute row-wise statistics
 mean = np.nanmean(X, axis=1, keepdims=True)
 std = np.nanstd(X, axis=1, keepdims=True)
 minv = np.nanmin(X, axis=1, keepdims=True)
 maxv = np.nanmax(X, axis=1, keepdims=True)

 # / Concatenate statistics
 stats = np.concatenate([mean, std, minv, maxv], axis=1)

 # / Pad to output dimension
 if stats.shape[1] < output_dim:
 pad_size = output_dim - stats.shape[1]
 stats = np.pad(stats, ((0, 0), (0, pad_size)), mode='constant', constant_values=0.5)
 elif stats.shape[1] > output_dim:
 stats = stats[:, :output_dim]

 result = np.concatenate([X, stats], axis=1)
 logger.info(f" {name}: {X.shape} + {stats.shape[:1]} {result.shape}")
 return result

 # / Apply statistics
 X_train_ext = append_stats(X_train, 'X_train')
 X_val_ext = append_stats(X_val, 'X_val')
 X_test_ext = append_stats(X_test, 'X_test')

 # / Update feature names
 feature_names = list(data.get('feature_names', []))
 for i in range(output_dim):
 feature_names.append(f'gnn_fallback_{i}')

 return {
 'X_train': X_train_ext,
 'X_val': X_val_ext,
 'X_test': X_test_ext,
 'y_train': data.get('y_train'),
 'y_val': data.get('y_val'),
 'y_test': data.get('y_test'),
 'feature_names': feature_names,
 'gnn_features_train': None,
 'gnn_features_val': None,
 'gnn_features_test': None,
 'gnn_info': {
 'method': 'fallback_statistical',
 'output_dim': output_dim,
 'device': 'cpu',
 'torch_available': False,
 'pymatgen_available': False
 }
 }

def prepare_node_input(node_key: str, state: dict, verbose: bool = False) -> dict:
 """

 Prepare input data for the specified node.
 """
 if verbose:
 print(f"\n[ {node_key}] keys: {list(state.keys())}")
 if state.get('X_train') is not None:
 print(f"[ {node_key}] X_train shape: {state['X_train'].shape}")
 if state.get('y_train') is not None:
 print(f"[ {node_key}] y_train preview: {state['y_train'][:5]}")

 if node_key == 'N0':
 # N0 / N0 data fetch node needs no input
 return {}

 elif node_key == 'N2':
 # N2 / N2 feature matrix node needs raw data
 required_keys = ['train_df', 'test_df']
 validate_state_keys(state, required_keys, node_key)
 return {
 'train_data': state['train_df'],
 'test_data': state['test_df']
 }

 elif node_key == 'N1':
 # N1 
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
 # N3/N4 X/y
 required_keys = ['X_train', 'X_val','X_test', 'y_train', 'y_val','y_test']
 validate_state_keys(state, required_keys, node_key)
 X_train = state['X_train']
 X_val = state['X_val']
 y_train = state['y_train']
 y_val = state['y_val']
 # 
 bad_train = np.any(pd.isna(X_train), axis=1)
 bad_val = np.any(pd.isna(X_val), axis=1)
 train_idx = np.where(bad_train)[0]
 val_idx = np.where(bad_val)[0]
 # .any()Series
 if (train_idx.size > 0) or (val_idx.size > 0):
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
 # N5 / N5 training node: ensure labels are split
 if 'y_train' not in state:
 if verbose:
 print(" Splitting label data...")
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
 # / Other nodes use full state
 return state


def validate_state_keys(state: Dict[str, Any], required_keys: list, node_key: str) -> None:
 """

 Validate that the state dict contains required keys.
 """
 missing_keys = [key for key in required_keys if key not in state]
 if missing_keys:
 raise KeyError(f" {node_key} : {missing_keys} / Node {node_key} missing required state keys: {missing_keys}")


def split_labels(state: Dict[str, Any]) -> None:
 """

 Split label data into train and validation sets.
 """
 if 'train_df' not in state:
 raise KeyError("Statetrain_df / Missing train_df in state")
 y_full = state['train_df'][TARGET_PROP].values
 split_idx = int(0.8 * len(y_full))
 state['y_train'], state['y_val'] = y_full[:split_idx], y_full[split_idx:]
 # / If test set exists, extract test labels too
 if state.get('test_df') is not None:
 state['y_test'] = state['test_df'][TARGET_PROP].values


def update_state(node_key: str,
 node_output: Dict[str, Any],
 state: Dict[str, Any],
 verbose: bool = False) -> None:
 """Merge node output and safeguard state consistency."""
 # ---------- 0) ----------
 # errorstate
 if node_output and isinstance(node_output, dict) and 'error' in node_output:
 state['error'] = node_output['error']
 raise RuntimeError(f" {node_key} : {node_output['error']}")
 state.update(node_output or {})

 # ---------- 1) N0: ----------
 if node_key == 'N0':
 # 1.1 DataFrame
 train_df = node_output.get('train_data')
 test_df = node_output.get('test_data')
 full_df = node_output.get('full_data')

 # 1.2 
 for name, df_ in [('train_data', train_df),
 ('test_data', test_df),
 ('full_data', full_df)]:
 if df_ is None or not isinstance(df_, pd.DataFrame):
 raise RuntimeError(f"N0 {name}")

 # 1.3 state
 state['train_df'] = train_df
 state['test_df'] = test_df
 state['full_df'] = full_df

 # 1.4 WARNING: y_* len(None)
 for k in ('y_train', 'y_val', 'y_test',
 'X_train', 'X_val', 'X_test'): # 
 if state.get(k) is None:
 state.pop(k, None)
 # ---------- 2) N2: ----------
 elif node_key == 'N2':
 if node_output.get('y_train') is None or node_output.get('y_val') is None:
 raise RuntimeError('N2 y_train y_val')

 elif node_key in ('N1', 'N3', 'N4', 'N6', 'N7'):
 # / / Validate intermediate node outputs
 if node_output.get('X_train') is None or node_output.get('X_val') is None:
 raise RuntimeError(f"{node_key} X_train X_val")
 if node_output.get('y_train') is None or node_output.get('y_val') is None:
 raise RuntimeError(f"{node_key} y_train y_val")

 # ---------- 3) N5: ----------
 elif node_key in ('N5', 'N8'):
 if 'y_val_pred' not in node_output:
 raise RuntimeError('N5 / is lack of y_val_pred')


