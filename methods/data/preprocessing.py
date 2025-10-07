"""
Preprocessing utilities for the 10-node pipeline

Contains:
- N3 Cleaning: IQR winsorization and Z-score clipping
- N4 GNN placeholder (no-op)
- N5 Knowledge Graph placeholder (no-op)
- N9 Termination placeholder

All functions accept and return a unified state dict with keys:
  {X_train, X_val, X_test, y_train, y_val, y_test, feature_names}
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, cast
import numpy as np


def _ensure_numpy(x):
	if x is None:
		return None
	return x if isinstance(x, np.ndarray) else np.asarray(x)


# ================= N3: Cleaning (Outlier/Noise) =================
def clean_data(data: Dict[str, Any], strategy: str = 'none', params: Optional[dict] = None) -> Dict[str, Any]:
	params = params or {}
	if strategy == 'outlier':
		return clean_outlier_iqr(data, **params)
	if strategy == 'noise':
		return clean_noise_zscore(data, **params)
	return clean_none(data)


def clean_outlier_iqr(data: Dict[str, Any], factor: Optional[float] = None, param: Optional[float] = None, **_kwargs) -> Dict[str, Any]:
	"""
	IQR-based winsorization (clip) using train statistics.
	factor: whisker factor; if None and param given in [0,1], map to [1.0, 3.0]. Default 1.5
	"""
	X_train = _ensure_numpy(data['X_train'])
	X_val = _ensure_numpy(data['X_val'])
	X_test = _ensure_numpy(data.get('X_test'))

	# Type safety for static checkers
	if X_train is None:
		raise RuntimeError("X_train is None in clean_outlier_iqr")
	X_train = cast(np.ndarray, X_train)

	# Map param->[1.0, 3.0]
	if factor is None:
		if param is not None:
			p = max(0.0, min(1.0, float(param)))
			factor = 1.0 + 2.0 * p
		else:
			factor = 1.5

	q1 = np.nanpercentile(X_train, 25, axis=0)
	q3 = np.nanpercentile(X_train, 75, axis=0)
	iqr = q3 - q1
	lower = q1 - factor * iqr
	upper = q3 + factor * iqr

	def clip_arr(arr):
		if arr is None:
			return None
		return np.clip(arr, lower, upper)

	X_train_c = clip_arr(X_train)
	X_val_c = clip_arr(X_val)
	X_test_c = clip_arr(X_test)

	return {
		'X_train': X_train_c,
		'X_val': X_val_c,
		'X_test': X_test_c,
		'y_train': data.get('y_train'),
		'y_val': data.get('y_val'),
		'y_test': data.get('y_test'),
		'feature_names': data.get('feature_names'),
		'cleaning_info': {'method': 'iqr', 'factor': factor}
	}


def clean_noise_zscore(data: Dict[str, Any], z: Optional[float] = None, param: Optional[float] = None, **_kwargs) -> Dict[str, Any]:
	"""
	Z-score clipping with threshold z. If z is None and param in [0,1], map to z in [1, 4]. Default z=3.
	Uses train mean/std and applies clipping to all splits.
	"""
	X_train = _ensure_numpy(data['X_train'])
	X_val = _ensure_numpy(data['X_val'])
	X_test = _ensure_numpy(data.get('X_test'))

	if X_train is None:
		raise RuntimeError("X_train is None in clean_noise_zscore")
	X_train = cast(np.ndarray, X_train)

	if z is None:
		if param is not None:
			p = max(0.0, min(1.0, float(param)))
			z = 1.0 + 3.0 * (1.0 - p)  # p=1 => z≈1, p=0 => z≈4
		else:
			z = 3.0

	mu = np.nanmean(X_train, axis=0)
	sigma = np.nanstd(X_train, axis=0)
	sigma = np.where(sigma == 0, 1.0, sigma)

	lower = mu - z * sigma
	upper = mu + z * sigma

	def clip_arr(arr):
		if arr is None:
			return None
		return np.clip(arr, lower, upper)

	X_train_c = clip_arr(X_train)
	X_val_c = clip_arr(X_val)
	X_test_c = clip_arr(X_test)

	return {
		'X_train': X_train_c,
		'X_val': X_val_c,
		'X_test': X_test_c,
		'y_train': data.get('y_train'),
		'y_val': data.get('y_val'),
		'y_test': data.get('y_test'),
		'feature_names': data.get('feature_names'),
		'cleaning_info': {'method': 'zscore', 'z': z}
	}


def clean_none(data: Dict[str, Any], **_kwargs) -> Dict[str, Any]:
	return {
		'X_train': data['X_train'],
		'X_val': data['X_val'],
		'X_test': data.get('X_test'),
		'y_train': data.get('y_train'),
		'y_val': data.get('y_val'),
		'y_test': data.get('y_test'),
		'feature_names': data.get('feature_names')
	}


# ================= Placeholders: N4 GNN / N5 KG / N9 End =================
def gnn_process(data: Dict[str, Any], strategy: str = 'none', params: Optional[dict] = None) -> Dict[str, Any]:
	# Lightweight placeholder: append simple row-wise stats as surrogate graph aggregates
	X_tr = _ensure_numpy(data['X_train'])
	X_va = _ensure_numpy(data['X_val'])
	X_te = _ensure_numpy(data.get('X_test'))

	if X_tr is None:
		return clean_none(data)

	def append_stats(X: Optional[np.ndarray]) -> Optional[np.ndarray]:
		if X is None:
			return None
		# Compute per-row mean, std, min, max (safe for NaN)
		mean = np.nanmean(X, axis=1, keepdims=True)
		std = np.nanstd(X, axis=1, keepdims=True)
		minv = np.nanmin(X, axis=1, keepdims=True)
		maxv = np.nanmax(X, axis=1, keepdims=True)
		return np.concatenate([X, mean, std, minv, maxv], axis=1)

	X_tr2 = append_stats(X_tr)
	X_va2 = append_stats(X_va)
	X_te2 = append_stats(X_te)

	feat_names = list(data.get('feature_names') or [])
	feat_names += ['gnn_row_mean', 'gnn_row_std', 'gnn_row_min', 'gnn_row_max']

	return {
		'X_train': X_tr2,
		'X_val': X_va2,
		'X_test': X_te2,
		'y_train': data.get('y_train'),
		'y_val': data.get('y_val'),
		'y_test': data.get('y_test'),
		'feature_names': feat_names
	}


def kg_process(data: Dict[str, Any], strategy: str = 'none', params: Optional[dict] = None) -> Dict[str, Any]:
	# Placeholder: currently no-op for any strategy
	# 保证输出所有字段，补全y_val_pred/y_test_pred等
	result = clean_none(data)
	result.setdefault('y_val_pred', None)
	result.setdefault('y_test_pred', None)
	return result


def terminate(data: Dict[str, Any], **_kwargs) -> Dict[str, Any]:
	# End node: no changes; could add flags if needed
	return data

