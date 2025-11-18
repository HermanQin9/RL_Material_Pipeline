"""Validation helpers for generated materials datasets."""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from methods.data_methods import fetch_data

logger = logging.getLogger(__name__)


def test_4k_data_loading(cache: bool = True, verbose: bool = True) -> tuple[bool, Optional[pd.DataFrame]]:
	"""Validate that the cached 4K dataset can be loaded via :func:`fetch_data`.

	Parameters
	----------
	cache: bool
		When ``True`` leverage the cache mechanism.  Set ``False`` to force regeneration.
	verbose: bool
		Emit detailed information to the log when ``True``.

	Returns
	-------
	tuple
		``(success, dataframe)`` where ``dataframe`` contains the loaded data when successful.
	"""
	logger.info("Testing 4K dataset loading | cache=%s", cache)

	try:
		start_time = time.time()
		df = fetch_data(cache=cache)
		load_time = time.time() - start_time

		if verbose:
			logger.info("4K dataset loaded successfully | shape=%s | load_time=%.1fs | memory=%.1f MB",
						df.shape, load_time, df.memory_usage(deep=True).sum() / (1024 * 1024))
			preview_cols = [col for col in ["material_id", "formula_pretty"] if col in df.columns]
			if preview_cols:
				logger.debug("Preview columns (%s):\n%s", preview_cols, df[preview_cols].head())

		return True, df

	except Exception as exc:  # pragma: no cover - diagnostic helper
		logger.exception("4K dataset loading failed: %s", exc)
		return False, None
