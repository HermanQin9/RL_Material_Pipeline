"""Data generation utilities for Materials Project datasets."""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from config import API_KEY, BATCH_SIZE, CACHE_FILE, N_TOTAL, TARGET_PROP
from methods.data_methods import get_value

try:  # pragma: no cover - optional dependency during testing
    from mp_api.client import MPRester  # type: ignore
except Exception:  # pragma: no cover
    MPRester = None  # type: ignore

logger = logging.getLogger(__name__)


def generate_4k_data_safe(check_api_key: bool = True) -> bool:
    """Generate the 4K dataset with extensive safety checks.

    Parameters
    ----------
    check_api_key: bool
        When ``True`` (default) the function validates that ``API_KEY`` is set before
        attempting to connect to the Materials Project API.  Disable for dry-run tests.

    Returns
    -------
    bool
        ``True`` when data generation and cache persistence succeed; ``False`` otherwise.
    """
    logger.info("Starting 4K dataset generation")
    logger.info("Target samples: %s | Batch size: %s | Cache file: %s", N_TOTAL, BATCH_SIZE, CACHE_FILE)

    if check_api_key and not API_KEY:
        logger.error("Materials Project API key is not configured")
        return False

    if MPRester is None:
        logger.error("mp_api is not available; please install the Materials Project API client")
        return False

    dfs: list[pd.DataFrame] = []
    fetched = 0
    error_count = 0
    start_time = time.time()

    try:
        logger.info("Connecting to Materials Project API…")
        with MPRester(API_KEY) as mpr:  # type: ignore[arg-type]
            logger.info("API connection established")

            # Fetch slightly more material chunks to guard against filtering losses
            num_chunks = (N_TOTAL // BATCH_SIZE) + 2
            docs_iter = mpr.materials.summary.search(
                fields=["material_id", "structure", "elements", "formula_pretty", TARGET_PROP],
                chunk_size=BATCH_SIZE,
                num_chunks=num_chunks,
            )

            for batch_index, docs in enumerate(tqdm(docs_iter, desc="获取MP数据", total=num_chunks), start=1):
                if not isinstance(docs, list):
                    docs = [docs]

                valid_docs: list[Any] = []
                for d in docs:
                    try:
                        target_value = get_value(d, TARGET_PROP)
                        structure = get_value(d, "structure")
                        if target_value is not None and structure is not None:
                            valid_docs.append(d)
                    except Exception:  # pragma: no cover - defensive
                        error_count += 1
                        continue

                if not valid_docs:
                    logger.warning("Batch %s has no valid documents", batch_index)
                    continue

                batch_records: list[dict[str, Any]] = []
                for d in valid_docs:
                    try:
                        structure = get_value(d, "structure")
                        composition = None
                        if structure and hasattr(structure, "composition"):
                            composition = structure.composition

                        record = {
                            "material_id": get_value(d, "material_id"),
                            "structure": structure,
                            "elements": get_value(d, "elements"),
                            "formula_pretty": get_value(d, "formula_pretty"),
                            TARGET_PROP: get_value(d, TARGET_PROP),
                            "composition": composition,
                        }
                        batch_records.append(record)
                    except Exception:  # pragma: no cover - defensive
                        error_count += 1
                        continue

                if not batch_records:
                    continue

                df_batch = pd.DataFrame(batch_records)
                df_batch = df_batch.dropna(subset=["structure"]).reset_index(drop=True)
                if not df_batch.empty:
                    dfs.append(df_batch)
                    fetched += len(df_batch)

                if fetched >= N_TOTAL:
                    logger.info("Reached target sample size: %s", fetched)
                    break

    except Exception as exc:  # pragma: no cover - network errors
        logger.exception("Failed to fetch data from Materials Project: %s", exc)
        return False

    if not dfs:
        logger.error("No valid data batches were collected")
        return False

    logger.info("Merging %s data batches", len(dfs))
    full_df = pd.concat(dfs, ignore_index=True)
    if len(full_df) > N_TOTAL:
        full_df = full_df.iloc[:N_TOTAL].reset_index(drop=True)

    elapsed_min = (time.time() - start_time) / 60
    logger.info("Generation completed in %.1f minutes", elapsed_min)
    logger.info("Final dataset size: %s", len(full_df))
    logger.info("Total doc parse errors: %s", error_count)

    cache_path = Path(CACHE_FILE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(full_df, f)
        logger.info("Cache saved to %s (%.1f MB)", cache_path, cache_path.stat().st_size / (1024 * 1024))

        with open(cache_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, pd.DataFrame) and len(loaded) == len(full_df):
            logger.info("Cache verification succeeded")
            return True
        logger.error("Cache verification failed: unexpected payload type or size")
        return False
    except Exception as exc:  # pragma: no cover - file system issues
        logger.exception("Failed to persist cache: %s", exc)
        return False


def fix_4k_data_generation(target_size: int = 4000, batch_size: int = 100) -> bool:
    """Regenerate a safe 4K dataset using stricter validation rules.

    Parameters
    ----------
    target_size: int
        Number of valid samples to collect.
    batch_size: int
        Batch size to request from the API.
    """
    logger.info("Repairing 4K dataset cache | target=%s | batch=%s", target_size, batch_size)

    if not API_KEY:
        logger.error("API key is required to regenerate the dataset")
        return False
    if MPRester is None:
        logger.error("mp_api client is unavailable; install mp-api")
        return False

    cache_file = Path("data/processed/mp_data_cache_4k.pkl")
    collected: list[dict[str, Any]] = []

    try:
        with MPRester(API_KEY) as mpr:  # type: ignore[arg-type]
            docs_iter = mpr.materials.summary.search(
                fields=["material_id", "structure", "elements", "formula_pretty", TARGET_PROP],
                chunk_size=batch_size,
                num_chunks=(target_size // batch_size) + 2,
            )

            for docs in tqdm(docs_iter, desc="获取MP数据"):
                if not docs:
                    continue
                if not isinstance(docs, list):
                    docs = [docs]

                for doc in docs:
                    try:
                        material_id = get_value(doc, "material_id")
                        structure = get_value(doc, "structure")
                        formation_energy = get_value(doc, TARGET_PROP)

                        if not material_id or structure is None or formation_energy is None:
                            continue
                        if not hasattr(structure, "composition") or structure.composition is None:
                            continue

                        composition = structure.composition
                        missing_radius = any(
                            not hasattr(element, "atomic_radius") or element.atomic_radius is None
                            for element in composition.elements
                        )
                        if missing_radius:
                            continue

                        collected.append(
                            {
                                "material_id": material_id,
                                "structure": structure,
                                "elements": get_value(doc, "elements"),
                                "formula_pretty": get_value(doc, "formula_pretty"),
                                TARGET_PROP: formation_energy,
                                "composition": composition,
                            }
                        )

                        if len(collected) >= target_size:
                            break
                    except Exception:  # pragma: no cover - defensive
                        continue
                if len(collected) >= target_size:
                    break

    except Exception as exc:  # pragma: no cover - network errors
        logger.exception("Failed to regenerate dataset: %s", exc)
        return False

    if not collected:
        logger.error("No valid entries collected during regeneration")
        return False

    df = pd.DataFrame(collected[:target_size])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        logger.info("Cache regenerated at %s (%.1f MB)", cache_file, cache_file.stat().st_size / (1024 * 1024))

        with open(cache_file, "rb") as f:
            test_df = pickle.load(f)
        logger.info("Cache load verified | shape=%s", getattr(test_df, "shape", None))
        return True
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to persist regenerated cache: %s", exc)
        return False