""" / Global project configuration

:
>>> from config import API_KEY, CACHE_FILE, PROC_DIR, BATCH_SIZE, N_TOTAL, TARGET_PROP
"""

from __future__ import annotations
import os
from pathlib import Path

# ------------------ Root & Data Paths ------------------
ROOT_DIR = Path(__file__).resolve().parent # / project root

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
DASH_DIR = ROOT_DIR / "dash_app" / "data"

# / auto‑create folders
for p in (RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR, DASH_DIR):
 p.mkdir(parents=True, exist_ok=True)

# ------------------ API API & Constants ------------------
# API Key MP_API_KEY 
API_KEY: str = os.getenv("MP_API_KEY", "f3qtz1d2EV47QfPtknWcFSOXTaUCzNli")

# / target property to predict
TARGET_PROP: str = "formation_energy_per_atom"

# ------------------ Pipeline Runtime Settings ------------------
# PIPELINE_TEST=1 
TEST_MODE: bool = os.getenv("PIPELINE_TEST", "1") == "1"

# 老师要求：小规模数据集400个点
BATCH_SIZE: int = 20 if TEST_MODE else 100
N_TOTAL: int = 400  # Fixed to 400 points as per advisor requirement

# / cache file path
CACHE_FILE: Path = PROC_DIR / (
 "mp_data_cache_400.pkl"  # Single cache file for 400 samples
)

# / export
__all__ = [
 "API_KEY",
 "CACHE_FILE",
 "PROC_DIR",
 "MODEL_DIR",
 "LOG_DIR",
 "BATCH_SIZE",
 "N_TOTAL",
 "TARGET_PROP",
]
