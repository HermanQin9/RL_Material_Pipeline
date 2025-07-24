"""项目全局配置 / Global project configuration

使用方法:
>>> from config import API_KEY, CACHE_FILE, PROC_DIR, BATCH_SIZE, N_TOTAL, TARGET_PROP
"""

from __future__ import annotations
import os
from pathlib import Path

# ------------------ 根目录与数据路径 Root & Data Paths ------------------
ROOT_DIR = Path(__file__).resolve().parent  # 项目根目录 / project root

DATA_DIR  = ROOT_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR   = ROOT_DIR / "logs"
DASH_DIR  = ROOT_DIR / "dash_app" / "data"

# 自动创建目录 / auto‑create folders
for p in (RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR, DASH_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ------------------ API 与常量 API & Constants ------------------
# API Key 可通过环境变量 MP_API_KEY 注入
API_KEY: str = os.getenv("MP_API_KEY", "f3qtz1d2EV47QfPtknWcFSOXTaUCzNli")

# 预测目标属性 / target property to predict
TARGET_PROP: str = "formation_energy_per_atom"

# ------------------ Pipeline 运行配置 Runtime Settings ------------------
# 测试模式开关：环境变量 PIPELINE_TEST=1 时走小数据集
TEST_MODE: bool = os.getenv("PIPELINE_TEST", "1") == "1"

BATCH_SIZE: int = 20 if TEST_MODE else 100
N_TOTAL: int   = 200 if TEST_MODE else 4000

# 缓存文件路径 / cache file path
CACHE_FILE: Path = PROC_DIR / (
    "mp_data_cache_200_test.pkl" if TEST_MODE else "mp_data_cache_4k.pkl"
)

# 公开接口 / export
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
