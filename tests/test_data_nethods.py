# tests/test_data_methods.py
""" fetch_and_featurize and helpers"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import pickle
import pandas as pd
import pytest
import numpy as np
from pathlib import Path

methods_data_loaded = False
import methods.data_methods as dm
from config import PROC_DIR, CACHE_FILE

# 
TMP_CACHE = Path(PROC_DIR) - "tmp_test_cache.pkl"

@ pytest.fixture(autouse=True)
def setup_and_teardown(tmp_path, monkeypatch):
 # 
 monkeypatch.setattr(dm, 'CACHE_PATH', tmp_path - "cache.pkl")
 monkeypatch.setattr(dm, 'PROC_PATH', tmp_path)
 yield
 # 

class X:
 """Picklable dummy composition with stable as_dict per instance."""
 def __init__(self, fe: int):
 self._fe = int(fe)

 def as_dict(self):
 return {"Fe": self._fe}

class DummyComp:
 """Picklable dummy composition with stable as_dict per instance."""
 def __init__(self, fe: int):
 self._fe = int(fe)
 def as_dict(self):
 return {'Fe': self._fe}

def make_dummy_df(n: int = 10) -> pd.DataFrame:
 # DataFramecomposition picklable Fe 
 data = {
 'composition': [X(i % 2) for i in range(n)],
 'structure': [None] * n,
 dm.TARGET_PROP: np.arange(n)
 }
 return pd.DataFrame(data)


def test_split_by_fe():
 df = make_dummy_df(5)
 train, test = dm.split_by_fe(df)
 # train Fe=0
 assert all(d['Fe'] == 0 for m in train['composition'] for d in [m.as_dict()])
 assert all(d['Fe'] == 1 for m in test['composition'] for d in [m.as_dict()])


def test_fetch_and_featurize_cache(tmp_path, monkeypatch):
 # dict 
 dummy = make_dummy_df(8)
 cache_dict = { 'train_data': dummy.iloc[:4], 'test_data': dummy.iloc[4:], 'full_data': dummy }
 with open(dm.CACHE_PATH, 'wb') as f:
 pickle.dump(cache_dict, f)
 out = dm.fetch_and_featurize(cache=True)
 assert 'train_data' in out and 'test_data' in out and 'full_data' in out
 assert out['train_data'].shape[0] == 4


def test_fetch_and_featurize_dataframe_cache(tmp_path, monkeypatch):
 # DataFrame 
 dummy = make_dummy_df(6)
 with open(dm.CACHE_PATH, 'wb') as f:
 pickle.dump(dummy, f)
 out = dm.fetch_and_featurize(cache=True)
 # dummy
 assert out['full_data'].shape[0] == 6
 # train+test = full
 assert out['train_data'].shape[0] + out['test_data'].shape[0] == 6

def test_no_selection():
 from methods.data_methods import no_selection
 import numpy as np
 arr = np.random.rand(10, 5)
 data = {'X_train': arr, 'X_val': arr, 'X_test': arr, 'feature_names': [f'f{i}' for i in range(5)]}
 out = no_selection(data)
 assert (out['X_train'] == arr).all()
 assert out['feature_names'] == data['feature_names']

def test_scale_standard():
 from methods.data_methods import scale_standard
 import numpy as np
 arr = np.random.rand(10, 5)
 data = {'X_train': arr, 'X_val': arr, 'X_test': arr}
 out = scale_standard(data)
 assert 'X_train' in out and 'X_val' in out and 'X_test' in out

def test_train_rf():
 from methods.model_methods import train_rf
 import numpy as np
 data = {'X_train': np.random.rand(20, 5), 'y_train': np.random.rand(20),
 'X_val': np.random.rand(5, 5), 'y_val': np.random.rand(5),
 'X_test': np.random.rand(5, 5)}
 result = train_rf(data)
 assert 'model' in result and 'y_val_pred' in result