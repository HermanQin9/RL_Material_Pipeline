import numpy as np
from methods.data.preprocessing import gnn_process, kg_process


def _fake_data(n=10, d=4):
    X = np.random.randn(n, d)
    return {
        'X_train': X,
        'X_val': X.copy(),
        'X_test': X.copy(),
        'y_train': np.random.randn(n),
        'y_val': np.random.randn(n),
        'y_test': np.random.randn(n),
        'feature_names': [f"f{i}" for i in range(d)]
    }


def test_gnn_process_appends_stats():
    data = _fake_data()
    out = gnn_process(data)
    assert out['X_train'].shape[1] == data['X_train'].shape[1] + 4
    assert len(out['feature_names']) == len(data['feature_names']) + 4


def test_kg_process_noop():
    data = _fake_data()
    out = kg_process(data)
    assert out['X_train'].shape == data['X_train'].shape