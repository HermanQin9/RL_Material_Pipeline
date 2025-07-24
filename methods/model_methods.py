# Node 5: Model Training
import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor  # Delayed import due to compatibility issues
from config import MODEL_DIR, LOG_DIR

def _get_catboost_regressor():
    """Dynamic import of CatBoostRegressor to avoid compatibility issues"""
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor
    except ImportError as e:
        print(f"Warning: CatBoost not available: {e}")
        return None

# Sub-functions for hyperparameter extraction and model fitting
def extract_search_param(params, default_dict):
    """
    用于RL自动化超参映射：用 param ∈ [0,1] 动态设置参数区间。
    """
    default_dict = default_dict.copy()  # Create a copy to avoid modifying the original dictionary
    if 'param' in params:
        p = params['param']
        for key, (low, high) in default_dict.items():
            val = low + p * (high - low)
            if isinstance(low, int):
                default_dict[key] = int(val)
            else:
                default_dict[key] = val
    return default_dict

def model_fit(model, data):
    return model.fit(data['X_train'], data['y_train'])

def model_predict(model, data, key='X_val'):
    return model.predict(data[key]) if data.get(key) is not None else None

# =========== 拆分后合并的标准 pipeline 封装 ===========
def fit_and_predict(model, data, model_name, best_params=None, search_results=None):
    model_fit(model, data)
    y_val_pred  = model_predict(model, data, 'X_val')
    y_test_pred = model_predict(model, data, 'X_test')
    joblib.dump(model, os.path.join(MODEL_DIR, f"formation_energy_{model_name}.joblib"))
    if search_results is not None:
        joblib.dump(search_results, os.path.join(LOG_DIR, f"{model_name}_rs_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"))
    return {
        'model': model,
        'params': best_params,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'X_train': data['X_train'], 'X_val': data['X_val'], 'X_test': data['X_test'],
        'y_train': data['y_train'], 'y_val': data['y_val'], 'y_test': data['y_test'],
    }


# ========== 1. 随机森林 ==========
def train_rf(data, n_estimators=100, max_depth=None, random_search=False, **params):
    param_range = {'n_estimators': (50, 200), 'max_depth': (3, 13)}
    default_param = {'n_estimators': n_estimators, 'max_depth': max_depth}
    default_param = extract_search_param(params, default_param)
    if random_search:
        param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10, 13],
                      'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        rs = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                                param_distributions=param_dist,
                                n_iter=10, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        rs.fit(data['X_train'], data['y_train'])
        model = rs.best_estimator_
        best_params = rs.best_params_
        return fit_and_predict(model, data, 'rf', best_params, rs.cv_results_)
    else:
        model = RandomForestRegressor(**default_param, random_state=42, n_jobs=-1)
        best_params = default_param
        return fit_and_predict(model, data, 'rf', best_params)

# ========== 2. GradientBoosting ==========
def train_gbr(data, n_estimators=100, learning_rate=0.1, max_depth=3, random_search=False, **params):
    param_range = {'n_estimators': (50, 200), 'max_depth': (3, 10), 'learning_rate': (0.05, 0.2)}
    default_param = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    default_param = extract_search_param(params, default_param)
    if random_search:
        param_dist = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [3, 6, 10]}
        rs = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                                param_distributions=param_dist,
                                n_iter=10, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        rs.fit(data['X_train'], data['y_train'])
        model = rs.best_estimator_
        best_params = rs.best_params_
        return fit_and_predict(model, data, 'gbr', best_params, rs.cv_results_)
    else:
        model = GradientBoostingRegressor(**default_param, random_state=42)
        best_params = default_param
        return fit_and_predict(model, data, 'gbr', best_params)

# ========== 3. LightGBM ==========
def train_lgbm(data, n_estimators=100, num_leaves=31, learning_rate=0.1, random_search=False, **params):
    param_range = {'n_estimators': (50, 200), 'num_leaves': (31, 100), 'learning_rate': (0.05, 0.2)}
    default_param = {'n_estimators': n_estimators, 'num_leaves': num_leaves, 'learning_rate': learning_rate}
    default_param = extract_search_param(params, default_param)
    if random_search:
        param_dist = {'n_estimators': [50, 100, 200], 'num_leaves': [31, 50, 100],
                      'learning_rate': [0.05, 0.1, 0.2], 'max_depth': [-1, 5, 10]}
        rs = RandomizedSearchCV(LGBMRegressor(random_state=42), # type: ignore
                                param_distributions=param_dist,
                                n_iter=10, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        rs.fit(data['X_train'], data['y_train'])
        model = rs.best_estimator_
        best_params = rs.best_params_
        return fit_and_predict(model, data, 'lgbm', best_params, rs.cv_results_)
    else:
        model = LGBMRegressor(**default_param, random_state=42)
        best_params = default_param
        return fit_and_predict(model, data, 'lgbm', best_params)

# ========== 4. XGBoost ==========
def train_xgb(data, n_estimators=100, max_depth=6, learning_rate=0.1, random_search=False, **params):
    param_range = {'n_estimators': (50, 200), 'max_depth': (3, 10), 'learning_rate': (0.05, 0.2)}
    default_param = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}
    default_param = extract_search_param(params, default_param)
    if random_search:
        param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10],
                      'learning_rate': [0.05, 0.1, 0.2], 'subsample': [0.8, 1.0]}
        rs = RandomizedSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                                param_distributions=param_dist,
                                n_iter=10, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        rs.fit(data['X_train'], data['y_train'])
        model = rs.best_estimator_
        best_params = rs.best_params_
        return fit_and_predict(model, data, 'xgb', best_params, rs.cv_results_)
    else:
        model = XGBRegressor(**default_param, objective='reg:squarederror', random_state=42)
        best_params = default_param
        return fit_and_predict(model, data, 'xgb', best_params)

# ========== 5. CatBoost ==========
def train_cat(data, iterations=100, depth=6, learning_rate=0.1, random_search=False, **params):
    param_range = {'iterations': (50, 200), 'depth': (4, 10), 'learning_rate': (0.05, 0.2)}
    default_param = {'iterations': iterations, 'depth': depth, 'learning_rate': learning_rate}
    default_param = extract_search_param(params, default_param)
    
    CatBoostRegressor = _get_catboost_regressor()
    if CatBoostRegressor is None:
        raise ImportError("CatBoost is not available")
    
    if random_search:
        param_dist = {'iterations': [50, 100, 200], 'depth': [4, 6, 10],
                      'learning_rate': [0.05, 0.1, 0.2]}
        rs = RandomizedSearchCV(CatBoostRegressor(silent=True, random_state=42), # type: ignore
                                param_distributions=param_dist,
                                n_iter=10, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        rs.fit(data['X_train'], data['y_train'])
        model = rs.best_estimator_
        best_params = rs.best_params_
        return fit_and_predict(model, data, 'cat', best_params, rs.cv_results_)
    else:
        model = CatBoostRegressor(**default_param, silent=True, random_state=42)
        best_params = default_param
        return fit_and_predict(model, data, 'cat', best_params)


# ========================= 指标计算和结果输出函数 / Metrics and Results Functions =========================

def compute_metrics_and_sizes(
    state: dict,
    start_time: float,
    sequence: list,
    execution_times: dict
) -> tuple:
    """
    计算性能指标和数据集大小统计。
    Compute performance metrics and dataset size statistics.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    import time
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # ---------- 1) 校验必须的预测结果 ----------
    required_pred_keys = ['y_val', 'y_val_pred']
    missing_keys = [key for key in required_pred_keys if key not in state]
    if missing_keys:
        raise KeyError(f"缺少预测结果: {missing_keys} / Missing prediction results: {missing_keys}")

    # ---------- 2) Non-Fe 验证集指标 ----------
    y_non_fe, y_pred_non_fe = state['y_val'], state['y_val_pred']
    mae_non_fe = mean_absolute_error(y_non_fe, y_pred_non_fe)
    r2_non_fe  = r2_score(y_non_fe, y_pred_non_fe)

    # ---------- 3) Fe 测试集指标（可选） ----------
    y_fe, y_pred_fe = state.get('y_test'), state.get('y_test_pred')
    if isinstance(y_fe, (list, np.ndarray)) and isinstance(y_pred_fe, (list, np.ndarray)) \
            and len(y_fe) > 0 and len(y_pred_fe) > 0:
        mae_fe = mean_absolute_error(y_fe, y_pred_fe)
        r2_fe  = r2_score(y_fe, y_pred_fe)
    else:
        mae_fe = r2_fe = None

    # ---------- 4) 数据集大小统计 ----------
    def safe_len(obj) -> int:
        """对 None、标量或其他无 __len__ 的对象返回 0。"""
        try:
            return len(obj) if obj is not None else 0
        except Exception:
            return 0

    sizes = {
        'total_non_fe': safe_len(state.get('train_df')),
        'non_fe_train': safe_len(state.get('y_train')),
        'non_fe_test' : safe_len(state.get('y_val')),
        'total_fe'    : safe_len(state.get('test_df')),
        'n_features'  : (
            state['X_train'].shape[1]
            if isinstance(state.get('X_train'), (pd.DataFrame, np.ndarray))
            else 0
        )
    }

    # ---------- 5) 组织指标字典 ----------
    metrics = {
        'mae_non_fe_test': mae_non_fe,
        'r2_non_fe_test' : r2_non_fe,
        'mae_fe_test'    : mae_fe,
        'r2_fe_test'     : r2_fe,
        'run_time_sec'   : time.time() - start_time,
        'sequence'       : sequence,
        'timestamp'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'execution_times': execution_times,
        'total_execution_time': sum(execution_times.values())
    }

    return metrics, sizes


def print_results(metrics: dict, sizes: dict) -> None:
    """
    打印流水线执行结果。
    Print pipeline execution results.
    """
    print("📈 数据集统计 / Dataset Statistics:")
    print("-" * 50)
    print(f"  Non-Fe数据集 / Non-Fe Dataset:")
    print(f"    总计 / Total: {sizes['total_non_fe']:,}")
    print(f"    训练集 / Train: {sizes['non_fe_train']:,}")
    print(f"    验证集 / Validation: {sizes['non_fe_test']:,}")
    print(f"  Fe数据集 / Fe Dataset: {sizes['total_fe']:,}")
    print(f"  特征数量 / Features: {sizes['n_features']:,}")
    print()
    print("🎯 性能指标 / Performance Metrics:")
    print("-" * 50)
    print(f"  Non-Fe测试集 / Non-Fe Test Set:")
    print(f"    MAE: {metrics['mae_non_fe_test']:.4f}")
    print(f"    R²:  {metrics['r2_non_fe_test']:.4f}")
    if metrics['mae_fe_test'] is not None:
        print(f"  Fe测试集 / Fe Test Set:")
        print(f"    MAE: {metrics['mae_fe_test']:.4f}")
        print(f"    R²:  {metrics['r2_fe_test']:.4f}")
    else:
        print("  Fe测试集 / Fe Test Set: 无数据 / No data")
    print()
    print("⏱️  执行时间 / Execution Time:")
    print("-" * 50)
    total_time = metrics['run_time_sec']
    for node, exec_time in metrics['execution_times'].items():
        percentage = (exec_time / total_time) * 100
        print(f"  {node}: {exec_time:.2f}s ({percentage:.1f}%)")
    print(f"  总计 / Total: {total_time:.2f}s")
    print()


def save_pipeline_outputs(state: dict, metrics: dict, verbose: bool = False) -> str:
    """
    保存流水线输出结果。
    Save pipeline output results.
    """
    import pandas as pd
    import json
    from datetime import datetime
    
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(MODEL_DIR, f"run_{dt_str}")
    
    # 准备测试结果 DataFrame / Prepare test results DataFrame
    test_results = None
    if state.get('y_test') is not None and state.get('y_test_pred') is not None:
        test_results = pd.DataFrame({
            'y_true': state['y_test'],
            'y_pred': state['y_test_pred']
        })
    
    # 创建输出目录
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 保存模型和其他组件
    if state.get('model') is not None:
        joblib.dump(state['model'], os.path.join(outputs_dir, 'model.joblib'))
    
    if state.get('scaler') is not None:
        joblib.dump(state['scaler'], os.path.join(outputs_dir, 'scaler.joblib'))
    
    if state.get('selector') is not None:
        joblib.dump(state['selector'], os.path.join(outputs_dir, 'selector.joblib'))
    
    # 保存特征名称
    if state.get('feature_names'):
        pd.Series(state['feature_names']).to_csv(os.path.join(outputs_dir, "feature_names.csv"), index=False)
    
    # 保存参数
    if state.get('params') is not None:
        if isinstance(state['params'], dict):
            pd.Series(state['params']).to_csv(os.path.join(outputs_dir, "best_params.csv"), index=True)
        else:
            with open(os.path.join(outputs_dir, "best_params.txt"), 'w') as f:
                f.write(str(state['params']))
    
    # 保存指标
    with open(os.path.join(outputs_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存测试结果用于Dashboard
    if test_results is not None:
        os.makedirs("dash_app/data", exist_ok=True)
        test_results.to_csv("dash_app/data/test_predictions.csv", index=False)
    
    if verbose:
        print(f"Pipeline outputs saved to: {outputs_dir}")
    
    return outputs_dir
