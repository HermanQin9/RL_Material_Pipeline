"""
Reinforcement Learning Environment for Pipeline Optimization
强化学习流水线优化环境模块
"""

import numpy as np
import random
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline


class PipelineEnv:
    """
    Gym-style environment for automated data processing pipeline construction
    支持自动构建数据处理流水线的Gym风格环境
    """
    
    def __init__(self):
        """初始化强化学习环境 / Initialize RL environment for node_action and select_node"""
        # 定义流水线节点顺序和方法集合 / Define pipeline nodes and their methods for select_node
        # 节点列表：N2 (特征矩阵)、N1 (缺失填充)、N3 (特征选择)、N4 (归一化)、N5 (训练)
        # Node list: ['N2','N1','N3','N4','N5'] (FeatureMatrix, Impute, feature_selection, Scale, Train)
        self.pipeline_nodes = ['N2', 'N1', 'N3', 'N4', 'N5']  # select_node options
        self.num_nodes = len(self.pipeline_nodes)
        
        # 每个节点的可用方法 / Available methods for each select_node operation
        self.methods_for_node = {
            'N1': ['mean', 'median', 'knn', 'none'],        # 缺失填充方法 / Imputation methods
            'N2': ['default'],                             # 构建特征矩阵方法 / Feature matrix methods  
            'N3': ['none', 'variance', 'univariate', 'pca'], # 特征选择方法 / feature_selection methods
            'N4': ['std', 'robust', 'minmax', 'none'],      # 数据归一化方法 / Scaling methods
            'N5': ['rf', 'gbr', 'xgb', 'cat']             # 模型训练方法 / Model training methods
        }
        
        # 只有这些节点使用超参数 / Only these nodes accept a 'param' hyperparameter for node_action
        self.param_nodes = {'N1', 'N3', 'N4', 'N5'}
        self.hyperparam_dim = 1  # 超参数维度 / Dimension of hyperparameter
        
        # 缓存初始化 / Cache initialization
        self._cache = None
        
        # 环境状态初始化 / Initialize state variables
        self.current_step = 0
        self.node_visited = [False] * self.num_nodes  # 每个节点是否已访问 / Whether each node has been visited
        
        # 每种方法调用次数 / Count of method usage
        self.method_calls = {m: 0 for methods in self.methods_for_node.values() for m in methods}
        
        # 指纹： [MAE, R2, 特征数量] / fingerprint = [MAE, R2, number of features]
        self.fingerprint = np.zeros(3, dtype=np.float32)

        # 动作空间描述（仅说明用，不使用gym Spaces）/ Action space description
        self.action_space = {
            'node': list(range(self.num_nodes)),
            'method': self.methods_for_node,
            'params': (0.0, 1.0)
        }

        # Pipeline配置初始化 / Pipeline configuration templates
        self.pipeline_config: Dict[str, Any] = {'sequence': []}
        
        # 数据初始化配置 (N0节点) / Data initialization with node N0
        self.data_init_config = {
            'sequence': ['N0'],
            'N0_method': 'api', 
            'N0_params': {}
        }
        
        # 默认完整流水线配置 / Default full pipeline config (compatible with run_pipeline)
        self.default_pipeline_config = {
            'cache': True,
            'impute_strategy': 'mean',
            'impute_params': None,
            'nan_thresh': 0.5,
            'train_val_ratio': 0.8,
            'selection_strategy': 'none',
            'selection_params': None,
            'scaling_strategy': 'standard',
            'scaling_params': None,
            'model_strategy': 'rf',
            'model_params': {'n_estimators': 10}
        }

    def reset(self) -> Dict[str, Any]:
        """
        重置环境并给出初始observation
        Reset environment and return initial observation
        """
        # 1. 初始化内部状态 / Initialize internal state
        self.current_step = 0
        self.node_visited = [False] * self.num_nodes
        self.method_calls = {m: 0 for methods in self.methods_for_node.values() for m in methods}

        # 2. 只在第一次reset时跑一次pipeline，其余用缓存 / Run pipeline only on first reset, use cache afterwards
        if getattr(self, '_cache', None) is None:
            outputs = run_pipeline(**self.default_pipeline_config)
            self._cache = outputs
        else:
            outputs = self._cache

        metrics = outputs.get('metrics', {}) if outputs else {}

        # 3. 更新fingerprint: [MAE, R2, feature_num]
        self.fingerprint = np.array([
            metrics.get('mae_fe_test', 0.0),
            metrics.get('r2_fe_test', 0.0),
            len(outputs.get('feature_names', []) if outputs else [])
        ], dtype=np.float32)

        # 4. 返回初始观测 / Return initial observation
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        """
        返回当前观测，包括指纹、节点访问标志、动作掩码
        Return current observation: fingerprint, visited flags, action mask
        """
        obs = {
            'fingerprint': self.fingerprint.copy(),
            'node_visited': np.array(self.node_visited, dtype=np.float32),
            'action_mask': self._compute_action_mask()
        }
        return obs

    def _compute_action_mask(self) -> np.ndarray:
        """
        计算动作掩码，屏蔽非法动作
        Compute action mask to block illegal actions
        """
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        
        if self.current_step == 0:
            # 第一步仅允许选择N2 (index 0) / First step: only N2 allowed
            mask[0] = 1.0
        elif self.current_step == self.num_nodes - 1:
            # 最后一步仅允许选择N5 (index最后) / Last step: only N5 allowed
            mask[-1] = 1.0
        else:
            # 中间步骤：禁止N2和N5，其他未访问节点可选 / Middle steps: disallow N2 and N5
            mask[:] = 1.0
            mask[0] = 0.0  # N2 done
            mask[-1] = 0.0 # N5 reserved for last
            for i in range(self.num_nodes):
                if self.node_visited[i]:
                    mask[i] = 0.0
        return mask

    def select_node(self, node_action: Dict[str, Any]) -> bool:
        """
        节点选择函数 / Node selection function
        Execute select_node operation with node_action
        
        Args:
            node_action: Dictionary containing node selection action
                        {'node': int, 'method': int, 'params': list}
        
        Returns:
            bool: Whether the select_node operation is valid
        """
        node_idx = node_action.get('node')
        method_idx = node_action.get('method')
        
        # Validate select_node parameters
        if node_idx is None or node_idx < 0 or node_idx >= self.num_nodes:
            return False
            
        node_name = self.pipeline_nodes[node_idx]
        methods = self.methods_for_node[node_name]
        
        if method_idx is None or method_idx < 0 or method_idx >= len(methods):
            return False
            
        # Check if this select_node operation is valid for current step
        if self.current_step == 0 and node_name != 'N2':
            return False  # First select_node must be N2
        elif self.current_step == self.num_nodes - 1 and node_name != 'N5':
            return False  # Last select_node must be N5
        elif self.current_step > 0 and self.current_step < self.num_nodes - 1:
            if node_name in ['N2', 'N5']:
                return False  # Middle steps cannot select_node N2 or N5
                
        if self.node_visited[node_idx]:
            return False  # Cannot select_node already visited node
            
        return True

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一个动作并返回结果 / Execute node_action and return results
        Process select_node operation through node_action parameter
        """
        node_idx = action.get('node')
        method_idx = action.get('method')
        params = np.array(action.get('params', [0.0]), dtype=float)
        params = np.clip(params, 0.0, 1.0)

        # 1. 使用select_node函数进行动作合法性检测 / Use select_node for node_action validity
        if not self.select_node(action):
            return self._get_obs(), -1.0, True, False, {}
        
        # After select_node validation, we know these are valid integers
        assert node_idx is not None and method_idx is not None
        node_idx = int(node_idx)  # Safe after select_node validation
        method_idx = int(method_idx)  # Safe after select_node validation
            
        node_name = self.pipeline_nodes[node_idx]
        methods = self.methods_for_node[node_name]
        method_name = methods[method_idx]

        # 2. 更新状态 / Update state
        self.node_visited[node_idx] = True
        self.method_calls[method_name] += 1
        
        if self.current_step == 0:
            self.pipeline_config = {
                'sequence': ['N0', node_name],
                'N0_method': 'api', 
                'N0_params': {}
            }
        else:
            if 'sequence' in self.pipeline_config:
                self.pipeline_config['sequence'].append(node_name)
            
        self.pipeline_config[f'{node_name}_method'] = method_name
        if node_name in self.param_nodes:
            params_dict = {'param': float(params[0])}
            self.pipeline_config[f'{node_name}_params'] = params_dict  
            
        print("[ENV] Config so far:", self.pipeline_config)

        # 3. 只有最后一步才运行pipeline / Only run pipeline on last step
        reward = 0.0
        done = False
        metrics = {}
        
        if self.current_step == self.num_nodes - 1:
            try:
                outputs = run_pipeline(**self.pipeline_config, verbose=False)
                metrics = outputs.get('metrics', {}) if outputs else {}
                
                # 计算奖励 / Calculate reward
                mae = metrics.get('mae_fe_test', 0.0) or 0.0
                r2 = metrics.get('r2_fe_test', 0.0) or 0.0
                
                # 复杂度惩罚 / Complexity penalty
                complexity_penalty = self._get_complexity_penalty(method_name)
                reward = r2 - mae - complexity_penalty
                
                # 重复方法惩罚 / Repeated method penalty
                if self.method_calls[method_name] > 1:
                    reward -= 0.5
                    
                done = True
                
                # 更新指纹 / Update fingerprint
                self.fingerprint = np.array([
                    mae, r2, len(outputs.get('feature_names', []) if outputs else [])
                ], dtype=np.float32)
                
            except Exception as e:
                print(f"Pipeline failed: {e}")
                reward = -1.0
                done = True

        self.current_step += 1
        return self._get_obs(), reward, done, False, metrics

    def _get_complexity_penalty(self, method_name: str) -> float:
        """
        获取方法的复杂度惩罚
        Get complexity penalty for a method
        """
        complexity_penalties = {
            'mean': 0.1, 'median': 0.1, 'knn': 0.3, 'none': 0.0,
            'variance': 0.2, 'univariate': 0.2, 'pca': 0.3,
            'std': 0.1, 'robust': 0.2, 'minmax': 0.1,
            'rf': 0.3, 'gbr': 0.3, 'lgbm': 0.2, 'xgb': 0.3, 'cat': 0.3,
            'default': 0.1
        }
        return complexity_penalties.get(method_name, 0.1)


# ========================= 环境工具函数 / Environment Utility Functions =========================

def create_random_action(env: PipelineEnv) -> Dict[str, Any]:
    """
    创建随机动作
    Create a random action for the environment
    """
    obs = env._get_obs()
    action_mask = obs['action_mask']
    
    # 选择可用的节点 / Select available node
    available_nodes = np.where(action_mask > 0)[0]
    if len(available_nodes) == 0:
        return {'node': 0, 'method': 0, 'params': [0.5]}
    
    node_idx = random.choice(available_nodes)
    node_name = env.pipeline_nodes[node_idx]
    
    # 随机选择方法 / Random method selection
    methods = env.methods_for_node[node_name]
    method_idx = random.randint(0, len(methods) - 1)
    
    # 随机参数 / Random parameters
    params = [random.random()]
    
    return {
        'node': node_idx,
        'method': method_idx, 
        'params': params
    }


def evaluate_pipeline_config(config: Dict[str, Any]) -> Dict[str, float]:
    """
    评估流水线配置的性能
    Evaluate performance of a pipeline configuration
    """
    try:
        outputs = run_pipeline(**config, verbose=False)
        metrics = outputs.get('metrics', {}) if outputs else {}
        
        return {
            'mae_fe_test': metrics.get('mae_fe_test', float('inf')),
            'r2_fe_test': metrics.get('r2_fe_test', 0.0),
            'n_features': len(outputs.get('feature_names', []) if outputs else []),
            'success': True
        }
    except Exception as e:
        print(f"Pipeline evaluation failed: {e}")
        return {
            'mae_fe_test': float('inf'),
            'r2_fe_test': 0.0,
            'n_features': 0,
            'success': False
        }
