"""
Reinforcement Learning Environment for Pipeline Optimization
强化学习流水线优化环境模块
"""

import numpy as np
import random
from typing import Dict, Any, Tuple, List, Optional

from ..pipelines.execution import run_pipeline, run_pipeline_config


class PipelineEnv:
    """
    Gym-style environment for automated data processing pipeline construction
    支持自动构建数据处理流水线的Gym风格环境
    """
    
    def __init__(self):
        """初始化强化学习环境 / Initialize RL environment for node_action and select_node"""
        # 控制台调试输出开关 / Debug print flag
        self.debug = False

        # 定义流水线节点顺序和方法集合 / Define pipeline nodes and their methods for select_node
        # 10-node (Option 2): fixed N0(start), N2(second), N8(pre-end), N9(end)
        # PPO controls N1,N3,N4,N5,N6,N7 order in the middle
        self.pipeline_nodes = ['N0', 'N2', 'N1', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
        self.num_nodes = len(self.pipeline_nodes)

        # 每个节点的可用方法 / Available methods for each select_node operation
        self.methods_for_node = {
            'N0': ['api'],
            'N1': ['mean', 'median', 'knn'],
            'N2': ['default'],
            'N3': ['outlier', 'noise', 'none'],
            'N4': ['gcn', 'gat', 'sage'],
            'N5': ['entity', 'relation', 'none'],
            'N6': ['variance', 'univariate', 'pca'],
            'N7': ['std', 'robust', 'minmax'],
            'N8': ['rf', 'gbr', 'xgb', 'cat'],
            'N9': ['terminate']
        }

        # 最大方法数（用于方法掩码的统一长度）/ Max methods across nodes for uniform masking
        self.max_methods = max(len(m) for m in self.methods_for_node.values())

        # 只有这些节点使用超参数 / Only these nodes accept a 'param' hyperparameter for node_action
        self.param_nodes = {'N1', 'N3', 'N6', 'N7', 'N8'}
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
        self.pipeline_config = {}

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
        # 方法数信息（用于方法级别的先验/提示）/ method-count hint for policy
        method_count = np.array([len(self.methods_for_node[n]) for n in self.pipeline_nodes], dtype=np.float32)
        method_count = method_count / (method_count.max() if method_count.max() > 0 else 1.0)

        # 方法级别掩码：形状 [num_nodes, max_methods]，每个节点有效方法置1
        # Method-level mask: shape [num_nodes, max_methods], mark valid methods with 1
        method_mask = np.zeros((self.num_nodes, self.max_methods), dtype=np.float32)
        for i, n in enumerate(self.pipeline_nodes):
            k = len(self.methods_for_node[n])
            if k > 0:
                method_mask[i, :k] = 1.0

        obs = {
            'fingerprint': self.fingerprint.copy(),
            'node_visited': np.array(self.node_visited, dtype=np.float32),
            'action_mask': self._compute_action_mask(),
            'method_count': method_count,
            'method_mask': method_mask,
        }
        return obs

    def _compute_action_mask(self) -> np.ndarray:
        """
        计算动作掩码，屏蔽非法动作
        Compute action mask to block illegal actions
        """
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        
        idx = {n:i for i,n in enumerate(self.pipeline_nodes)}
        if self.current_step == 0:
            mask[idx['N0']] = 1.0
        elif self.current_step == 1:
            mask[idx['N2']] = 1.0
        else:
            v8 = self.node_visited[idx['N8']]
            v9 = self.node_visited[idx['N9']]
            if v8 and not v9:
                # After training, only allow termination
                mask[idx['N9']] = 1.0
            else:
                # Middle phase: allow any unvisited middle nodes
                for n in ['N1','N3','N4','N5','N6','N7']:
                    i = idx[n]
                    if not self.node_visited[i]:
                        mask[i] = 1.0
                # Also allow jumping to training (N8) at any time after N2
                if not v8:
                    mask[idx['N8']] = 1.0
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
            
        # Enforce by mask
        if self._compute_action_mask()[node_idx] <= 0.0:
            return False

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
                'sequence': ['N0'],
                'N0_method': 'api', 
                'N0_params': {}
            }
        else:
            # Append selected node name after the first step
            if 'sequence' in self.pipeline_config:
                self.pipeline_config['sequence'].append(node_name)
            
        self.pipeline_config[f'{node_name}_method'] = method_name
        if node_name in self.param_nodes:
            params_dict = {'param': float(params[0])}
            self.pipeline_config[f'{node_name}_params'] = params_dict  
            
        if self.debug:
            print("[ENV] Config so far:", self.pipeline_config)

        # 3. 触发终止时运行pipeline / Run pipeline when N9 is chosen
        reward = 0.0
        done = False
        metrics = {}

        if node_name == 'N9':
            try:
                from ..pipelines.execution import run_pipeline_config
                outputs = run_pipeline_config(**self.pipeline_config)  # type: ignore
                metrics = outputs.get('metrics', {}) if outputs else {}

                # 计算奖励 / Calculate reward
                mae = metrics.get('mae_fe_test', 0.0) or 0.0
                r2 = metrics.get('r2_fe_test', 0.0) or 0.0

                # 复杂度惩罚 / Complexity penalty (use last method)
                complexity_penalty = self._get_complexity_penalty(method_name)
                reward = r2 - mae - complexity_penalty

                # 重复方法惩罚 / Repeated method penalty
                if self.method_calls[method_name] > 1:
                    reward -= 0.5

                done = True

                # 更新指纹 / Update fingerprint (sizes from outputs)
                n_feats = 0
                try:
                    n_feats = int(outputs.get('sizes', {}).get('n_features', 0)) if outputs else 0
                except Exception:
                    n_feats = 0
                self.fingerprint = np.array([
                    mae, r2, n_feats
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
        outputs = run_pipeline_config(**config) # type: ignore
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
