# PPO方法选择机制详解 / PPO Method Selection Mechanism Explained

## 目录 / Table of Contents
1. [概述 / Overview](#概述--overview)
2. [核心架构 / Core Architecture](#核心架构--core-architecture)
3. [方法选择流程 / Method Selection Flow](#方法选择流程--method-selection-flow)
4. [神经网络结构 / Neural Network Structure](#神经网络结构--neural-network-structure)
5. [动作空间设计 / Action Space Design](#动作空间设计--action-space-design)
6. [具体实现细节 / Implementation Details](#具体实现细节--implementation-details)
7. [完整示例 / Complete Example](#完整示例--complete-example)

---

## 概述 / Overview

本系统使用PPO（Proximal Policy Optimization）强化学习算法来**自动选择最优的流水线节点和方法组合**。PPO智能体通过学习决定：
1. **选择哪个节点** (Node Selection)
2. **使用哪个方法** (Method Selection)  
3. **设置什么参数** (Parameter Tuning)

This system uses PPO (Proximal Policy Optimization) reinforcement learning to **automatically select the optimal pipeline node and method combinations**. The PPO agent learns to decide:
1. **Which node to select**
2. **Which method to use**
3. **What parameters to set**

---

## 核心架构 / Core Architecture

### 5节点流水线结构 / 5-Node Pipeline Structure

当前系统使用5个节点的流水线：
Current system uses a 5-node pipeline:

```
N0 (DataFetch) → N2 (FeatureMatrix) → N1 (Imputation) → N3 (FeatureSelection) → N4 (Scaling) → N5 (ModelTraining)
                 ↑                    ↑                  ↑                       ↑                ↑
                 PPO决策起点           PPO选择节点         PPO选择节点              PPO选择节点       PPO选择节点
```

### 节点与方法映射 / Node-Method Mapping

每个节点都有多个可选方法（定义在 `env/pipeline_env.py:38-44`）：

| 节点 Node | 用途 Purpose | 可选方法 Available Methods |
|-----------|--------------|---------------------------|
| **N1** | 缺失值填充 Imputation | `mean`, `median`, `knn`, `none` |
| **N2** | 特征矩阵 Feature Matrix | `default` |
| **N3** | 特征选择 Feature Selection | `none`, `variance`, `univariate`, `pca` |
| **N4** | 数据归一化 Scaling | `std`, `robust`, `minmax`, `none` |
| **N5** | 模型训练 Model Training | `rf`, `gbr`, `xgb`, `cat` |

---

## 方法选择流程 / Method Selection Flow

### 完整决策链 / Complete Decision Chain

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 环境观察 / Environment Observation                            │
│    - 当前流水线状态 (fingerprint: [MAE, R², n_features])         │
│    - 节点访问标志 (node_visited: [False, True, False, ...])     │
│    - 动作掩码 (action_mask: [1.0, 0.0, 1.0, ...])              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. PPO神经网络处理 / PPO Neural Network Processing              │
│    输入 Input: 观察值 obs (flatten后的tensor)                    │
│    输出 Output: 4个头部的logits/values                          │
│    - node_logits: [6个节点的概率分布]                           │
│    - method_logits: [10个方法的概率分布]                        │
│    - params: [归一化的参数值 0.0-1.0]                           │
│    - value: [状态价值估计]                                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 动作采样 / Action Sampling                                   │
│    A. 节点选择 / Node Selection                                 │
│       - 从node_logits创建分类分布 Categorical(logits)            │
│       - 采样节点索引: node_action ~ Categorical                  │
│       - 计算log概率: node_log_prob                               │
│                                                                  │
│    B. 方法选择 / Method Selection                                │
│       - 根据选中节点获取可用方法数量                              │
│       - 屏蔽多余的method_logits                                  │
│       - 从有效范围创建分类分布                                    │
│       - 采样方法索引: method_action ~ Categorical                │
│       - 计算log概率: method_log_prob                             │
│                                                                  │
│    C. 参数采样 / Parameter Sampling                              │
│       - 从正态分布采样: param ~ Normal(params, 0.1)              │
│       - 限制范围: param = clamp(param, 0.0, 1.0)                 │
│       - 计算log概率: param_log_prob                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 环境执行 / Environment Execution                             │
│    - 验证动作合法性 (select_node函数)                            │
│    - 更新流水线配置                                              │
│    - 执行节点方法                                                │
│    - 计算奖励                                                    │
│    - 返回新状态                                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. 学习更新 / Learning Update                                   │
│    - 收集经验 (obs, action, reward, log_prob)                   │
│    - 计算优势函数 Advantage                                      │
│    - PPO裁剪损失函数 Clipped Loss                               │
│    - 梯度反向传播更新网络                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 神经网络结构 / Neural Network Structure

### PPOPolicy网络架构 (`ppo/policy.py`)

```python
PPOPolicy(
    obs_dim=观察空间维度,  # fingerprint(3) + node_visited(5) + action_mask(5) = 13
    action_dim=动作空间维度,  # 6节点 + 10方法 + 1参数 = 17
    hidden_dim=256
)
```

#### 网络层级 / Network Layers

```
输入层 Input Layer
    ↓
┌───────────────────────────────────────┐
│ Shared Feature Extraction            │
│ - Linear(obs_dim → 256) + ReLU       │
│ - Linear(256 → 256) + ReLU           │
│ - Linear(256 → 128) + ReLU           │
└───────────────────────────────────────┘
    ↓
    分成4个头部 / Split into 4 heads
    ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Node Head    │ Method Head  │ Param Head   │ Value Head   │
│              │              │              │              │
│ Linear(128   │ Linear(128   │ Linear(128   │ Linear(128   │
│   → 64)      │   → 64)      │   → 64)      │   → 64)      │
│ + ReLU       │ + ReLU       │ + ReLU       │ + ReLU       │
│              │              │              │              │
│ Linear(64    │ Linear(64    │ Linear(64    │ Linear(64    │
│   → 6)       │   → 10)      │   → 1)       │   → 1)       │
│              │              │ + Sigmoid    │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
      ↓              ↓              ↓              ↓
  node_logits   method_logits    params        value
   [6 dims]      [10 dims]       [0-1]       [scalar]
```

---

## 动作空间设计 / Action Space Design

### 分层动作结构 / Hierarchical Action Structure

动作被设计为**三层结构** (定义在 `ppo/trainer.py:92-133`):

```python
action = {
    'node': int,      # 节点索引 0-4 (对应N2,N1,N3,N4,N5)
    'method': int,    # 方法索引 0-n (n取决于节点)
    'params': list    # 参数列表 [0.0-1.0]
}
```

### 动作约束与掩码 / Action Constraints and Masking

#### 流水线顺序约束 / Pipeline Order Constraints

```python
# 在 env/pipeline_env.py:136-157
def _compute_action_mask(self):
    if self.current_step == 0:
        # 第一步必须选择N2 / First step must be N2
        mask[0] = 1.0  # Only N2 allowed
    elif self.current_step == self.num_nodes - 1:
        # 最后一步必须选择N5 / Last step must be N5
        mask[-1] = 1.0  # Only N5 allowed
    else:
        # 中间步骤：禁止N2和N5 / Middle steps: disallow N2 and N5
        mask[:] = 1.0
        mask[0] = 0.0   # N2 already done
        mask[-1] = 0.0  # N5 reserved for last
        # 禁止已访问节点 / Disallow visited nodes
        for i in range(self.num_nodes):
            if self.node_visited[i]:
                mask[i] = 0.0
```

---

## 具体实现细节 / Implementation Details

### 1. 方法选择的核心代码 / Core Method Selection Code

位置：`ppo/trainer.py:92-133`

```python
def select_action(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """选择动作 / Select action"""
    with torch.no_grad():
        # Step 1: 前向传播得到所有logits
        # Forward pass to get all logits
        node_logits, method_logits, params, value = self.policy(obs)
        
        # Step 2: 节点选择 - 使用分类分布
        # Node selection using Categorical distribution
        node_dist = torch.distributions.Categorical(logits=node_logits)
        node_action = node_dist.sample()  # 采样节点 / Sample node
        node_log_prob = node_dist.log_prob(node_action)
        
        # Step 3: 方法选择 - 根据选中的节点屏蔽
        # Method selection - mask based on selected node
        node_idx = int(node_action.item())
        node_name = self.env.pipeline_nodes[node_idx]
        num_methods = len(self.env.methods_for_node[node_name])
        
        # 关键：只使用该节点可用的方法数量
        # Key: only use available methods for this node
        method_logits_masked = method_logits[:num_methods]
        method_dist = torch.distributions.Categorical(logits=method_logits_masked)
        method_action = method_dist.sample()  # 采样方法 / Sample method
        method_log_prob = method_dist.log_prob(method_action)
        
        # Step 4: 参数采样 - 使用正态分布
        # Parameter sampling using Normal distribution
        param_dist = torch.distributions.Normal(params, 0.1)
        param_action = torch.clamp(param_dist.sample(), 0.0, 1.0)
        param_log_prob = param_dist.log_prob(param_action).sum()
        
        # Step 5: 组装动作字典
        # Assemble action dictionary
        action = {
            'node': int(node_action.item()),
            'method': int(method_action.item()),
            'params': param_action.numpy().tolist()
        }
        
        return action, log_probs
```

### 2. 动作验证 / Action Validation

位置：`env/pipeline_env.py:159-196`

```python
def select_node(self, node_action: Dict[str, Any]) -> bool:
    """验证节点选择是否合法 / Validate node selection"""
    node_idx = node_action.get('node')
    method_idx = node_action.get('method')
    
    # 检查索引有效性 / Check index validity
    if node_idx is None or node_idx < 0 or node_idx >= self.num_nodes:
        return False
    
    node_name = self.pipeline_nodes[node_idx]
    methods = self.methods_for_node[node_name]
    
    # 检查方法索引 / Check method index
    if method_idx is None or method_idx < 0 or method_idx >= len(methods):
        return False
    
    # 检查流水线约束 / Check pipeline constraints
    if self.current_step == 0 and node_name != 'N2':
        return False  # First must be N2
    elif self.current_step == self.num_nodes - 1 and node_name != 'N5':
        return False  # Last must be N5
    elif self.current_step > 0 and self.current_step < self.num_nodes - 1:
        if node_name in ['N2', 'N5']:
            return False  # Middle steps cannot be N2 or N5
    
    # 检查是否已访问 / Check if already visited
    if self.node_visited[node_idx]:
        return False
    
    return True
```

### 3. 方法执行 / Method Execution

位置：`env/pipeline_env.py:198-277`

```python
def step(self, action: Dict[str, Any]) -> Tuple[...]:
    """执行动作 / Execute action"""
    # 1. 验证动作 / Validate action
    if not self.select_node(action):
        return self._get_obs(), -1.0, True, False, {}
    
    # 2. 获取节点和方法名称 / Get node and method names
    node_idx = int(action['node'])
    method_idx = int(action['method'])
    node_name = self.pipeline_nodes[node_idx]
    methods = self.methods_for_node[node_name]
    method_name = methods[method_idx]
    
    # 3. 更新流水线配置 / Update pipeline config
    self.pipeline_config['sequence'].append(node_name)
    self.pipeline_config[f'{node_name}_method'] = method_name
    
    if node_name in self.param_nodes:
        params_dict = {'param': float(params[0])}
        self.pipeline_config[f'{node_name}_params'] = params_dict
    
    # 4. 最后一步执行流水线 / Execute pipeline on last step
    if self.current_step == self.num_nodes - 1:
        outputs = run_pipeline(**self.pipeline_config, verbose=False)
        metrics = outputs.get('metrics', {})
        
        # 计算奖励 / Calculate reward
        mae = metrics.get('mae_fe_test', 0.0)
        r2 = metrics.get('r2_fe_test', 0.0)
        reward = r2 - mae - complexity_penalty
        done = True
    
    return self._get_obs(), reward, done, False, metrics
```

---

## 完整示例 / Complete Example

### 示例：训练一个回合 / Example: Training One Episode

```python
# 初始化环境和训练器 / Initialize environment and trainer
env = PipelineEnv()
trainer = PPOTrainer(env, learning_rate=3e-4, hidden_size=64)

# 重置环境 / Reset environment
obs = env.reset()
# obs = {
#     'fingerprint': [mae, r2, n_features],
#     'node_visited': [False, False, False, False, False],
#     'action_mask': [1.0, 0.0, 0.0, 0.0, 0.0]  # Only N2 available
# }

# Step 1: 第一步必须选择N2 / First step must select N2
action, log_probs = trainer.select_action(obs)
# action = {'node': 0, 'method': 0, 'params': [0.5]}
# 对应: N2节点, 'default'方法

obs, reward, done, _, info = env.step(action)

# Step 2: 选择N1 (Imputation) / Select N1 (Imputation)
action, log_probs = trainer.select_action(obs)
# action = {'node': 1, 'method': 2, 'params': [0.3]}
# 对应: N1节点, 'knn'方法, 参数0.3
# 方法选择过程:
# - node_logits = [0.1, 0.8, 0.3, 0.2, 0.1]  # N1概率最高
# - N1有4个方法: ['mean', 'median', 'knn', 'none']
# - method_logits_masked = method_logits[:4]
# - 从4个方法中采样，选中index=2 → 'knn'

obs, reward, done, _, info = env.step(action)

# Step 3: 选择N3 (Feature Selection) / Select N3
action, log_probs = trainer.select_action(obs)
# action = {'node': 2, 'method': 1, 'params': [0.7]}
# 对应: N3节点, 'variance'方法, 参数0.7

obs, reward, done, _, info = env.step(action)

# Step 4: 选择N4 (Scaling) / Select N4
action, log_probs = trainer.select_action(obs)
# action = {'node': 3, 'method': 0, 'params': [0.5]}
# 对应: N4节点, 'std'方法

obs, reward, done, _, info = env.step(action)

# Step 5: 最后一步选择N5 (Model Training) / Last step select N5
action, log_probs = trainer.select_action(obs)
# action = {'node': 4, 'method': 1, 'params': [0.8]}
# 对应: N5节点, 'gbr'方法

obs, reward, done, _, info = env.step(action)
# 此时执行完整流水线，计算最终奖励
# Pipeline executes completely, final reward calculated
# reward = r2_score - mae - complexity_penalty
```

### 生成的流水线配置 / Generated Pipeline Config

```python
pipeline_config = {
    'sequence': ['N0', 'N2', 'N1', 'N3', 'N4', 'N5'],
    'N0_method': 'api',
    'N0_params': {},
    'N2_method': 'default',
    'N1_method': 'knn',
    'N1_params': {'param': 0.3},
    'N3_method': 'variance',
    'N3_params': {'param': 0.7},
    'N4_method': 'std',
    'N4_params': {'param': 0.5},
    'N5_method': 'gbr',
    'N5_params': {'param': 0.8}
}
```

---

## 关键要点总结 / Key Takeaways

### 1. 分层决策机制 / Hierarchical Decision Mechanism

PPO使用**三层决策**而非单一动作：
- **第一层**：选择节点（N1-N5）
- **第二层**：选择该节点的方法（取决于第一层）
- **第三层**：设置方法参数（0.0-1.0范围）

### 2. 动态方法屏蔽 / Dynamic Method Masking

方法选择是**动态屏蔽**的：
```python
# N1有4个方法 → method_logits[:4]
# N3有4个方法 → method_logits[:4]
# N5有4个方法 → method_logits[:4]
```

### 3. 概率分布采样 / Probability Distribution Sampling

使用**PyTorch分布**进行采样：
- 节点和方法：`Categorical(logits)` - 离散选择
- 参数：`Normal(mean, std)` - 连续值

### 4. 约束验证 / Constraint Validation

两层验证确保动作合法：
- **环境层**：`select_node()` 函数验证
- **掩码层**：`action_mask` 预先屏蔽

### 5. 学习机制 / Learning Mechanism

通过**PPO算法**持续优化：
```
损失函数 = 策略损失 + 价值损失 - 熵奖励
Loss = Policy Loss + Value Loss - Entropy Bonus
```

---

## 相关文件索引 / Related Files

| 文件 | 行数 | 说明 |
|------|------|------|
| `ppo/policy.py` | 12-105 | PPOPolicy神经网络定义 |
| `ppo/trainer.py` | 92-133 | select_action方法选择逻辑 |
| `ppo/trainer.py` | 199-258 | _update_policy学习更新 |
| `env/pipeline_env.py` | 38-44 | methods_for_node方法定义 |
| `env/pipeline_env.py` | 136-157 | _compute_action_mask动作掩码 |
| `env/pipeline_env.py` | 159-196 | select_node动作验证 |
| `env/pipeline_env.py` | 198-277 | step动作执行 |

---

## 进一步阅读 / Further Reading

- **PPO算法原理**：查看 `docs/PPO_TRAINING_ANALYSIS.md`
- **环境设计文档**：查看 `docs/ENV_PPO_DOCUMENTATION_STRUCTURE.md`
- **训练验证报告**：查看 `docs/PPO_VALIDATION_REPORT.md`
- **10节点升级计划**：查看 `NODE_ARCHITECTURE_SUMMARY.md`

---

**文档版本 Document Version**: 1.0  
**创建日期 Created**: 2025-11-04  
**作者 Author**: RL Material Pipeline Team
