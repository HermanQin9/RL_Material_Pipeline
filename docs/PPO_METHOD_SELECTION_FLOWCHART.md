# PPO方法选择流程图 / PPO Method Selection Flowchart

## 完整流程可视化 / Complete Flow Visualization

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        PPO方法选择完整流程图                                 ║
║                   PPO Method Selection Complete Flowchart                   ║
╚════════════════════════════════════════════════════════════════════════════╝

                           ┌─────────────────┐
                           │  开始训练回合    │
                           │  Start Episode  │
                           └────────┬────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   env.reset()                 │
                    │   初始化环境 / Initialize Env  │
                    │   - current_step = 0          │
                    │   - node_visited = [F,F,F...] │
                    │   - fingerprint = [0,0,0]     │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    获取观察 / Get Observation                     │
    │  obs = {                                                         │
    │    'fingerprint': [mae, r2, n_features],  # 流水线性能指纹       │
    │    'node_visited': [False, False, ...],   # 节点访问标志         │
    │    'action_mask': [1.0, 0.0, ...]         # 动作掩码             │
    │  }                                                               │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              PPO神经网络前向传播 / Neural Network Forward        │
    │                                                                  │
    │  Input: flatten(obs) → [13维tensor]                             │
    │         ↓                                                        │
    │  ┌─────────────────────────────────────────────────┐            │
    │  │  Shared Layers (特征提取)                        │            │
    │  │  Linear(13→256) → ReLU                          │            │
    │  │  Linear(256→256) → ReLU                         │            │
    │  │  Linear(256→128) → ReLU                         │            │
    │  └──────────────────┬──────────────────────────────┘            │
    │                     │                                            │
    │         ┌───────────┴───────────┐                                │
    │         ▼                       ▼                                │
    │  ┌─────────────┐      ┌─────────────────┐                       │
    │  │  Node Head  │      │  Method Head    │                       │
    │  │  128→64→6   │      │  128→64→10      │                       │
    │  └─────┬───────┘      └────────┬────────┘                       │
    │        │                       │                                 │
    │        ▼                       ▼                                 │
    │  node_logits[6]        method_logits[10]                        │
    │                                                                  │
    │         ┌─────────────┐      ┌─────────────────┐                │
    │         │  Param Head │      │  Value Head     │                │
    │         │  128→64→1   │      │  128→64→1       │                │
    │         └──────┬──────┘      └────────┬────────┘                │
    │                │                      │                          │
    │                ▼                      ▼                          │
    │         params[0-1]            value[scalar]                    │
    └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │            Step 1: 节点选择 / Node Selection                     │
    │                                                                  │
    │  node_logits = [logit_N2, logit_N1, logit_N3, logit_N4, ...N5] │
    │                    ↓                                             │
    │  Categorical Distribution:                                      │
    │    node_dist = Categorical(logits=node_logits)                  │
    │                    ↓                                             │
    │  Sample:                                                        │
    │    node_action ~ node_dist                                      │
    │    例如: node_action = 1 (对应N1节点)                            │
    │                    ↓                                             │
    │  Log Probability:                                               │
    │    node_log_prob = node_dist.log_prob(node_action)              │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │          Step 2: 方法选择 / Method Selection                     │
    │                                                                  │
    │  # 获取选中节点的可用方法                                          │
    │  node_name = pipeline_nodes[node_action]  # 'N1'                │
    │  methods = methods_for_node['N1']  # ['mean','median','knn'...] │
    │  num_methods = len(methods)  # 4                                │
    │                    ↓                                             │
    │  # 屏蔽多余的method_logits (关键步骤!)                            │
    │  method_logits = [logit_m0, logit_m1, ..., logit_m9]            │
    │  method_logits_masked = method_logits[:num_methods]  # 只取前4个 │
    │                    ↓                                             │
    │  Categorical Distribution:                                      │
    │    method_dist = Categorical(logits=method_logits_masked)       │
    │                    ↓                                             │
    │  Sample:                                                        │
    │    method_action ~ method_dist                                  │
    │    例如: method_action = 2 (对应'knn'方法)                       │
    │                    ↓                                             │
    │  Log Probability:                                               │
    │    method_log_prob = method_dist.log_prob(method_action)        │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │          Step 3: 参数采样 / Parameter Sampling                   │
    │                                                                  │
    │  params = 0.5  # 从神经网络输出 (0-1范围)                         │
    │                    ↓                                             │
    │  Normal Distribution:                                           │
    │    param_dist = Normal(mean=params, std=0.1)                    │
    │                    ↓                                             │
    │  Sample:                                                        │
    │    param_action ~ param_dist                                    │
    │                    ↓                                             │
    │  Clip:                                                          │
    │    param_action = clamp(param_action, 0.0, 1.0)                 │
    │    例如: param_action = 0.53                                     │
    │                    ↓                                             │
    │  Log Probability:                                               │
    │    param_log_prob = param_dist.log_prob(param_action).sum()     │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              组装动作 / Assemble Action                          │
    │                                                                  │
    │  action = {                                                     │
    │    'node': 1,           # N1节点                                 │
    │    'method': 2,         # 'knn'方法                              │
    │    'params': [0.53]     # 参数值                                 │
    │  }                                                              │
    │                                                                  │
    │  log_probs = {                                                  │
    │    'node': node_log_prob,                                       │
    │    'method': method_log_prob,                                   │
    │    'param': param_log_prob,                                     │
    │    'value': value                                               │
    │  }                                                              │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │          环境验证 / Environment Validation                       │
    │                                                                  │
    │  select_node(action) → bool:                                    │
    │    ✓ 检查node索引范围 [0-4]                                      │
    │    ✓ 检查method索引范围 [0-num_methods)                          │
    │    ✓ 检查流水线约束:                                              │
    │      - Step 0 必须是 N2                                          │
    │      - Step 4 必须是 N5                                          │
    │      - 中间步骤不能是 N2 或 N5                                    │
    │    ✓ 检查节点是否已访问                                           │
    │                                                                  │
    │  如果验证失败 → return (obs, -1.0, True, ...)                    │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              执行动作 / Execute Action                           │
    │                                                                  │
    │  1. 更新环境状态:                                                 │
    │     - node_visited[node_idx] = True                             │
    │     - method_calls[method_name] += 1                            │
    │     - current_step += 1                                         │
    │                                                                  │
    │  2. 更新流水线配置:                                               │
    │     - pipeline_config['sequence'].append(node_name)             │
    │     - pipeline_config[f'{node_name}_method'] = method_name      │
    │     - pipeline_config[f'{node_name}_params'] = {'param': ...}   │
    │                                                                  │
    │  3. 如果是最后一步 (step == 4):                                   │
    │     - 执行完整流水线: run_pipeline(**pipeline_config)            │
    │     - 获取性能指标: mae, r2_score                                 │
    │     - 计算奖励: reward = r2 - mae - complexity_penalty          │
    │     - done = True                                               │
    │                                                                  │
    │  4. 返回新状态:                                                   │
    │     return (next_obs, reward, done, truncated, info)            │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────┴─────────────┐
                    │                            │
                    │   done == True?            │
                    │                            │
                    └──┬─────────────────────┬───┘
                       │ No                  │ Yes
                       │                     │
                       ▼                     ▼
            ┌───────────────────┐    ┌──────────────────┐
            │  继续下一步         │    │  存储经验并学习   │
            │  Continue          │    │  Store & Learn   │
            │                    │    │                  │
            │  回到"获取观察"     │    │  - 计算回报      │
            └──────┬─────────────┘    │  - 计算优势      │
                   │                  │  - PPO更新       │
                   │                  │  - 梯度下降      │
                   └──────────────────┤                  │
                                      └────────┬─────────┘
                                               │
                                               ▼
                                    ┌────────────────────┐
                                    │  回合结束           │
                                    │  Episode Complete  │
                                    └────────────────────┘
```

---

## 方法选择决策树 / Method Selection Decision Tree

```
观察状态 obs
    │
    ├─ Step 0: current_step = 0
    │       │
    │       └─> action_mask = [1,0,0,0,0]  # 只能选N2
    │               │
    │               └─> 强制选择: node=0 (N2), method=0 ('default')
    │
    ├─ Step 1: current_step = 1
    │       │
    │       └─> action_mask = [0,1,1,1,0]  # 可选N1,N3,N4
    │               │
    │               ├─> 如果选N1 → 可用方法: ['mean','median','knn','none'] (4个)
    │               │       └─> method_logits_masked = method_logits[:4]
    │               │
    │               ├─> 如果选N3 → 可用方法: ['none','variance','univariate','pca'] (4个)
    │               │       └─> method_logits_masked = method_logits[:4]
    │               │
    │               └─> 如果选N4 → 可用方法: ['std','robust','minmax','none'] (4个)
    │                       └─> method_logits_masked = method_logits[:4]
    │
    ├─ Step 2: current_step = 2
    │       │
    │       └─> action_mask = [0,X,X,X,0]  # X表示取决于Step 1选择
    │               │
    │               └─> 只能选未访问的N1/N3/N4节点
    │
    ├─ Step 3: current_step = 3
    │       │
    │       └─> action_mask = [0,X,X,X,0]  # 只剩1个未访问节点
    │               │
    │               └─> 自动选择最后一个N1/N3/N4节点
    │
    └─ Step 4: current_step = 4
            │
            └─> action_mask = [0,0,0,0,1]  # 只能选N5
                    │
                    └─> 强制选择: node=4 (N5)
                            │
                            ├─> 可用方法: ['rf','gbr','xgb','cat'] (4个)
                            │       └─> method_logits_masked = method_logits[:4]
                            │
                            └─> 执行完整流水线，计算最终奖励
```

---

## 神经网络输出到方法的映射 / Neural Network Output to Method Mapping

```
神经网络输出 / Neural Network Output:
┌─────────────────────────────────────────────────────────────┐
│ method_logits = [logit_0, logit_1, ..., logit_9]  (10维)   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            根据选中节点动态屏蔽 / Dynamic Masking by Selected Node
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  N1 (4方法)   │   │  N3 (4方法)   │   │  N5 (4方法)   │
│               │   │               │   │               │
│ Mask: [:4]    │   │ Mask: [:4]    │   │ Mask: [:4]    │
│               │   │               │   │               │
│ Index → 方法: │   │ Index → 方法: │   │ Index → 方法: │
│ 0 → 'mean'    │   │ 0 → 'none'    │   │ 0 → 'rf'      │
│ 1 → 'median'  │   │ 1 → 'variance'│   │ 1 → 'gbr'     │
│ 2 → 'knn'     │   │ 2 → 'univar.' │   │ 2 → 'xgb'     │
│ 3 → 'none'    │   │ 3 → 'pca'     │   │ 3 → 'cat'     │
└───────────────┘   └───────────────┘   └───────────────┘

        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  N2 (1方法)   │   │  N4 (4方法)   │   │               │
│               │   │               │   │               │
│ Mask: [:1]    │   │ Mask: [:4]    │   │               │
│               │   │               │   │               │
│ 0 → 'default' │   │ 0 → 'std'     │   │               │
│               │   │ 1 → 'robust'  │   │               │
│               │   │ 2 → 'minmax'  │   │               │
│               │   │ 3 → 'none'    │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## 学习更新流程 / Learning Update Flow

```
┌──────────────────────────────────────────────────────────────┐
│          收集完整轨迹 / Collect Complete Trajectory           │
│                                                               │
│  trajectory = [                                              │
│    (obs_0, action_0, reward_0, log_prob_0, value_0, done_0),│
│    (obs_1, action_1, reward_1, log_prob_1, value_1, done_1),│
│    ...                                                       │
│    (obs_4, action_4, reward_4, log_prob_4, value_4, done_4) │
│  ]                                                           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              计算折扣回报 / Compute Returns                   │
│                                                               │
│  returns = []                                                │
│  R = 0                                                       │
│  for t in reversed(range(T)):                                │
│      if done[t]: R = 0                                       │
│      R = reward[t] + gamma * R                               │
│      returns.insert(0, R)                                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│            计算优势函数 / Compute Advantage                    │
│                                                               │
│  advantages = returns - values                               │
│  advantages = (advantages - mean) / (std + 1e-8)             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              PPO更新 (4 epochs) / PPO Update                  │
│                                                               │
│  for epoch in range(4):                                      │
│    for each step in trajectory:                              │
│                                                               │
│      1. 重新前向传播 / Re-forward pass                        │
│         node_logits, method_logits, params, value = policy(obs)│
│                                                               │
│      2. 重新计算概率 / Re-compute probabilities               │
│         new_log_prob = node_log_prob + method_log_prob +     │
│                        param_log_prob                        │
│                                                               │
│      3. 计算比率 / Compute ratio                              │
│         ratio = exp(new_log_prob - old_log_prob)             │
│                                                               │
│      4. PPO裁剪损失 / Clipped surrogate loss                  │
│         surr1 = ratio * advantage                            │
│         surr2 = clamp(ratio, 1-ε, 1+ε) * advantage           │
│         policy_loss = -min(surr1, surr2)                     │
│                                                               │
│      5. 价值损失 / Value loss                                 │
│         value_loss = MSE(value, return)                      │
│                                                               │
│      6. 熵奖励 / Entropy bonus                                │
│         entropy = node_entropy + method_entropy +            │
│                   param_entropy                              │
│                                                               │
│      7. 总损失 / Total loss                                   │
│         loss = policy_loss + 0.5*value_loss - 0.01*entropy   │
│                                                               │
│      8. 梯度下降 / Gradient descent                           │
│         optimizer.zero_grad()                                │
│         loss.backward()                                      │
│         optimizer.step()                                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                ┌────────────────────┐
                │  策略更新完成       │
                │  Policy Updated    │
                └────────────────────┘
```

---

## 关键代码映射 / Key Code Mapping

| 流程图步骤 | 代码位置 | 函数/方法 |
|-----------|----------|-----------|
| 获取观察 | `env/pipeline_env.py:124-134` | `_get_obs()` |
| 计算动作掩码 | `env/pipeline_env.py:136-157` | `_compute_action_mask()` |
| 神经网络前向 | `ppo/policy.py:61-101` | `PPOPolicy.forward()` |
| 节点选择 | `ppo/trainer.py:100-103` | 创建Categorical分布并采样 |
| 方法选择 | `ppo/trainer.py:105-113` | 动态屏蔽并采样 |
| 参数采样 | `ppo/trainer.py:115-118` | Normal分布并裁剪 |
| 动作验证 | `env/pipeline_env.py:159-196` | `select_node()` |
| 动作执行 | `env/pipeline_env.py:198-277` | `step()` |
| 计算回报 | `ppo/trainer.py:186-196` | `_compute_returns()` |
| PPO更新 | `ppo/trainer.py:199-258` | `_update_policy()` |

---

**流程图版本 Flowchart Version**: 1.0  
**创建日期 Created**: 2025-11-04  
**作者 Author**: RL Material Pipeline Team
