# PPO方法选择快速参考 / PPO Method Selection Quick Reference

## 核心问题 / Core Question
**PPO如何选择方法？/ How does PPO select methods?**

---

## 简短答案 / Short Answer

PPO通过**三层神经网络决策**选择方法：

1. **神经网络输出** → 节点logits、方法logits、参数值
2. **动态屏蔽** → 根据选中节点屏蔽无效方法
3. **概率采样** → 从有效方法分布中随机采样

```python
# 核心代码 (ppo/trainer.py:105-113)
node_idx = int(node_action.item())
node_name = self.env.pipeline_nodes[node_idx]
num_methods = len(self.env.methods_for_node[node_name])

# 关键：只使用该节点的可用方法数量
method_logits_masked = method_logits[:num_methods]
method_dist = Categorical(logits=method_logits_masked)
method_action = method_dist.sample()  # 采样方法索引
```

---

## 详细流程 / Detailed Process

### Step 1: 神经网络推理 / Neural Network Inference

```
输入观察 obs → PPOPolicy → 输出4个值:
├─ node_logits[6]     # 6个节点的评分
├─ method_logits[10]  # 10个方法的评分 (所有节点共享)
├─ params[1]          # 参数值 0-1
└─ value[1]           # 状态价值
```

### Step 2: 节点选择 / Node Selection

```python
node_dist = Categorical(logits=node_logits)
node_action = node_dist.sample()
# 例如: node_action = 1 → N1节点
```

### Step 3: 方法动态屏蔽 / Method Dynamic Masking

```python
# N1有4个方法: ['mean', 'median', 'knn', 'none']
if node == N1:
    method_logits_masked = method_logits[:4]  # 只取前4个
elif node == N2:
    method_logits_masked = method_logits[:1]  # 只取1个
elif node == N3:
    method_logits_masked = method_logits[:4]  # 前4个
# ... 依此类推
```

### Step 4: 方法采样 / Method Sampling

```python
method_dist = Categorical(logits=method_logits_masked)
method_action = method_dist.sample()
# 例如: method_action = 2 → 'knn'
```

---

## 关键设计理念 / Key Design Principles

### 1. 共享方法空间 / Shared Method Space

神经网络输出**统一的10维method_logits**，而不是为每个节点单独输出。
- **优点**: 参数共享，减少网络复杂度
- **机制**: 通过动态屏蔽适配不同节点

### 2. 动态屏蔽策略 / Dynamic Masking Strategy

方法选择是**上下文相关**的：
- 先选节点 → 确定可用方法集合 → 屏蔽无效方法 → 从有效方法中采样

### 3. 概率分布采样 / Probability Distribution Sampling

使用PyTorch分布而非直接argmax：
- **探索性**: 保持探索-利用平衡
- **可微性**: 支持梯度反向传播
- **随机性**: 避免过早收敛

---

## 节点-方法映射表 / Node-Method Mapping Table

| 节点 | 可用方法数 | 方法列表 | 屏蔽范围 |
|------|-----------|---------|----------|
| **N1** | 4 | `mean`, `median`, `knn`, `none` | `[:4]` |
| **N2** | 1 | `default` | `[:1]` |
| **N3** | 4 | `none`, `variance`, `univariate`, `pca` | `[:4]` |
| **N4** | 4 | `std`, `robust`, `minmax`, `none` | `[:4]` |
| **N5** | 4 | `rf`, `gbr`, `xgb`, `cat` | `[:4]` |

---

## 完整示例 / Complete Example

```python
# 假设网络输出 / Assume network outputs:
node_logits = [0.1, 0.8, 0.3, 0.2, 0.1, 0.05]  # N1概率最高
method_logits = [0.2, 0.5, 0.9, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Step 1: 采样节点
node_action = sample(node_logits)  # → 1 (N1)

# Step 2: 获取N1的可用方法
node_name = 'N1'
methods = ['mean', 'median', 'knn', 'none']  # 4个方法
num_methods = 4

# Step 3: 屏蔽method_logits
method_logits_masked = method_logits[:4]
# = [0.2, 0.5, 0.9, 0.3]  # 只保留前4个

# Step 4: 采样方法
method_action = sample(method_logits_masked)  # → 2 (概率最高)
# → 'knn' (methods[2])

# 最终动作
action = {
    'node': 1,      # N1
    'method': 2,    # 'knn'
    'params': [0.5]
}
```

---

## 为什么这样设计？/ Why This Design?

### ✅ 优点 / Advantages

1. **参数效率**: 一个method_head服务所有节点
2. **灵活性**: 支持不同节点有不同数量的方法
3. **可扩展性**: 添加新节点/方法只需修改配置
4. **学习效率**: 方法之间可以共享特征表示

### ⚠️ 注意事项 / Caveats

1. **假设**: 所有节点的方法数 ≤ 10
2. **顺序重要**: 方法列表顺序必须与索引对应
3. **屏蔽关键**: 必须正确屏蔽，否则会选到无效方法

---

## 代码位置索引 / Code Location Index

| 功能 | 文件 | 行数 |
|------|------|------|
| 方法定义 | `env/pipeline_env.py` | 38-44 |
| 神经网络 | `ppo/policy.py` | 40-44 |
| 方法选择 | `ppo/trainer.py` | 105-113 |
| 动作验证 | `env/pipeline_env.py` | 159-196 |

---

## 进阶阅读 / Further Reading

- 📖 **完整文档**: `docs/PPO_METHOD_SELECTION_EXPLAINED.md`
- 📊 **流程图**: `docs/PPO_METHOD_SELECTION_FLOWCHART.md`
- 🧪 **代码示例**: `ppo/trainer.py`, `env/pipeline_env.py`
- 🎓 **训练分析**: `docs/PPO_TRAINING_ANALYSIS.md`

---

**文档版本**: 1.0 | **创建**: 2025-11-04 | **语言**: 中文/English
