# 损失函数实现指南

## 一、交叉熵损失与负对数似然的关系

### 数学上：完全等价

**交叉熵损失（Cross Entropy）**：
$$\text{CE} = -\sum_{i} y_i \log(p_i)$$

**负对数似然损失（Negative Log-Likelihood）**：
$$\text{NLL} = -\log(p_{\text{真实类}})$$

当 $y$ 是 one-hot 编码时，两个公式退化成同一个：
$$\text{CE} = -\log(p_{\text{真实类}}) = \text{NLL}$$

### 工程上：输入格式不同

| 函数 | 输入 | 内部操作 | PyTorch 对应 |
|------|------|----------|-------------|
| CrossEntropyLoss | logits（原始分数） | 内部做 log_softmax | `nn.CrossEntropyLoss` |
| NLLLoss | log-probabilities | 不做 softmax，直接挑选 | `nn.NLLLoss` |

**等价关系**：
```python
CrossEntropyLoss = log_softmax + NLLLoss
```

---

## 二、多分类交叉熵（索引格式）

### 2.1 公式

$$\text{loss} = -\log(p_{\text{真实类}}) = -\log\left(\frac{e^{z_{\text{真实类}}}}{\sum_j e^{z_j}}\right)$$

其中 $z$ 是 logits（原始分数），$p$ 是 softmax 后的概率。

### 2.2 输入输出

```python
输入：
  - pred_scores: (batch_size, num_classes)  # 原始分数（logits）
  - target: (batch_size,)                    # 类别索引，值在 [0, num_classes-1]

输出：
  - loss: 标量                               # 平均损失
```

### 2.3 维度变化流程

```
pred_scores: (bs, n)                    # 输入原始分数
    ↓ F.log_softmax(dim=-1)
log_probs: (bs, n)                      # log(概率)
    ↓ [batch_index, target] 高级索引挑选
selected_log_probs: (bs,)               # 每个样本对应真实类的 log(概率)
    ↓ 取负号
loss_per_sample: (bs,)                  # 每个样本的损失
    ↓ .mean()
loss: 标量                               # 最终损失
```

### 2.4 完整实现

#### 版本 1：分步（教学清晰）

```python
def cross_entropy_by_index(pred_scores, target):
    batch_size = pred_scores.shape[0]
    
    # 步骤 1: softmax 变概率
    probs = F.softmax(pred_scores, dim=-1)  # (bs, n)
    
    # 步骤 2: 取对数
    log_probs = torch.log(probs)            # (bs, n)
    
    # 步骤 3: 挑出每个样本对应真实类的 log(概率)
    batch_index = torch.arange(batch_size)
    selected_log_probs = log_probs[batch_index, target]  # (bs,)
    
    # 步骤 4: 取负号
    loss = -selected_log_probs              # (bs,)
    
    # 步骤 5: 求平均
    return loss.mean()                      # 标量
```

**中间变量维度**：
- `probs`: `(bs, n)` — 每个样本在 n 个类上的概率
- `log_probs`: `(bs, n)` — 对数概率
- `batch_index`: `(bs,)` — [0, 1, 2, ..., bs-1]
- `selected_log_probs`: `(bs,)` — 只保留真实类的 log(概率)
- `loss`: 标量

#### 版本 2：优化（数值稳定）

```python
def cross_entropy_by_index(pred_scores, target):
    batch_size = pred_scores.shape[0]
    
    # log_softmax = softmax + log 两步合并，数值更稳定
    log_probs = F.log_softmax(pred_scores, dim=-1)  # (bs, n)
    
    batch_index = torch.arange(batch_size)
    selected_log_probs = log_probs[batch_index, target]  # (bs,)
    
    loss = -selected_log_probs  # (bs,)
    return loss.mean()
```

**为什么更稳定？**  
`log_softmax` 内部使用 log-sum-exp trick，避免了 `exp` 溢出和 `log(0)` 的问题。

### 2.5 高级索引详解

```python
log_probs = [[−1.0, −2.0, −3.0],   # 样本 0
             [−0.5, −1.5, −2.5],   # 样本 1
             [−2.0, −1.0, −3.0]]   # 样本 2

target = [1, 0, 1]  # 真实类别

batch_index = [0, 1, 2]

# 挑选逻辑
selected_log_probs = [
    log_probs[0, 1],  # -2.0
    log_probs[1, 0],  # -0.5
    log_probs[2, 1]   # -1.0
]
# 结果: [-2.0, -0.5, -1.0]
```

---

## 三、多分类交叉熵（one-hot 格式）

### 3.1 公式

$$\text{loss} = -\sum_{i=0}^{n-1} y_i \log(p_i)$$

其中 $y$ 是 one-hot 向量（只有真实类位置是 1，其他是 0）。

### 3.2 输入输出

```python
输入：
  - pred_scores: (batch_size, num_classes)  # 原始分数
  - one_hot: (batch_size, num_classes)      # one-hot 编码，如 [0, 1, 0]

输出：
  - loss: 标量
```

### 3.3 维度变化流程

```
pred_scores: (bs, n)                    # 输入原始分数
    ↓ F.log_softmax(dim=-1)
log_probs: (bs, n)                      # log(概率)
    ↓ one_hot * log_probs 逐元素乘
masked: (bs, n)                         # 只保留真实类位置，其他为 0
    ↓ .sum(dim=-1)
loss_per_sample: (bs,)                  # 每个样本的损失（求和后只有一个非零项）
    ↓ 取负号
loss_per_sample: (bs,)                  
    ↓ .mean()
loss: 标量
```

### 3.4 完整实现

```python
def cross_entropy_by_onehot(pred_scores, one_hot):
    # log_softmax
    log_probs = F.log_softmax(pred_scores, dim=-1)  # (bs, n)
    
    # 逐元素乘：只保留 one_hot=1 的位置
    # one_hot: [[0, 1, 0], [1, 0, 0], ...]
    # log_probs: [[-1.0, -2.0, -3.0], ...]
    # 结果: [[0, -2.0, 0], [-1.0, 0, 0], ...]
    masked = one_hot * log_probs  # (bs, n)
    
    # 求和：每行只有一个非零值（真实类）
    loss = -masked.sum(dim=-1)  # (bs,)
    
    return loss.mean()
```

### 3.5 逐元素乘的作用

```python
one_hot  = [[0, 1, 0]]      # 真实类是第 1 类
log_probs = [[-1.0, -2.0, -3.0]]

masked = one_hot * log_probs = [[0, -2.0, 0]]
# 只保留第 1 类的 log(概率)

.sum(dim=-1) = -2.0
-(...) = 2.0  # 最终损失
```

**本质**：one-hot 乘法相当于"用 0 把非真实类的位置全屏蔽掉"。

---

## 四、二分类交叉熵（BCE）

### 4.1 公式

$$\text{BCE} = -(y \log(p) + (1-y) \log(1-p))$$

其中：
- $y \in \{0, 1\}$ 是真实标签（0=负类，1=正类）
- $p = \text{sigmoid}(z)$ 是正类概率
- $1-p$ 是负类概率

### 4.2 为什么有 `1-p` 项？

因为 **Sigmoid 只输出一个值**（正类概率 $p$），负类概率是 **人为定义的** `1-p`。

- 当 $y=1$（正样本）：loss = $-\log(p)$，要求 $p$ 尽量大
- 当 $y=0$（负样本）：loss = $-\log(1-p)$，要求 $p$ 尽量小（即 $1-p$ 大）

**对比 Softmax**：多分类用 Softmax 输出 n 个值，每个负类都有自己的 logit，所以不需要显式写负类项。

### 4.3 输入输出

```python
输入：
  - scores: (batch_size,)     # 原始分数（logits），标量
  - target: (batch_size,)     # 0 或 1

输出：
  - loss: 标量
```

### 4.4 维度变化流程

```
scores: (bs,)                           # 输入原始分数
    ↓ torch.sigmoid()
probs: (bs,)                            # 正类概率
    ↓ target * log(probs) 计算正样本贡献
positive_term: (bs,)                    # 正样本的 log(p)
    ↓ (1-target) * log(1-probs) 计算负样本贡献
negative_term: (bs,)                    # 负样本的 log(1-p)
    ↓ 相加取负
loss_per_sample: (bs,)                  
    ↓ .mean()
loss: 标量
```

### 4.5 完整实现

```python
def binary_cross_entropy(scores, target):
    # Sigmoid 得到正类概率
    probs = torch.sigmoid(scores)  # (bs,)
    
    # 正样本贡献：y * log(p)
    # 负样本贡献：(1-y) * log(1-p)
    loss = -(target * torch.log(probs) + 
             (1 - target) * torch.log(1 - probs))  # (bs,)
    
    return loss.mean()
```

### 4.6 逐样本分析

```python
target = torch.tensor([1, 0, 1])       # 正、负、正
probs  = torch.tensor([0.9, 0.2, 0.6]) # 预测的正类概率

# 样本 0: y=1, p=0.9
# 贡献 = -(1 * log(0.9) + 0 * log(0.1)) = -log(0.9) = 0.105

# 样本 1: y=0, p=0.2
# 贡献 = -(0 * log(0.2) + 1 * log(0.8)) = -log(0.8) = 0.223

# 样本 2: y=1, p=0.6
# 贡献 = -(1 * log(0.6) + 0 * log(0.4)) = -log(0.6) = 0.511

# 平均 loss = (0.105 + 0.223 + 0.511) / 3 = 0.280
```

---

## 五、二分类的两种方案对比

### 方案 A：Sigmoid + BCE（专用）

```python
# 网络输出：1 个神经元
logit: (batch_size,) 或 (batch_size, 1)

# 激活函数
p = sigmoid(logit)  # 正类概率

# 损失
BCE = -(y·log(p) + (1-y)·log(1-p))
```

**特点**：
- 只需 1 个输出神经元
- 负类概率是 `1-p`（隐式）

### 方案 B：Softmax + CE（通用）

```python
# 网络输出：2 个神经元
logits: (batch_size, 2)

# 激活函数
probs = softmax(logits)  # [p0, p1]，p0 + p1 = 1

# 损失
CE = -log(p_真实类)
```

**特点**：
- 需要 2 个输出神经元
- 负类有自己的 logit 和概率

### 数学上等价

如果设定 `logits = [0, z]`（第 0 类固定为 0）：
$$p_1 = \frac{e^z}{1 + e^z} = \text{sigmoid}(z)$$
$$p_0 = 1 - p_1$$

**两种方案完全等价！**

### 实际选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 纯二分类任务 | Sigmoid + BCE | 省一个神经元，更简洁 |
| 可能扩展成多分类 | Softmax + CE | 统一架构 |
| 多标签分类 | Sigmoid（每类独立） | 允许多个类同时为正 |

---

## 六、多分类 vs 二分类对比总结

| 特性 | 多分类交叉熵 | 二分类交叉熵（BCE） |
|------|-------------|---------------------|
| **类别数** | n ≥ 2 | 2 |
| **激活函数** | Softmax | Sigmoid |
| **输出维度** | `(bs, n)` | `(bs,)` 或 `(bs, 1)` |
| **公式** | $-\log(p_{\text{真实类}})$ | $-(y\log(p) + (1-y)\log(1-p))$ |
| **是否有负类项** | 不需要显式写（Softmax 隐式处理） | 需要 `(1-y)log(1-p)` |
| **为什么不同** | Softmax 保证 $\sum p_i=1$，正类上升→负类自动下降 | Sigmoid 只输出一个值，需显式优化两边 |
| **PyTorch** | `nn.CrossEntropyLoss` | `nn.BCELoss` / `nn.BCEWithLogitsLoss` |

---

## 七、常见面试追问

### Q1: "为什么交叉熵用 `-log` 而不是 `1-p`？"

**A**: 
1. **信息论基础**：`-log(p)` 衡量的是"编码这个事件需要多少信息量"
2. **数值特性**：
   - p=1 → loss=0（完美）
   - p→0 → loss→∞（严重惩罚）
3. **梯度特性**：和 softmax 组合时梯度简洁，不易消失

### Q2: "为什么 softmax 和 log 要分开算？"

**A**: 可以一起算，但直接 `log(softmax(...))` 数值不稳定。实际用 `F.log_softmax`（内部做了优化）或 `F.cross_entropy`（一步到位）。

优化版：
```python
log_probs = F.log_softmax(logits, dim=-1)  # 数值稳定
loss = -log_probs[batch_index, target].mean()
```

### Q3: "多分类不考虑负样本吗？"

**A**: 考虑了，但通过 Softmax 隐式考虑。Softmax 保证所有概率和为 1，提高正类概率的唯一方法是降低所有负类概率。所以优化 `-log(p_正)` 时，模型自动学会压低负类。

### Q4: "Sigmoid 没有'和为 1'的约束吗？"

**A**: 没有。Sigmoid 只保证单个输出在 [0,1]，`p + (1-p) = 1` 是数学恒等式，不是 Sigmoid 强制的优化约束。Softmax 的分母包含所有类，所以能强制和为 1。

### Q5: "二分类能用 Softmax 吗？"

**A**: 完全可以。把二分类当成"2 类的多分类"，输出 2 个神经元，用 `CrossEntropyLoss`。效果和 BCE 一样，只是多一个参数。

---

## 八、代码实现对照表

| 损失函数 | 输入格式 | 激活函数 | PyTorch 等价 |
|---------|---------|---------|-------------|
| 多分类（索引） | logits `(bs, n)`, target `(bs,)` | Softmax | `F.cross_entropy` |
| 多分类（one-hot） | logits `(bs, n)`, one_hot `(bs, n)` | Softmax | 手动实现 |
| 二分类 | logits `(bs,)`, target `(bs,)` | Sigmoid | `F.binary_cross_entropy_with_logits` |

**记忆口诀**：
- **多分类**：Softmax 一家亲，正类上升负类沉
- **二分类**：Sigmoid 管一边，两边都要显式见
- **`1-p` 的本质**：二分类 Sigmoid 只输出正类概率，负类是推导的

---

## 九、维度变化速查表

### 多分类（索引）
```
logits: (bs, n) → log_softmax → (bs, n) → 高级索引 → (bs,) → mean → 标量
```

### 多分类（one-hot）
```
logits: (bs, n) → log_softmax → (bs, n) → *one_hot → (bs, n) → sum(dim=-1) → (bs,) → mean → 标量
```

### 二分类
```
logits: (bs,) → sigmoid → (bs,) → BCE公式 → (bs,) → mean → 标量
```

**关键点**：
- `log_softmax` / `sigmoid` 不改变维度
- 高级索引 / sum 会降维（从 `(bs, n)` → `(bs,)`）
- 最后的 `mean()` 把 `(bs,)` 变成标量

---

## 十、回归损失函数

### 10.1 不确定性损失（NLL）基础

#### Gaussian NLL vs Laplace NLL

在轨迹预测等场景中，常用不确定性损失来建模预测误差的分布：

| 分布 | 公式（完整版） | 简化版（优化等价） | 对应基础损失 |
|------|---------------|-------------------|-------------|
| **Gaussian** | `log(σ√2π) + (y-μ)²/(2σ²)` | `log(σ) + (y-μ)²/(2σ²)` | L2 (MSE) |
| **Laplace** | `log(2b) + |y-μ|/b` | `log(b) + |y-μ|/b` | L1 (MAE) |

**为什么可以简化？**  
常数项 `log(√2π)` 和 `log(2)` 在优化时梯度为 0，不影响最优解位置，所以省略。

**记忆口诀**：
```
两个公式骨架相同：log(不确定性) + 归一化误差
高斯平方除二西格方，拉式绝对除个b
```

#### 数值稳定实现

**问题**：直接预测 σ 或 b 有数值风险：
- 网络可能输出负值 → `log(负数)` 无效
- 非常小的值 → `log(0)` = `-∞`

**解决方案**：预测 `log(σ²)` 或 `log(b)`

```python
# Gaussian NLL（稳定版）
log_var = model.predict()  # 预测 log(σ²)，无约束
loss = 0.5 * (torch.exp(-log_var) * (y - mu)**2 + log_var)

# Laplace NLL（稳定版）
log_b = model.predict()  # 预测 log(b)，无约束
loss = torch.exp(-log_b) * torch.abs(y - mu) + log_b
```

**为什么稳定？**
1. 网络输出无约束（可正可负）
2. `exp` 保证结果非负
3. `log_var` 限制在合理范围（如 `[-10, 10]`）避免极端值

---

### 10.2 不确定性损失 vs 普通 MAE/MSE

#### 能否替代？

**可以，而且通常更好**。NLL 是 MAE/MSE 的泛化：

```python
# 数学上的等价关系
Gaussian NLL (固定 σ=1) → (y-μ)²/2  ≡  MSE
Laplace NLL  (固定 b=1) → |y-μ|     ≡  MAE
```

#### NLL 的优势

1. **自适应权重**  
   对噪声大的样本，模型自动预测大的 σ/b，降低该样本的梯度贡献：
   ```python
   # 样本 A: 噪声大 → σ 大 → (y-μ)²/σ² 小 → 梯度小
   # 样本 B: 噪声小 → σ 小 → (y-μ)²/σ² 大 → 梯度大
   ```

2. **输出可用于下游**  
   预测的 σ/b 可直接用于规划、风险评估（普通 MSE 没有这个输出）

3. **异方差数据友好**  
   轨迹数据天然异方差（直道误差小、弯道误差大），NLL 能自适应

#### 风险与注意事项

| 问题 | 说明 | 解决方案 |
|------|------|---------|
| **不确定性坍缩** | 模型通过预测极大 σ"作弊"，loss→0 但质量差 | 用 `log_var` 并限制范围 |
| **训练不稳定** | 早期 σ 不准导致梯度爆炸 | 预测 `log(σ²)` 代替直接预测 σ |
| **小数据集** | 额外参数需要更多数据 | 小数据集优先用 MAE/MSE |

#### 使用建议

```
场景                          推荐损失
────────────────────────────────────────
只关心点预测，数据均匀         MAE / MSE
数据噪声不均匀                 Laplace / Gaussian NLL
下游需要不确定性（规划）       NLL（必须）
轨迹预测、自动驾驶             Laplace NLL（业界主流）
```

---

### 10.3 Huber Loss 实现细节

#### 公式

Huber Loss 结合 L2（小误差）和 L1（大误差）的优点：

$$
L(e) = \begin{cases}
\frac{1}{2}e^2 & \text{if } |e| \leq \delta \\
\delta(|e| - \frac{1}{2}\delta) & \text{if } |e| > \delta
\end{cases}
$$

**设计目的**：
- 小误差用 L2 → 平滑，梯度稳定
- 大误差用 L1 → 对离群点鲁棒
- 在 `|e| = δ` 处连续可导

#### 常见实现错误

**❌ 错误 1：大误差区域保持原值**
```python
abs_err = torch.abs(pred - target)
abs_err[abs_err <= delta] = 0.5 * abs_err[abs_err <= delta]**2
# 大误差没处理，还是 |err| → 这是 MAE，不是 Huber！
```

**❌ 错误 2：原地修改时索引混乱**
```python
abs_err[small] = 0.5 * abs_err[small]**2  # 先修改小误差
abs_err[large] = delta * abs_err[large]   # 但 abs_err[large] 已经变了！
```

**✅ 正确方案 1：先计算 mask**
```python
abs_err = torch.abs(pred - target)
small_mask = abs_err <= delta
large_mask = ~small_mask  # 基于原始 abs_err 计算

# 分别处理（顺序无关，mask 已固定）
abs_err[small_mask] = 0.5 * abs_err[small_mask]**2
abs_err[large_mask] = delta * (abs_err[large_mask] - 0.5 * delta)
return abs_err.mean()
```

**✅ 正确方案 2：使用 `torch.where`（推荐）**
```python
abs_err = torch.abs(pred - target)
loss = torch.where(
    abs_err <= delta,
    0.5 * abs_err**2,
    delta * (abs_err - 0.5 * delta)
)
return loss.mean()
```

#### 归一化版本的等价性

有时会看到这样的写法：
```python
# L2 区域除以 delta
abs_err[l2_region] = 0.5 * abs_err[l2_region]**2 / delta
# L1 区域去掉 delta 系数
abs_err[l1_region] = abs_err[l1_region] - 0.5 * delta
```

**这是标准 Huber Loss 除以 δ 的归一化版本**，数学上等价：
- ✅ 梯度方向不变
- ✅ 最优解位置不变
- ⚠️ 损失数值范围变化
- ⚠️ 如果与其他损失组合，权重需重新平衡

**建议**：非标准写法需加注释说明：
```python
# 注意：归一化版本 Huber Loss（整体除以 delta）
# 避免 delta 过大导致损失尺度失衡
```

---

### 10.4 Laplace NLL 实现陷阱

#### 问题：参数顺序与命名

PyTorch 惯例通常是 `loss(pred, target)`，但手写时容易搞反：

```python
# ❌ 不符合惯例
def forward(self, target, pred):
    pass

# ✅ 推荐
def forward(self, pred, target):
    pass
```

#### 问题：未使用的 `reduction` 参数

```python
# ❌ 定义了但没用
def __init__(self, reduction='MAE'):  # MAE 不是 reduction 类型
    self.reduction = 'MAE'

def forward(self, pred, target):
    return loss.mean()  # 固定用了 mean，没用 self.reduction
```

**解决方案**：要么删掉，要么实现它：

```python
def __init__(self, reduction='mean'):
    self.reduction = reduction

def forward(self, pred, target):
    # ... 计算 loss ...
    if self.reduction == 'mean':
        return loss.mean()
    elif self.reduction == 'sum':
        return loss.sum()
    elif self.reduction == 'none':
        return loss  # 返回每个元素的 loss
```

#### 问题：`abs` vs `torch.abs`

```python
# ❌ Python 内置函数，梯度可能不正确
loss = log_scale + abs(target - loc) / scale

# ✅ PyTorch 版本，保证梯度正确
loss = log_scale + torch.abs(target - loc) / scale
```

#### 完整实现模板

```python
class LaplaceNLLoss(nn.Module):
    """
    Laplace 负对数似然损失（数值稳定版）
    假设模型预测 log(b) 而不是 b
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, agent, steps, 4) - [log_b_x, log_b_y, loc_x, loc_y]
            target: (batch, agent, steps, 2) - 真值位置
        """
        log_scale = pred[..., :2]  # 前两维：log(b_x), log(b_y)
        loc = pred[..., 2:]        # 后两维：μ_x, μ_y
        
        # 数值稳定性：限制 log_scale 范围
        log_scale = torch.clamp(log_scale, min=-10, max=10)
        
        # Laplace NLL = log(b) + |y-μ|/b
        #              = log_b + |y-μ| * exp(-log_b)
        lap_nll = log_scale + torch.abs(target - loc) * torch.exp(-log_scale)
        
        if self.reduction == 'mean':
            return lap_nll.mean()
        elif self.reduction == 'sum':
            return lap_nll.sum()
        elif self.reduction == 'none':
            return lap_nll
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
```

---

### 10.5 张量切片常见错误

#### Python 不支持 `..` 语法

```python
# ❌ 语法错误
scale = pred[:, :, :, 0..1]
loc = pred[:, :, :, 2..3]

# ✅ 使用切片 [start:end]（左闭右开）
scale = pred[:, :, :, 0:2]  # 索引 0, 1
loc = pred[:, :, :, 2:4]    # 索引 2, 3

# ✅ 使用 ... 省略前面维度
scale = pred[..., 0:2]
loc = pred[..., 2:4]

# ✅ 使用 torch.split（更语义化）
scale, loc = torch.split(pred, 2, dim=-1)
```

**记忆**：
- `[a:b]` = 索引 a 到 b-1（**左闭右开**）
- `[..., x:y]` = 所有前面维度 + 最后一维的 x 到 y-1
- `torch.split(tensor, size, dim)` = 沿 dim 切分成大小为 size 的块

---

### 10.6 多模态轨迹 Winner-Take-All（选最佳轨迹）

多模态预测中，每个 agent 输出多条候选轨迹，训练时只对**误差最小的那条**计算损失。

#### 数据维度约定

```python
trajs: (batch, agent, steps, modes, coord)  # (32, 10, 50, 6, 2) 多模态预测
gt:    (batch, agent, steps, coord)          # (32, 10, 50, 2)   真值
```

#### 完整正确流程

```python
# 1. 扩展 gt 用于广播（在 mode 维度插入）
gt = gt.unsqueeze(3)  # (32, 10, 50, 1, 2)

# 2. 计算每个点的 L2 距离（coord 维度消失）
l2_norm = torch.norm(trajs - gt, p=2, dim=-1)  # (32, 10, 50, 6)

# 3. 对 steps 维度求和（保留 mode 维度！）
l2_norm = l2_norm.sum(dim=2)  # (32, 10, 6)

# 4. 找最佳 mode
best_mode = l2_norm.argmin(dim=-1)  # (32, 10)

# 5. 提取最佳轨迹（必须加 view）
batch_index = torch.arange(batch_size).view(-1, 1)  # (32, 1)
agent_index = torch.arange(agent_num).view(1, -1)   # (1, 10)
best_traj = trajs[batch_index, agent_index, :, best_mode, :]  # (32, 10, 50, 2)
```

#### 易错点 1：`norm` 降维 vs `**2` 不降维

```python
diff = trajs - gt  # (32, 10, 50, 6, 2)

# norm 会消除被作用的维度
torch.norm(diff, p=2, dim=-1)  # (32, 10, 50, 6)  ← coord 维度消失

# **2 是逐元素操作，不降维
diff ** 2  # (32, 10, 50, 6, 2)  ← 维度不变，需手动 sum
```

等价关系：
```python
torch.norm(diff, p=2, dim=-1) == torch.sqrt((diff ** 2).sum(dim=-1))
```

**优化技巧**：WTA 只比较大小，可省略开方（`argmin` 结果不变，更快）：
```python
l2_squared = (diff ** 2).sum(dim=-1)  # 不开方
best_mode = l2_squared.sum(dim=2).argmin(dim=-1)  # 结果与开方版一致
```

#### 易错点 2：求和维度搞错，丢失 mode

```python
# ❌ 错误：对最后一维（mode）求和，mode 维度消失
l2_norm = torch.norm(trajs - gt, p=2, dim=-1).sum(dim=-1)  # (32, 10, 50)
# 此时无法区分哪个 mode 最好！

# ❌ 错误：算了但没赋值，等于白算
l2_norm.sum(dim=2)  # 没有 l2_norm = ...

# ✅ 正确：对 steps 维度（dim=2）求和，保留 mode
l2_norm = torch.norm(trajs - gt, p=2, dim=-1).sum(dim=2)  # (32, 10, 6)
```

**记忆**：要保留 mode 维度去比较，就**不能**对 mode 维度求和。对 steps 求和（dim=2），把整条轨迹的误差累计起来。

#### 易错点 3：提取轨迹必须用 arange + view，不能用 `:`

`best_mode` 形状是 `(batch, agent)`，它的值**由 batch 和 agent 共同决定**，所以索引时必须把 batch、agent 的坐标和 best_mode 配对。

```python
# ❌ 错误：用 : 会触发笛卡尔积，维度爆炸
best_traj = trajs[:, :, :, best_mode, :]  # (32, 10, 50, 32, 10, 2) 灾难

# ❌ 错误：不加 view，一维索引无法广播
batch_index = torch.arange(32)  # (32,)
agent_index = torch.arange(10)  # (10,)
trajs[batch_index, agent_index, :, best_mode, :]
# IndexError: shapes [32], [10], [32,10] 无法广播

# ✅ 正确：view 成可广播形状
batch_index = torch.arange(32).view(-1, 1)  # (32, 1)
agent_index = torch.arange(10).view(1, -1)  # (1, 10)
best_traj = trajs[batch_index, agent_index, :, best_mode, :]  # (32, 10, 50, 2)
```

**广播原理**：三个张量索引广播到统一形状 `(32, 10)`，形成一一对应的坐标：
```python
batch_index (32, 1)  ──┐
agent_index (1, 10)  ──┼── 广播 ──> (32, 10)
best_mode   (32, 10) ──┘

# 实际执行: best_traj[b,a] = trajs[b, a, :, best_mode[b,a], :]
```

#### 索引维度判断规则

| 维度 | best_mode 是否依赖它 | 操作 | 原因 |
|------|--------------------|------|------|
| batch | ✅ 随 batch 变 | `arange().view(-1,1)` | 与 best_mode 配对定位 |
| agent | ✅ 随 agent 变 | `arange().view(1,-1)` | 与 best_mode 配对定位 |
| steps | ❌ 无关 | `:` | 整段保留 |
| mode | （选择目标） | `best_mode` | 选中的索引 |
| coord | ❌ 无关 | `:` | 整段保留 |

**一句话**：要"挑特定位置"的维度用张量索引（arange/best_mode），要"整段保留"的维度用 `:`。

#### 等价方案：`torch.gather`

```python
# gather 不需要构造 arange 坐标，更适合批量场景
best_mode_idx = best_mode.view(batch_size, agent_num, 1, 1, 1).expand(
    batch_size, agent_num, steps, 1, coord_dim
)  # (32, 10, 50, 1, 2)
best_traj = torch.gather(trajs, dim=3, index=best_mode_idx).squeeze(3)  # (32, 10, 50, 2)
```

#### 常见拼写陷阱

```python
torch.arrange(n)  # ❌ 拼写错误（两个 r）
torch.arange(n)   # ✅ 正确（一个 r）
```

---

## 十一、回归损失速查表

| 损失类型 | 公式 | 适用场景 | PyTorch |
|---------|------|---------|---------|
| **MAE** | `|y - ŷ|` | 数据有离群点 | `F.l1_loss` |
| **MSE** | `(y - ŷ)²` | 数据分布均匀 | `F.mse_loss` |
| **Huber** | L2 (小误差) + L1 (大误差) | 鲁棒回归 | `F.smooth_l1_loss` |
| **Gaussian NLL** | `log(σ) + (y-μ)²/(2σ²)` | 需要不确定性估计 | `F.gaussian_nll_loss` |
| **Laplace NLL** | `log(b) + |y-μ|/b` | 轨迹预测（业界主流） | 手动实现 |

**选择建议**：
1. 快速原型 → MSE
2. 数据有噪声/离群点 → Huber 或 Laplace NLL
3. 需要不确定性输出 → NLL（Gaussian 或 Laplace）
4. 轨迹规划 → Laplace NLL（对多模态友好）
