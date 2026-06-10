# 注意力机制面试要点整理

## 一、基础概念

### 1.1 自注意力 vs 交叉注意力

| 类型 | Q 来源 | K 来源 | V 来源 | 典型场景 |
|------|--------|--------|--------|----------|
| 自注意力 | X | X | X | Transformer Encoder、BERT |
| 交叉注意力 | X | Y | Y | Transformer Decoder（关注 Encoder 输出）|

**核心区别**：
- 自注意力：序列内部"自己和自己"算关注度
- 交叉注意力：一个序列去"查询"另一个序列的信息

### 1.2 注意力计算流程

```
输入 X ──线性变换──► Q, K, V
                     │
    scores = Q @ K^T / √d_k   ← 算相关度
                     │
    attn = softmax(scores)    ← 变成权重（每行和为1）
                     │
    out = attn @ V            ← 加权汇总
```

---

## 二、为什么用 Conv1d 做映射？

### 问题背景
```python
self.query_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
```

### 回答要点

**`Conv1d(kernel_size=1)` 等价于逐位置的全连接层**，和 `nn.Linear` 做的事情一样，只是数据排布不同：

| 操作 | 输入形状 | 权重形状 | 输出形状 |
|------|----------|----------|----------|
| `Linear(in, out)` | `(batch, seq, in)` | `(in, out)` | `(batch, seq, out)` |
| `Conv1d(in, out, 1)` | `(batch, in, seq)` | `(out, in, 1)` | `(batch, out, seq)` |

**为什么有人用 Conv1d？**

1. **输入格式匹配**：当输入是 `(batch, channels, seq)` 格式时（点云、图网络、信号处理），用 Conv1d 可以直接处理，不用转置
2. **代码习惯/历史遗留**：PointNet、图神经网络等领域习惯用 Conv1d
3. **本质无区别**：`kernel_size=1` 不做空间聚合，每个位置独立做线性变换

**面试标准答案**：
> "Conv1d(kernel_size=1) 等价于逐位置的全连接层，用它是为了适配 (batch, channel, seq) 的输入格式，没有利用卷积的局部性，纯粹是线性映射。"

### 维度变化详解

#### Linear 版本的维度流动

```python
# 输入：(batch, seq, d_model)
X = torch.randn(2, 5, 64)  # 2个样本，5个token，每个64维

# 线性变换
self.lin_q = nn.Linear(64, 128)  # d_model=64 → d_k=128
Q = self.lin_q(X)  # (2, 5, 128)

# 关键：Linear 作用在最后一维，前面的维度(batch, seq)保持不变
```

**维度映射**：
- 输入：`(batch, seq, d_model)` 
- 权重：`(d_model, d_k)` 
- 输出：`(batch, seq, d_k)` 
- **作用维度**：最后一维（特征维）

#### Conv1d 版本的维度流动

```python
# 输入：(batch, d_model, seq) ← 注意维度顺序不同
x = torch.randn(2, 64, 5)  # 2个样本，64个通道，5个位置

# 1x1 卷积
self.query_linear = nn.Conv1d(64, 128, kernel_size=1)
Q_raw = self.query_linear(x)  # (2, 128, 5)

# 问题：Q_raw 是 (batch, d_k, seq)，但注意力计算需要 (batch, seq, d_k)
Q = Q_raw.transpose(1, 2)  # (2, 5, 128) ← 必须转置
```

**维度映射**：
- 输入：`(batch, d_model, seq)` 
- 权重：`(d_k, d_model, 1)` 
- 输出：`(batch, d_k, seq)` 
- **作用维度**：中间维度（通道维）

#### 为什么需要两次转置？

完整的 Conv1d 版本注意力流程：

```python
# 1. Conv1d 输入输出都是 channel-first
x: (b, c, n)  # 输入
Q = self.query_linear(x)  # (b, hidden_dim, n)
K = self.key_linear(x)    # (b, hidden_dim, n)
V = self.value_linear(x)  # (b, hidden_dim, n)

# 2. 第一次转置：为了做注意力计算
Q = Q.transpose(1, 2)  # (b, n, hidden_dim)
K = K.transpose(1, 2)  # (b, n, hidden_dim)
V = V.transpose(1, 2)  # (b, n, hidden_dim)

# 3. 注意力计算（需要 sequence-first 格式）
scores = Q @ K.transpose(-2, -1)  # (b, n, n)
attn = F.softmax(scores, dim=-1)   # (b, n, n)
out = attn @ V                     # (b, n, hidden_dim)

# 4. 第二次转置：转回 channel-first 格式输出
out = out.transpose(1, 2)  # (b, hidden_dim, n)
```

**关键点**：
- Conv1d 的输入输出约定是 `(batch, channel, length)`
- 注意力的矩阵乘法 `Q @ K.T` 需要 `(batch, seq, dim)` 格式（seq 在中间）
- 所以必须 **进去转一次、出来转一次**

#### 两种方式的等价性证明

假设输入是同一批数据，只是维度排列不同：

```python
# Linear 版本
X_linear = torch.randn(2, 5, 64)  # (batch, seq, d_model)
W_linear = torch.randn(64, 128)   # (d_model, d_k)
Q_linear = X_linear @ W_linear    # (2, 5, 128)

# Conv1d 版本
X_conv = X_linear.transpose(1, 2)  # (2, 64, 5) = (batch, d_model, seq)
W_conv = W_linear.T.unsqueeze(-1)  # (128, 64, 1) = (out, in, kernel)
Q_conv_raw = F.conv1d(X_conv, W_conv)  # (2, 128, 5)
Q_conv = Q_conv_raw.transpose(1, 2)     # (2, 5, 128)

# 结果相等（权重相同的情况下）
assert torch.allclose(Q_linear, Q_conv)
```

**本质**：两者做的都是 `X @ W`，只是因为输入的维度排列不同，所以用不同的算子。

#### 实际使用建议

| 场景 | 推荐方式 | 原因 |
|------|----------|------|
| NLP、Transformer | `nn.Linear` | 输入自然是 `(batch, seq, dim)`，不用转置 |
| 点云（PointNet系列） | `nn.Conv1d` | 输入是 `(batch, channel, npoints)`，直接匹配 |
| 图神经网络 | `nn.Conv1d` | 节点特征常存为 `(batch, feature, nodes)` |
| 时序信号处理 | `nn.Conv1d` | 信号格式 `(batch, channel, time)` |

**记忆要点**：
- **Linear 吃最后一维，Conv1d 吃中间一维**
- Conv1d 做注意力需要**两次转置**（进出各一次）
- 选哪个看输入格式，本质计算量完全相同

#### 常见面试追问

**Q: "Conv1d 需要转置两次，不是更慢吗？"**

**A**: 转置操作在 PyTorch 里是 **view 操作**（只改元数据，不移动数据），几乎零开销。真正的计算量在矩阵乘法上，两种方式的乘法次数完全一样。性能差异主要看：
1. 内存布局的连续性（contiguous）
2. 算子的底层优化（cuBLAS vs cuDNN）
3. 实际测试中差异通常<5%，可忽略

**Q: "为什么不统一用一种格式？"**

**A**: 历史原因 + 领域习惯：
- NLP 领域先用 Linear + `(batch, seq, dim)`
- CV 领域习惯 `(batch, channel, height, width)`，延伸到点云/图就成了 `(batch, channel, nodes)`
- 跨领域代码融合时就会出现两种风格混用

**Q: "transpose(-2, -1) 和 transpose(1, 2) 有什么区别？"**

**A**: 
- `transpose(1, 2)` 明确指定交换维度 1 和 2，**只适用于 3D 张量**
- `transpose(-2, -1)` 交换倒数第二和倒数第一维，**适用于任意维度**
  - 3D: `(b, n, c)` → `(b, c, n)` 
  - 4D: `(b, h, n, c)` → `(b, h, c, n)` ← 多头注意力的情况
- 写代码时优先用 `-2, -1`，扩展性更好

---

## 三、Mask 有什么用？

### 3.1 三种典型场景

#### (a) Padding Mask（处理变长序列）

真实句子长短不一，batch 里会 pad 到统一长度。Mask 掉 pad 位置，防止注意力分散到无意义的 padding token 上。

```
原始序列：[我, 爱, 你]
padding后：[我, 爱, 你, <pad>, <pad>]
mask: [1, 1, 1, 0, 0]  ← 0 的位置会被屏蔽
```

#### (b) Causal Mask / Look-ahead Mask（自回归生成）

语言模型生成时，第 i 个位置**只能看到前 i 个 token**，不能"偷看"未来。Mask 成下三角矩阵：

```
scores:        mask (1=可见, 0=屏蔽):
[[s00 s01 s02]    [[1  0  0]
 [s10 s11 s12]     [1  1  0]      ← 第1行只能看到位置0,1
 [s20 s21 s22]]    [1  1  1]]
```

```python
# 生成 Causal Mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
```

#### (c) 自定义 Mask（图结构/无效边）

图神经网络里，只有有边的节点对之间才计算注意力，用邻接矩阵作为 mask。

### 3.2 实现方式

```python
# 在 softmax 之前，把 mask=False 的位置替换成 -inf
scores = scores.masked_fill(~mask, float('-inf'))
attn = F.softmax(scores, dim=-1)  # mask 掉的位置权重自动≈0
```

**为什么用 `-inf`？**
- Softmax 前置 0 依然会分到一定权重
- 置 `-inf` 才能让 `exp(-inf) = 0`，权重彻底归零

**面试标准答案**：
> "Mask 用于屏蔽无效位置（padding、未来 token、无边节点），实现方式是在 softmax 前把对应位置置为 -inf，使权重为 0。"

### 3.3 masked_fill 的语义和取反操作

#### masked_fill 的基本逻辑

```python
tensor.masked_fill(mask, value)
```

**`mask` 为 `True` 的位置会被填充成 `value`。**

示例：
```python
scores = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

mask = torch.tensor([[True, False, True],
                     [False, True, False]])

result = scores.masked_fill(mask, -999)
# 输出：
# tensor([[-999.,    2., -999.],
#         [   4., -999.,    6.]])
```

**`True` 的位置被填成 `-999`，`False` 的位置保持原值。**

#### 为什么代码里要 `~mask` 取反？

在我们的注意力代码中：

```python
mask = torch.randint(0, 2, (b, n, n)).bool()
# mask 语义：True=保留该位置，False=屏蔽该位置

scores = scores.masked_fill(~mask, float('-inf'))
#                           ^^^^^^ 取反！
```

**逻辑推导**：
1. `mask` 里 `True` 表示"这个位置应该**保留**"（有效位置）
2. 我们要把**屏蔽的位置**（即 `mask=False` 的位置）填成 `-inf`
3. `~mask` 取反后，原来的 `False`（需要屏蔽）变成 `True`
4. `masked_fill(~mask, -inf)` 就是"把需要屏蔽的位置（现在是 True）填成 `-inf`"

#### 两种写法对比

**写法 1：mask 表示"保留"（推荐，更符合直觉）**

```python
# mask: True=保留，False=屏蔽
mask = torch.tensor([[True, True, False],   # 前两个位置保留，第三个屏蔽
                     [True, False, False]])

scores = scores.masked_fill(~mask, float('-inf'))  # 取反，填充屏蔽位置
attn = F.softmax(scores, dim=-1)
# 结果：第三列权重≈0（因为被填成了-inf）
```

**写法 2：mask 表示"屏蔽"（不推荐，反直觉）**

```python
# mask: True=屏蔽，False=保留
mask = torch.tensor([[False, False, True],   # 第三个位置屏蔽
                     [False, True, True]])

scores = scores.masked_fill(mask, float('-inf'))  # 直接填充屏蔽位置
attn = F.softmax(scores, dim=-1)
```

**为什么写法 1 更常见？**
1. **更符合人类直觉**："True=有效/保留"比"True=屏蔽"更自然
2. **与常用数据格式匹配**：
   - Padding mask：`valid_mask[i] = (i < actual_length)` → True 表示有效
   - 邻接矩阵：`adj[i][j] = 1` 表示有边（有效连接）
   - 所以拿到的 mask 通常就是"True=保留"的语义

#### 记忆口诀

**`masked_fill(condition, value)`：`condition` 为 True 的地方填 `value`。**

- 如果你的 mask 是"True=保留"的语义 → 用 `~mask` 取反再填
- 如果你的 mask 是"True=屏蔽"的语义 → 直接填

**推荐统一用"True=保留"，代码里写 `masked_fill(~mask, -inf)`，避免混淆。**

---

## 四、关键追问题

### Q1: "为什么是 `-inf` 而不是 0？"

**A**: Softmax 前置 0 依然会分到一定权重（因为 softmax 是归一化的，所有位置的指数和作为分母）。置 `-inf` 才能让 `exp(-inf) = 0`，权重彻底归零。

### Q2: "如果 mask 全是 False 会怎样？"

**A**: 整行都是 `-inf` → softmax 输出 `NaN` → 实践中需要检测这种情况，或保证至少有一个 True。可以在数据预处理阶段检查，或用 `torch.nan_to_num` 处理。

### Q3: "Conv1d 和 Linear 哪个快？"

**A**: 理论计算量一样（都是矩阵乘法）。实际速度取决于：
- 框架底层优化（cuBLAS、cuDNN）
- 内存布局（连续性）
- 硬件特性

通常 Linear 对 NLP 更友好，Conv1d 对信号/点云场景更自然。性能差异不大，主要看代码习惯和输入格式。

### Q4: "为什么要除以 √d_k？"

**A**: 点积的方差随维度 `d_k` 线性增长。不除会导致：
1. scores 数值过大 → softmax 变得极端尖锐（几乎全部权重压到一个位置）
2. 注意力退化成"只看一个位置"，失去加权汇总的意义
3. Softmax 在饱和区梯度≈0，训练学不动

除以 `√d_k` 将方差缩回到 1 量级，让 softmax 输出平滑、梯度健康。

### Q5: "为什么 mask 形状是 (b, n, n) 而不是 (b, c, c)？"

**A**: Mask 控制的是"哪个 token 能看哪个 token"，和特征维度 `c` 无关。

注意力分数：
```python
scores = Q @ K.T → (b, n, n)  # 第 i 个 token 对第 j 个 token 的分数
```

`scores[i][j]` 表示"第 i 个 token 对第 j 个 token 的关注度"，是 token 之间的关系，所以 mask 也是 `(n, n)`。

### Q6: "d_k 和 d_v 的区别？"

**A**: 
- `d_k` 决定**"如何算相关度"的空间维度**。Q 和 K 在 `d_k` 维空间做点积，产出标量分数，`d_k` 不出现在输出里。
- `d_v` 决定**"每个 token 携带多少信息"以及最终输出的维度**。`out = attn @ V`，V 是 `(seq, d_v)`，所以 `out` 是 `(seq, d_v)`。

类比：`d_k` 是"用多精细的尺子量相似度"，`d_v` 是"实际搬运多少内容"。两者独立。

---

## 五、数据格式对比

### 5.1 两种常见格式

| 格式 | 形状 | 常用场景 | 映射层 |
|------|------|----------|--------|
| sequence-first | `(batch, seq, dim)` | NLP、Transformer | `nn.Linear` |
| channel-first | `(batch, dim, seq)` | 点云、图网络、PointNet | `nn.Conv1d` |

### 5.2 维度含义

以测试代码 `b, c, n = 2, 3, 5` 为例：
- `b` = batch size（批次大小）
- `c` = 特征维度（相当于 `d_model`，每个 token 的向量维度）
- `n` = 序列长度（相当于 `seq_len`，token 数量）

所以 `x: (b, c, n)` = `(batch, d_model, seq_len)` 是 **channel-first** 格式。

### 5.3 格式转换

在 Conv1d 版本的自注意力实现中，需要两次转置：

```python
# Conv1d 输出是 (b, c, n)
Q = self.query_linear(x)  # (b, hidden_dim, n)

# 注意力计算需要 (b, n, hidden_dim)
Q = Q.transpose(1, 2)      # (b, n, hidden_dim)

# 计算注意力...
out = attn @ V             # (b, n, hidden_dim)

# 转回 channel-first
out = out.transpose(1, 2)  # (b, hidden_dim, n)
```

---

## 六、完整实现对比

### 6.1 标准 Linear 版本（sequence-first）

```python
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.lin_q = torch.nn.Linear(d_model, d_k)
        self.lin_k = torch.nn.Linear(d_model, d_k)
        self.lin_v = torch.nn.Linear(d_model, d_v)
        self.d_k = d_k

    def forward(self, X):
        # X: (seq, d_model) or (batch, seq, d_model)
        query = self.lin_q(X)
        key = self.lin_k(X)
        value = self.lin_v(X)
        score = query @ key.transpose(-2, -1) / (self.d_k ** 0.5)
        attn = F.softmax(score, dim=-1)
        out = attn @ value
        return out
```

### 6.2 Conv1d 版本 + Mask（channel-first）

```python
class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.key_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.value_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask):
        # x: (b, c, n)，mask: (b, n, n)
        Q = self.query_linear(x).transpose(1, 2)  # (b, n, hidden_dim)
        K = self.key_linear(x).transpose(1, 2)
        V = self.value_linear(x).transpose(1, 2)
        
        scores = Q @ K.transpose(-2, -1) / (self.hidden_dim ** 0.5)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        
        return out.transpose(1, 2)  # 转回 (b, hidden_dim, n)
```

---

## 七、多头注意力（Multi-Head Attention）

### 7.1 核心思想

把 Q/K/V 拆成多个"头"，每个头在低维子空间里独立算注意力，最后拼接。
好处：让模型能从多个角度（多个子空间）同时关注信息，且总计算量不增加。

### 7.2 参数定义：d_k 是拆分前还是拆分后的维度？

**这是最容易混淆的点。存在两种惯例：**

| 惯例 | d_k 含义 | 关系 |
|------|----------|------|
| 惯例 A | d_k = 总维度（拆分前），通常等于 d_model | d_k_per_head = d_k // num_heads |
| 惯例 B ⭐ | d_k = 每个头的维度（拆分后） | d_model = num_heads × d_k |

**标准答案（Transformer 原论文用法）：`d_k` 一般指拆分后每个头的维度（惯例 B）。**

```python
d_model = 512
num_heads = 8
d_k = d_model // num_heads  # = 64，每个头的维度
# 验证：d_model == num_heads * d_k → 512 == 8 * 64 ✓
```

所以 `query` 拆分后的实际维度是 `(batch, num_heads, seq, d_k)`，这里 `d_k = 64` 是**每个头**的维度。

**约束**：`d_model` 必须能被 `num_heads` 整除，否则没法均匀拆分。

```python
assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
```

### 7.3 维度变化全流程

```
输入: (batch, seq, d_model)
  ↓ Linear (d_model → d_model)
(batch, seq, d_model)
  ↓ view: (batch, seq, num_heads, d_k)
  ↓ transpose(1,2): (batch, num_heads, seq, d_k)   ← 拆成多头
  ↓ Q @ K.T
(batch, num_heads, seq, seq)                        ← 每个头的注意力分数
  ↓ softmax
(batch, num_heads, seq, seq)                        ← 每个头的注意力权重
  ↓ @ V
(batch, num_heads, seq, d_k)                        ← 每个头的输出
  ↓ transpose(1,2): (batch, seq, num_heads, d_k)
  ↓ view: (batch, seq, d_model)                     ← 拼接回原维度
  ↓ Linear (d_model → d_model)
(batch, seq, d_model)                               ← 最终输出
```

**关键：拆分用 `view + transpose`，拼接用 `transpose + view`，是互逆操作。**

### 7.4 拆分后 Q·K 在哪两个维度相乘？

拆分后形状：
```python
Q: (batch, num_heads, seq, d_k)
K: (batch, num_heads, seq, d_k)
```

矩阵乘法 `Q @ K.transpose(-2, -1)`：
```python
K.transpose(-2, -1):  (batch, num_heads, d_k, seq)  # 只交换最后两维

scores = Q @ K.transpose(-2, -1)
# (batch, num_heads, seq, d_k) @ (batch, num_heads, d_k, seq)
#                    ^^^^  ^^^                    ^^^  ^^^^
# → (batch, num_heads, seq, seq)
```

**乘法发生在最后两维 `seq` 和 `d_k` 上：**
- `d_k` 维被内积消掉
- 得到 `(seq, seq)` 注意力分数矩阵
- `batch` 和 `num_heads` 维度自动广播，相当于每个头独立计算

分解理解（对 batch=0, head=0 这一小块）：
```python
scores[0, 0] = Q[0, 0] @ K[0, 0].T
#              (seq, d_k) @ (d_k, seq) → (seq, seq)
```

### 7.5 关于 @ 运算符的核心规则

**PyTorch 的 `@`（矩阵乘法）总是在最后两个维度进行，前面的维度做批量广播。**

```python
A: (..., m, k)
B: (..., k, n)
C = A @ B: (..., m, n)
```

- **最后两维**参与矩阵乘法：`(m, k) @ (k, n) → (m, n)`
- **前面的维度**（`...` 部分）必须兼容广播，结果保留这些维度

例子：
```python
# 2D：标准矩阵乘法
(3, 4) @ (4, 5) → (3, 5)

# 3D：批量矩阵乘法（对每个 batch 独立）
(10, 3, 4) @ (10, 4, 5) → (10, 3, 5)

# 4D：多头注意力（对每个 batch+head 独立）
(2, 8, 100, 64) @ (2, 8, 64, 100) → (2, 8, 100, 100)

# 广播：A 在第一维被广播到 5
(3, 4) @ (5, 4, 2) → (5, 3, 2)
```

**为什么这样设计？** 让前面的维度自动批量处理，不用写循环：
```python
# 多头注意力一行搞定，不用嵌套循环
scores = Q @ K.transpose(-2, -1)  # (batch, heads, seq, seq)

# 等价于（但快得多）
for b in range(batch):
    for h in range(heads):
        scores[b, h] = Q[b, h] @ K[b, h].T
```

**常见错误**：
- 内积维度不匹配：`(*, k) @ (k, *)` 中间的 k 必须相等
- 批量维度不兼容：前面的维度必须相等或其中一个为 1（可广播）

### 7.6 完整实现

```python
import torch
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        self.lin_q = torch.nn.Linear(d_model, d_model)
        self.lin_k = torch.nn.Linear(d_model, d_model)
        self.lin_v = torch.nn.Linear(d_model, d_model)
        self.lin_out = torch.nn.Linear(d_model, d_model)  # 拼接后的输出变换

    def forward(self, X, mask=None):
        batch_size, seq_len, _ = X.shape

        # 1. 线性变换 (batch, seq, d_model)
        Q = self.lin_q(X)
        K = self.lin_k(X)
        V = self.lin_v(X)

        # 2. 拆成多头 (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 每个头独立算注意力 (batch, num_heads, seq, seq)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        # 4. 应用 mask（扩展成 (batch, 1, seq, seq) 广播到所有头）
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # 5. softmax + 加权汇总
        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (batch, num_heads, seq, d_k)

        # 6. 拼接多头 (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 7. 最后的线性变换
        out = self.lin_out(out)
        return out
```

### 7.7 实现细节注意点

1. **`view` 拆分前后维度顺序**：先 `view(b, seq, heads, d_k)` 再 `transpose(1,2)`，
   不能直接 `view(b, heads, seq, d_k)`，那样会打乱数据（元素不是按头连续排列的）。

2. **拼接时必须 `.contiguous()`**：`transpose` 后张量在内存里不连续，
   直接 `view` 会报错，要先 `.contiguous()` 让内存连续。

3. **mask 要扩展头维度**：mask 原始是 `(batch, seq, seq)`，
   用 `mask.unsqueeze(1)` 变成 `(batch, 1, seq, seq)`，1 这一维会广播到所有头。

4. **两次 transpose 用 `(1, 2)`**：因为是 4D 张量交换第 1、2 维（heads 和 seq），
   也可写成 `transpose(-3, -2)` 兼容性更好。

---

## 八、总结：面试核心要点

1. **自注意力 vs 交叉注意力**：区别在 Q、K、V 的来源
2. **Conv1d(kernel_size=1) = Linear**：只是适配不同的输入格式
3. **Mask 的作用**：屏蔽无效位置（padding、未来、无边），用 `-inf` 实现
4. **√d_k 缩放**：防止 softmax 饱和，保证梯度健康
5. **数据格式转换**：channel-first 和 sequence-first 需要 transpose
6. **维度理解**：mask 是 `(n, n)` 因为控制的是 token 间关系，与特征维度无关
7. **多头注意力**：d_k 是每个头的维度（d_model // num_heads），拆分用 view+transpose
8. **@ 运算符**：总是在最后两维做矩阵乘法，前面维度批量广播

**记忆口诀**：
- 注意力 = 查询（Q）+ 键（K）+ 值（V）
- 步骤 = 算分数 → mask 屏蔽 → softmax 归一化 → 加权汇总
- 多头 = 拆（view+transpose）→ 各自算 → 拼（transpose+view）
- @ 乘法 = 只看最后两维，前面批量循环
- Conv1d 只是换装，本质还是线性映射
