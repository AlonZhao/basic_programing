# 基于自行车模型的 iLQR 优化问题

## 题目描述

待补充

## 基本思路

iLQR（iterative Linear Quadratic Regulator）是一种用于非线性系统轨迹优化的算法，通过反复线性化系统动力学并求解 LQR 子问题来逼近最优解。

### 问题建模

状态量：x = (x, y, θ, v)（自行车模型）
控制量：u = (δ, a)

目标：最小化代价函数

J = Σ_{k=0}^{N-1} l(x_k, u_k) + l_f(x_N)

其中：
- l(x, u) = (x - x_ref)ᵀ Q (x - x_ref) + uᵀ R u（运行代价）
- l_f(x) = (x - x_ref)ᵀ Q_f (x - x_ref)（终端代价）

### 算法流程

1. **前向模拟**：用初始控制序列 u₀...u_{N-1} 在自行车模型上 rollout 得到标称轨迹
2. **反向传播**：沿标称轨迹线性化动力学，二次近似代价函数，递推求解反馈增益 K_k 和前馈修正 k_k
3. **前向更新**：用线搜索（line search）沿修正方向更新轨迹
4. **收敛判断**：代价变化量小于阈值则停止，否则回到步骤 2

### 线性化（自行车模型）

在标称点 (x̄_k, ū_k) 处：
- A_k = ∂f/∂x |_{x̄_k, ū_k}（状态 Jacobian）
- B_k = ∂f/∂u |_{x̄_k, ū_k}（控制 Jacobian）

离散化：x_{k+1} ≈ f(x̄_k, ū_k) + A_k (x_k - x̄_k) + B_k (u_k - ū_k)

### Backward Pass 递推

从终端开始：
- V_N = Q_f, v_N = Q_f (x̄_N - x_ref)

递推 k = N-1 ... 0：
- Q_xx = l_xx + Aᵀ V_{k+1} A
- Q_uu = l_uu + Bᵀ V_{k+1} B
- Q_ux = Bᵀ V_{k+1} A
- Q_u = l_u + Bᵀ v_{k+1}
- K_k = -Q_uu⁻¹ Q_ux
- k_k = -Q_uu⁻¹ Q_u
- V_k = Q_xx + Q_ux^T K_k
- v_k = Q_u + Q_uu k_k（简化表示）

### Forward Pass（带 line search）

α ∈ (0, 1]，逐步缩小直到代价下降：
- u_k^{new} = ū_k + α * k_k + K_k (x_k^{new} - x̄_k)

## 关键参数

- Q, R, Q_f：权重矩阵，调节跟踪精度与控制平滑的权衡
- 迭代次数上限
- 收敛阈值 ε
- Line search 缩减系数

## 代码实现

```cpp
// 待补充
```

## 易错点

待补充
