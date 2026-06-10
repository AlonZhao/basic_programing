import torch
import torch.nn.functional as F

# 自注意力机制：每个位置去"询问"所有位置，按相关度加权汇总信息
# 流程：输入 X ──线性变换──► Q, K, V
#         scores = Q·Kᵀ / √d_k   ← 算"我和每个位置有多相关"
#         attn = softmax(scores)  ← 把相关度变成权重（每行和为1）
#         out  = attn · V         ← 按权重加权汇总
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        # 三个线性变换，权重作用在特征维度上，与 seq_len 无关，
        # 同一组权重被句子里每个 token 共享，不随句长改变。
        # nn.Linear(in, out) 内部做 x @ W.T，等价于约定 B 的行向量写法。
        self.lin_q = torch.nn.Linear(d_model,d_k)
        self.lin_k = torch.nn.Linear(d_model,d_k)
        self.lin_v = torch.nn.Linear(d_model,d_v) # 可以定制dv，最终输出维度由 d_v 决定
        self.d_k = d_k       # 存下来供 forward 缩放用
        self.d_v = d_v
    def forward(self, X):
        # X: (seq_len, d_model)，每一行是一个 token 的向量
        # 第 2 步：线性变换 → Q/K: (seq_len, d_k)，V: (seq_len, d_v)
        query = self.lin_q(X)
        key = self.lin_k(X)
        value = self.lin_v(X)
        # 第 3 步：注意力分数 scores[i][j] = 第 i 个 query 与第 j 个 key 的点积
        # 用 @ 矩阵乘法(不是 * 逐元素乘)，结果 (seq, seq)。
        # 除以 √d_k：点积随 d_k 增大方差增大，不缩放会让 softmax 过尖锐 →
        #   权重退化成"只看一个位置"，且 softmax 饱和区梯度≈0，训练学不动。
        # transpose(-2,-1) 只交换最后两维，比 .T 更安全（兼容 batch 维度）。
        score = query @ key.transpose(-2,-1) / (self.d_k)**0.5
        # 第 4 步：softmax 沿最后一维归一化，使每行权重和为 1。
        # 用 dim=-1 而非 dim=1，是为了兼容后续加 batch/head 维度时不出错。
        attention = F.softmax(score, dim = -1)
        # 第 5 步：加权汇总 value，out[i] = Σ_j attn[i][j] * value[j]，
        # 即第 i 个 token 以 attn[i] 为权重对所有 value 做加权平均；同样用 @。
        out = attention @ value
        return out
    
if __name__ == "__main__":
    seq_length = 100
    d_model = 64
    d_k = 256
    d_v = 256
    batch_size = 32

    self_attan = SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v)

    # 情况一：无 batch，X 形状 (seq_len, d_model)
    X = torch.randn(seq_length, d_model)
    out = self_attan(X)
    print("无 batch  输入:", tuple(X.shape), "→ 输出:", tuple(out.shape))  # (100, 256)

    # 情况二：带 batch，X 形状 (batch, seq_len, d_model)
    # forward 不用改也能跑：transpose(-2,-1) 只交换最后两维，@ 在 batch 维自动广播
    Xb = torch.randn(batch_size, seq_length, d_model)
    outb = self_attan(Xb)
    print("带 batch  输入:", tuple(Xb.shape), "→ 输出:", tuple(outb.shape))  # (32, 100, 256)
