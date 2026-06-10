import torch
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, head_num):
        super().__init__()
        assert d_model % head_num == 0, "d_model 必须能被 head_num 整除"
        self.d_model = d_model
        self.d_k = d_model // head_num
        self.d_v = d_model // head_num
        self.head_num = head_num
        self.lin_q = torch.nn.Linear(d_model, d_model)
        self.lin_k = torch.nn.Linear(d_model, d_model)
        self.lin_v = torch.nn.Linear(d_model, d_model)
        # 多头新增
        self.lin_out = torch.nn.Linear(d_model, d_model)

    def forward(self, X, mask):
        # X (bs, seq, dm)
        bs = X.shape[0]
        seq_len = X.shape[1]
        # 拆多头：先 view(bs, seq, head_num, d_k) 再 transpose 交换 head 和 seq
        # (bs, seq, dm) -> (bs, seq, head_num, d_k) -> (bs, head_num, seq, d_k)
        query = self.lin_q(X).view(bs, seq_len, self.head_num, self.d_k).transpose(1, 2)
        key   = self.lin_k(X).view(bs, seq_len, self.head_num, self.d_k).transpose(1, 2)
        value = self.lin_v(X).view(bs, seq_len, self.head_num, self.d_v).transpose(1, 2)
        # qkv (bs, head_num, seq, d_k)
        # 多头新增
        # mask (bs, seq, seq) -> (bs, 1, seq, seq) 后续广播到所有头
        mask = mask.unsqueeze(1)
        # score (bs, head_num, seq, seq)
        score = query @ key.transpose(-2, -1) / (self.d_k) ** 0.5
        score = score.masked_fill(~mask, float('-inf'))
        atten = F.softmax(score, dim=-1)
        out = atten @ value  # (bs, head_num, seq, d_v)
        # 拼回多头：transpose 回去再 view 合并 head 维
        # (bs, head_num, seq, d_v) -> (bs, seq, head_num, d_v) -> (bs, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        out = self.lin_out(out)
        return out


if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_model = 64
    head_num = 8
    mha = MultiHeadAttention(d_model=d_model, head_num=head_num)
    X = torch.randn(batch_size, seq_len, d_model)
    mask = torch.randint(0, 2, (batch_size, seq_len, seq_len)).bool()
    out = mha(X, mask)
    print("输入形状:", X.shape)
    print("输出形状:", out.shape)  # 期望 (32, 10, 64)
