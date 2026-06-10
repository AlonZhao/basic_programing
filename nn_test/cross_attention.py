import torch
import torch.nn.functional as F

class CrossAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.lin_q = torch.nn.Linear(d_model, d_k)
        self.lin_k = torch.nn.Linear(d_model, d_k)
        self.lin_v = torch.nn.Linear(d_model,d_v)
        self.d_k = d_k

    def forward(self, X, Y):
        query = self.lin_q(X)
        key   = self.lin_k(Y)
        value = self.lin_v(Y)
        score = query @ key.transpose(-2,-1) / (self.d_k)**0.5
        atten = F.softmax(score, dim=-1)
        out = atten @ value
        return out


if __name__ == "__main__":
    seq_q = 5
    seq_kv = 10
    d_model = 64
    d_k = 128
    d_v = 128

    cross_attn = CrossAttention(d_model, d_k, d_v)

    X = torch.randn(seq_q, d_model)
    Y = torch.randn(seq_kv, d_model)
    out = cross_attn(X, Y) ## q 和 kv对应的不同源头
    print("无 batch")
    print(f"  X: {tuple(X.shape)} (query 方)")
    print(f"  Y: {tuple(Y.shape)} (被查询方)")
    print(f"  out: {tuple(out.shape)} → seq_q={seq_q}, d_v={d_v} ✓")

    batch = 4
    Xb = torch.randn(batch, seq_q, d_model)
    Yb = torch.randn(batch, seq_kv, d_model)
    outb = cross_attn(Xb, Yb)
    print("\n带 batch")
    print(f"  Xb: {tuple(Xb.shape)}")
    print(f"  Yb: {tuple(Yb.shape)}")
    print(f"  outb: {tuple(outb.shape)} → (batch, seq_q, d_v) ✓")


