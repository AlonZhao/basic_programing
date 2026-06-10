import torch
import torch.nn.functional as F
import numpy as np

# 补全单头注意力计算代码
class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.key_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.value_linear = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask):
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        # (b, c, n) -> (b, hd, n)
        query = query.transpose(-2,-1)
        key = key.transpose(-2,-1)
        value = value.transpose(-2,-1)
        # qkv (b, n, hd)
        score = query @ key.transpose(-2,-1) / (self.hidden_dim ** 0.5)
        # score (b, n, n) 
        score = score.masked_fill(~mask, float('-inf'))
        atten = F.softmax(score, dim = -1)
        out = atten @ value
        return out.transpose(-2,-1)

        # x: (b, c, n)
        # mask: (b, n, n)
        # qkv: (b, dk, n)


if __name__ == '__main__':
    # 测试SelfAttention
    b, c, n = 2, 3, 5
    x = torch.rand(b, c, n)
    mask = torch.randint(0, 2, (b, n, n)).bool()
    sa = SelfAttention(c, c)
    y = sa(x, mask)
    print(y.shape)