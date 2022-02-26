import torch
import torch.nn as nn
import torch.nn.functional as F


b = 20
t = 30
k = 10

x = torch.ones((b,t,k))

raw_weights = torch.bmm(x, x.transpose(1,2))
weights = F.softmax(raw_weights, dim=2)

y = torch.bmm(weights, x)

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_keys = nn.Linear(k, k*heads, bias=False)
        self.to_queries = nn.Linear(k, k*heads, bias=False)
        self.to_values = nn.Linear(k, k*heads, bias=False)

        self.unify_heads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        return self.unify_heads(out)
