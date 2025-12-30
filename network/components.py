import numpy as np
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads = 8, head_dim = 64):
        inner_dim = heads * head_dim
        self.query_dim = query_dim

        context_dim = default(context_dim, query_dim)

        self.scale = head_dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, 2*inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

    #####################
    def forward(self, x, context=None, mask=None, window_size=1):
        '''
           x = learned latent weights
           context = point embeddings
        '''
        h = self.heads

        B, N, _ = x.shape

        #assert D == self.query_dim

        to_q = self.to_q(x)
        context = default(context, x)
        to_kv = self.to_kv(context)

        q = rearrange('b n (h d) -> b n h d', h = h)
        kv = rearrange('b m (p h d) -> b m p h d', h = h, p = 2)

        out = flash_attn_kvpacked_func(q.bfloat16(), kv.bfloat16(), window_size=(window_size, window_size))
        out = out.to(x.dtype)

        out = rearrange('b n h d -> b n (h d)')
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GatedGELU(),
                nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x

class GatedGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
