import numpy as np
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat

#from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
from timm.models.layers import DropPath
from model_utils import default, exists

class Attention(nn.Module):
    def __init__(self, queries_dim, context_dim=None, heads = 8, head_dim = 64, drop_path_rate=0.0):
        super().__init__()

        inner_dim = heads * head_dim
        self.query_dim = queries_dim

        context_dim = default(context_dim, queries_dim)

        self.scale = head_dim ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(queries_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, 2*inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, queries_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    #####################
    def forward(self, x, context=None, mask=None, window_size=1):
        '''
           x = learned latent weights
           context = point embeddings
        '''
        h = self.heads

        #B, N, _ = x.shape
        #print("h = ", h)
        #print("x shape = ", x.shape)

        #assert D == self.query_dim

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2,dim=-1)
        #print(q.shape)
        #print(k.shape)
        #print(v.shape)

        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        

        #to_q = self.to_q(x)
        #to_kv = self.to_kv(context)
        #q = rearrange(to_q, 'b n (h d) -> b n h d', h = h)
        #kv = rearrange(to_kv, 'b m (p h d) -> b m p h d', h = h, p = 2)
        #out = flash_attn_varlen_kvpacked_func(q.bfloat16(), kv.bfloat16(), window_size=(window_size, window_size))
        #out = out.to(x.dtype)
        #out = rearrange(out,'b n h d -> b n (h d)')

        out = rearrange(out,'(b h) n d -> b n (h d)', h = h)
        #print("out shape = ", out.shape, flush=True)
        out = self.to_out(out)
        out = self.drop_path(out)
        out = out.to(x.dtype)

        return out

class GatedGELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GatedGELU(),
                nn.Linear(dim * mult, dim))

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        x = self.fn(x, **kwargs)
        return x


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.device = self.mean.device

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        if other is None:
            return 0.5 * torch.mean(torch.pow(self.mean,2) + self.var - 1.0 - self.logvar, dim=[1,2])
        else:
            return 0.5 * torch.mean(torch.pow(self.mean - other.mean, 2)/other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

