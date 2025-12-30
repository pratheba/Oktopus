import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from flash_attn import flash_attn_kvpacked_func

from pos_encoding import PosEncoding, PosEncodingMLP
from components import Attention, FeedForward, PreNorm


class SurfaceEncoder(nn.Module):
    def __init__(self, arg):
        self.num_latents = arg['num_latents']

        if arg['learn_posenc']:
            self.pos_enc = PosEncodingMLP(num_freqs = arg['num_pos_encoding'], dim=arg['query_dim'])
            self.query_dim = arg['query_dim']
        else:
            self.pos_enc = PosEncoding(arg['num_pos_encoding'])
            self.query_dim = 2*arg['num_pos_encoding']+3

        self.latents = nn.Embedding(self.num_latents, self.query_dim)

        self.context_dim = arg['context_dim']
        self.heads = arg['heads']

        self.cross_attn_blocks = nn.ModuleList([
            PreNorm(self.query_dim, Attention(self.query_dim, self.context_dim, heads = self.heads, dim_head = self.head_dim)),
            PreNorm(self.query_dim, FeedForward(self.query_dim))
            ])


    def forward(self, x):
        local_samples = local_samples
        B, N, _ = x.shape
        x = repeat(self.latents.weight, 'n d -> b n d', b = B)

        fourier_embedding = self.pos_enc(local_samples)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context = fourier_embedding, mask = None) + x
        x = cross_ff(x) + x

        return x


