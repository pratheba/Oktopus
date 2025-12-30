import numpy as np
import torch
import torch.nn as nn

from components import Attention, FeedForward, PreNorm
from pos_encoding import PosEncoderMLP 

class ShapeDecoder(nn.Module):
    def __init__(self, arg):
        super(ShapeDecoder).__init__()

        self.query_dim = arg['query_dim']
        self.context_dim = arg['context_dim']
        self.heads = arg['heads']
        self.head_dim = arg['head_dim']
        self.output_dim = arg['output_dim']


        self.shape_learning = ShapeLearning(arg)

        self.pos_enc = PosEncoderMLP()
        self.cross_attn = PreNorm(self.query_dim, Attention(self.query_dim, self.context_dim, heads = self.heads, head_dim = self.head_dim))

        self.to_output = nn.Sequential(
                nn.LayerNorm(self.query_dim),
                nn.Linear(self.query_dim, self.output_dim))

        self.weight_initialize()

    def weight_initialize(self):
        nn.init.zeros_(self.to_output[1].weight)
        nn.init.zeros_(self.to_output[1].bias)


    def forward(self, x, queries):
        x = self.shape_learning(x)

        queries_embedding = self.pos_enc(queries)
        latents = self.cross_attn(queries_embedding, context = x)

        output = self.to_output(latents)
        return output


class ShapeLearning(nn.Module):
    def __init__(self, arg):
        super(ShapeLearning).__init__()

        self.depth = arg['depth']
        self.dim = arg['dim']
        self.heads = arg['heads']
        self.head_dim = arg['head_dim']

        self.layers = nn.ModuleList([])

        for i in range(self.depth):
            self.layers.append(nn.ModuleList([
                PreNorm(self.dim, Attention(self.dim, heads = self.heads, head_dim = self.head_dim)),
                PreNorm(self.dim, FeedForward(self.dim))]))

    def forward(self, x):

        for self_attn , ff in self.layers:
            x = self_attn(x) + x
            x = ff(x) + x

        return x
