import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np


class PointEmbed(nn.Module):
    def __init__(self, num_freq=6, dim=128):
        super().__init__()

        self.embedding_dim = num_freq * 6

        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e])
            ])

        self.register_buffer('basis', e) # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(x, basis):
        projections = torch.einsum(
                'bnd, de -> bne', x, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=-1)
        return embeddings


    def forward(self, x):
        # x : B X N 3

        embed = self.mlp(torch.cat([self.embed(x, self.basis), x], dim=-1))
        return embed

