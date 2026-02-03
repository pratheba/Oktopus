import torch
import torch.nn as nn
import numpy as np

class PosEncoding(torch.nn.Module):
    """
    Implementation of NeRF's positional encoding from Pixel-Nerf
    """

    def __init__(self, num_freqs=6, start= 0, d_in=3, dim=128, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        #self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.freqs = freq_factor * 2.0 ** torch.arange(start, start+num_freqs)

        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        #self.register_buffer(
        #    "_inf_freqs", torch.repeat_interleave(self.inference_freqs, 2).view(1, -1, 1)
        #)
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

        self.mlp = nn.Linear(self.d_out, dim)

        #self.linear = torch.nn.Linear(self.d_in, 128)
        #self.relu = torch.nn.ReLU()
        #self.linear2 = torch.nn.Linear(128 , 36)

    def forwardsimple(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (N_batch, batch_size, self.d_in)
        :return (N_batch, batch_size, self.d_out)
        """
        xsize = list(x.shape)
        embed = x.view(-1, xsize[-1])
        embed = embed.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        
        xsize[-1] = -1
        embed = embed.view(xsize)
        #print(embed.shape)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        return embed
        #print(embed.shape)
        #print(self.d_in)
        
        #embed = self.mlp(torch.cat((embed,x), dim=-1))
        #return embed

        #emb = self.linear2(self.relu(self.linear(x)))
        #emb = torch.cat((x, emb), dim=-1)
        #return emb

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (N_batch, batch_size, self.d_in)
        :return (N_batch, batch_size, self.d_out)
        """
        return self.forwardsimple(x)

    @torch.no_grad()
    def inference(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (N_batch, batch_size, self.d_in)
        :return (N_batch, batch_size, self.d_out)
        """
        return self.forwardsimple(x)


class PosEncodingMLP(torch.nn.Module):
    """
    Implementation of NeRF's positional encoding from Pixel-Nerf
    """

    def __init__(self, num_freqs=6, d_in=3, dim=128, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)

        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

        self.mlp = nn.Linear(self.d_out, self.dim)

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (N_batch, batch_size, self.d_in)
        :return (N_batch, batch_size, self.d_out)
        """
        xsize = list(x.shape)
        embed = x.view(-1, xsize[-1])
        embed = embed.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        
        xsize[-1] = -1
        embed = embed.view(xsize)
        #print(embed.shape)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)
        #embed = self.mlp(embed)
        return embed

