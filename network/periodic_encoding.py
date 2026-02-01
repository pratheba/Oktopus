import torch
import torch.nn as nn
import numpy as np

class PeriodicEncoding(torch.nn.Module):
    """
    Implementation of NeRF's positional encoding from Pixel-Nerf
    """

    def __init__(self, num_freqs=8, d_in=1, freq_factor=1, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        # 3.1416,   6.2832,  12.5664,  25.1327,  50.2655, 100.5310
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)

        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        #3.1416, 3.1416,   6.2832,  6.2832, 12.5664, 12.5664, ..  
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 0 pi/2 0 pi/2... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))


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


