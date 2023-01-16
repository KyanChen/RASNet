# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule


class SineCosPE(BaseModule):
    """Position encoding with sine and cosine functions.
    """
    def __init__(self, input_dim, N_freqs=10, max_freq=10-1, periodic_fns=[torch.sin, torch.cos],
                 log_sampling=True, include_input=True, trainable=False):
        super().__init__()

        self.periodic_fns = periodic_fns
        self.include_input = include_input or len(periodic_fns) == 0
        self.out_dim = len(periodic_fns) * input_dim * N_freqs

        # Identity map if no periodic_fns provided
        if self.include_input:
            self.out_dim += input_dim

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        if trainable:
            self.freq_bands = nn.Parameter(freq_bands, requires_grad=True)
        else:
            self.register_buffer('freq_bands', freq_bands, persistent=False)

    def forward(self, inputs):
        # inputs: BxC
        N_freqs = len(self.freq_bands)

        embeds = []
        for periodic_fn in self.periodic_fns:
            x_freq = inputs[..., None].expand(inputs.shape+(N_freqs,)) * self.freq_bands # [batch_shape, C, N_freq]
            x_freq = periodic_fn(x_freq)
            embeds.append(x_freq.transpose(-1, -2))  # [batch_shape, N_freq, C]
        embeds = torch.stack(embeds, -2)  # [batch_shape, N_freq, N_fns, C]
        embeds = embeds.reshape(inputs.shape[:-1]+(-1,)) # [batch_shape, N_freq x N_fns x C]

        if self.include_input:
            embeds = torch.cat([inputs, embeds], -1)

        return embeds
