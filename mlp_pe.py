# from ..builder import BACKBONES
from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init)
from mmcv.cnn.bricks import DropPath, build_activation_layer
from mmcv.runner import BaseModule
from positional_encoding import SineCosPE
from einops import repeat


class MLP_PE(BaseModule):
    def __init__(self,
                 inner_layers=6,
                 in_channels=2,
                 out_channels=3,
                 base_channels=512,
                 num_modulation=512,
                 bias=True,
                 expansions=[1],
                 init_cfg=None,
                 ):
        
        super(MLP_PE, self).__init__(init_cfg)
        if len(expansions) == 1:
            self.expansions = expansions * inner_layers
        assert inner_layers == len(self.expansions)
        self.expansions = torch.tensor(self.expansions)
        self.inner_layers = inner_layers

        self.pe = SineCosPE(input_dim=in_channels, N_freqs=32, max_freq=10 - 1)
        _in_channels = self.pe.out_dim
        self.pe_reshape = nn.Sequential(
                nn.Linear(self.pe.out_dim, base_channels, bias=bias)
            )

        self.layers = []
        out_channels_list = base_channels * self.expansions
        _in_channels = base_channels + num_modulation
        for i in range(self.inner_layers):
            _out_channels = out_channels_list[i]

            layer = nn.Sequential(
                nn.Linear(_in_channels, _out_channels, bias=bias),
                nn.LeakyReLU()
            )

            _in_channels = _out_channels
            if i % 3 == 2:  # 2,5
                _in_channels = _out_channels + num_modulation
            layer_name = f'Layer_{i}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)
        layer = nn.Sequential(
            nn.Linear(_in_channels, _in_channels//4, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(_in_channels//4, out_channels, bias=bias),
        )
        self.add_module('Layer_last', layer)
        self.layers.append('Layer_last')
        self.split_modulation_list = [num_modulation]*(1+inner_layers//3)
        _out_channels = (1+inner_layers//3)*num_modulation
        self.shift_modulation_layer = nn.Sequential(
            nn.Linear(num_modulation, num_modulation*2, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(num_modulation*2, _out_channels, bias=bias),
        )
        # self.init_weights()

    def get_bias_size(self):
        parameters_size = OrderedDict()
        for name, parm in self.named_parameters():
            if '.weight' in name:
                parameters_size[name.replace('.weight', '.bias')] = parm.size(0)
        parameters_size.popitem(last=True)
        return parameters_size

    def get_parameters_size(self):
        parameters_size = dict()
        for name, parm in self.named_parameters():
            parameters_size[name] = parm.size()
        return parameters_size

    def freeze_model_w(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.requires_grad = False

    def freeze_model_b(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = False

    def train_model_w(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.requires_grad = True

    def train_model_b(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True

    def get_model_b_data(self):
        data = {}
        for name, param in self.named_parameters():
            if 'bias' in name:
                data[name] = param.data
        return data

    def set_model_b_data(self, data):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data = data[name]

    def zero_model_b(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data = torch.zeros_like(param)

    def freeze_model_w_b(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def init_weights(self):
        super(MLP_PE, self).init_weights()
        # nn.init.normal_(self.shift_modulation_layer.weight.data, -1/256., 1/256.)
        # nn.init.normal_(self.shift_modulation_layer.bias.data, -1 / 256., 1 / 256.)
        # nn.init.constant_(self.shift_modulation_layer.bias.data, 0)

    def forward(self, x, modulations):
        shift_modulations = self.shift_modulation_layer(modulations)
        shift_modulations_split = torch.split(shift_modulations, self.split_modulation_list, dim=1)
        x = self.pe(x)
        # import pdb
        # pdb.set_trace()
        x = self.pe_reshape(x)
        modulation_tmp = repeat(shift_modulations_split[0], '1 c -> t c', t=x.size(0))
        x = torch.cat((x, modulation_tmp), dim=1)
        id_modulation = 1
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            if i != 0 and i % 3 == 0:
                modulation_tmp = repeat(shift_modulations_split[id_modulation], '1 c -> t c', t=x.size(0))
                x = torch.cat((x, modulation_tmp), dim=1)
                id_modulation += 1
            x = layer(x)
        return x

    def train_model_w_b(self, mode=True):
        super(MLP_PE, self).train(mode)
        for param in self.parameters():
            param.requires_grad = True

