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
from mmcv.utils.parrots_wrapper import _BatchNorm
import torch.nn.functional as F

eps = 1.0e-5


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x, modulations):
        x = torch.sin(self.w0 * (x + modulations))
        # x = F.relu(x + modulations)
        # x = x.sigmoid()
        return x


class SirenLayer(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_fcs=1,
                 bias=True,
                 act_cfg=dict(type='Sine', w0=30.),
                 init_cfg=dict(type='Uniform', layer='Linear', a=-0.01, b=0.01),
                 **kwargs):
        super(SirenLayer, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        if act_cfg['type'] == 'Identity':
            self.activation_func = nn.Identity()
        else:
            self.activation_func = Sine()

        _in_channels = in_channels
        _out_channels = out_channels
        self.layers = []
        for i in range(0, num_fcs):
            self.add_module(f'layer_{i}_fc', nn.Linear(_in_channels, _out_channels, bias=bias))
            self.add_module(f'layer_{i}_actfunc', self.activation_func)
            self.layers.append([
                f'layer_{i}_fc',
                f'layer_{i}_actfunc'
            ])
            nn.init.uniform_(getattr(self, f'layer_{i}_fc').weight, init_cfg['a'], init_cfg['b'])
            # # nn.init.uniform_(getattr(self, f'layer_{i}_fc').weight, -0.0001, 0.0001)
            nn.init.uniform_(getattr(self, f'layer_{i}_fc').bias, init_cfg['a'], init_cfg['b'])
            # nn.init.constant_(getattr(self, f'layer_{i}_fc').bias, 0)
            _in_channels = _out_channels
        # self.init_weights()

    def init_weights(self):
        super(SirenLayer, self).init_weights()

    def forward(self, x, *args):
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name[0])
            x = layer(x)
            layer = getattr(self, layer_name[1])
            x = layer(x, *args)
        return x


class Siren_Res(BaseModule):
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

        super(Siren_Res, self).__init__(init_cfg)
        if len(expansions) == 1:
            self.expansions = expansions * inner_layers
        assert inner_layers == len(self.expansions)
        if isinstance(self.expansions, list):
            self.expansions = torch.tensor(self.expansions)

        self.inner_layers = inner_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bias = bias

        self.layers = []
        _in_channels = in_channels
        out_channels_list = base_channels * self.expansions
        out_channels_list = torch.cat((out_channels_list, torch.tensor([self.out_channels])))
        for i in range(self.inner_layers + 1):
            _out_channels = out_channels_list[i]

            w0 = 30.
            if i == 0:
                w_std = 1. / _in_channels
            else:
                c = 6
                w_std = torch.sqrt(c / _in_channels) / w0
            init_cfg = dict(type='Uniform', layer='Linear', a=-w_std, b=w_std)
            if i == self.inner_layers:
                act_cfg = dict(type='Identity')
            else:
                act_cfg = dict(type='Sine', w0=w0)

            layer = SirenLayer(
                _in_channels, _out_channels, num_fcs=1,
                bias=bias, act_cfg=act_cfg,
                init_cfg=init_cfg
            )
            _in_channels = _out_channels
            layer_name = f'SirenLayer_{i}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

        self.modulation_size_dict = self.get_bias_size()
        _out_channels = sum(self.modulation_size_dict.values())
        self.shift_modulation_layer = nn.Sequential(
            nn.Linear(num_modulation, num_modulation * 2, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(num_modulation * 2, _out_channels, bias=bias),
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
        super(Siren_Res, self).init_weights()
        # nn.init.normal_(self.shift_modulation_layer.weight.data, -1/256., 1/256.)
        # nn.init.normal_(self.shift_modulation_layer.bias.data, -1 / 256., 1 / 256.)
        # nn.init.constant_(self.shift_modulation_layer.bias.data, 0)

    def forward(self, x, modulations):
        shift_modulations = self.shift_modulation_layer(modulations)
        # shift_modulations = modulations
        shift_modulations_split = torch.split(shift_modulations, list(self.modulation_size_dict.values()), dim=1)
        shift_modulations_split = shift_modulations_split + (None,)

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            if shift_modulations_split[i] is not None:
                # shift_modulations[:, t:t + sizes[i]]
                x = layer(x, shift_modulations_split[i])
                # t += sizes[i]
            else:
                x = layer(x)
        x = 0.5 * x + 0.5
        return x

    def train_model_w_b(self, mode=True):
        super(Siren_Res, self).train(mode)
        for param in self.parameters():
            param.requires_grad = True

