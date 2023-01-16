from collections import OrderedDict
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from einops import rearrange, repeat


def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x, modulations):
        x = torch.sin(self.w0 * (x + modulations))
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
        elif act_cfg['type'] == 'Sigmoid':
            self.activation_func = nn.Sigmoid()
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
            # nn.init.uniform_(getattr(self, f'layer_{i}_fc').bias, init_cfg['a'], init_cfg['b'])
            nn.init.constant_(getattr(self, f'layer_{i}_fc').bias, 0)
            _in_channels = _out_channels

    def forward(self, x, *args):
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name[0])
            x = layer(x)
            layer = getattr(self, layer_name[1])
            x = layer(x, *args)
        return x


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers, is_BN=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.is_BN = is_BN
        if is_BN:
            self.bn_layer = nn.Sequential(
                nn.Linear(dim_in, dim_hidden), 
                nn.BatchNorm1d(dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_in), 
                nn.BatchNorm1d(dim_in)
            )
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden)
            ))
        print('is_BN:', is_BN)

    def forward(self, z):
        x = z
        if self.is_BN:
            x = self.bn_layer(x)
        
        hiddens = []
        for layer in self.layers:
            # x = layer(z)
            # hiddens.append(x)
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)

class SimpleModulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            self.layers.append(
                nn.Linear(dim_in, dim_hidden)
            )
        print('simple modulator')

    def forward(self, z):
        hiddens = []
        for layer in self.layers:
            # x = layer(z)
            # hiddens.append(x)
            x = layer(z)
            hiddens.append(x)
        return tuple(hiddens)


class Siren_Modulation(BaseModule):
    def __init__(self,
                 num_inner_layers=5,
                 in_channels=2,
                 out_channels=3,
                 base_channels=256,
                 latent_dim=512,
                 is_diff_mods=False,
                 is_shift=False,
                 is_residual=False,
                 is_BN=False,
                 bias=True,
                 expansions=[1],
                 init_cfg=None,
                 ):

        super(Siren_Modulation, self).__init__(init_cfg)
        if len(expansions) == 1:
            self.expansions = expansions * num_inner_layers
        assert num_inner_layers == len(self.expansions)
        if isinstance(self.expansions, list):
            self.expansions = torch.tensor(self.expansions)

        self.num_inner_layers = num_inner_layers
        self.bias = bias
        self.is_residual = is_residual
        self.is_diff_mods = is_diff_mods
        self.is_shift = is_shift
        self.is_BN = is_BN

        self.layers = []
        out_channels_list = base_channels * self.expansions
        out_channels_list = torch.cat((out_channels_list, torch.tensor([out_channels])))
        _in_channels = in_channels
        # inner layer
        for i in range(self.num_inner_layers+1):
            _out_channels = out_channels_list[i]

            w0 = 30.0
            c = 6.0
            w_std = torch.sqrt(torch.tensor(c) / _in_channels) / w0
            if i == 0:
                w_std = 1. / _in_channels

            init_cfg = dict(type='Uniform', layer='Linear', a=-w_std, b=w_std)
            act_cfg = dict(type='Sine', w0=w0)
            if i == self.num_inner_layers:
                act_cfg = dict(type='Identity')

            layer = SirenLayer(
                _in_channels, _out_channels, num_fcs=1,
                bias=bias, act_cfg=act_cfg,
                init_cfg=init_cfg
            )
            layer_name = f'SirenLayer_{i}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)
            _in_channels = _out_channels

        print(f'is_diff_mods: {is_diff_mods} is_shift: {is_shift} is_residual: {is_residual}')
        if self.is_diff_mods:
            self.modulation_size_dict = self.get_bias_size()
            self.modulation_dims = sum(self.modulation_size_dict.values())
        
        if self.is_shift:
            self.modulator = Modulator(
                dim_in=latent_dim, 
                dim_hidden=out_channels_list[0], 
                num_layers=self.num_inner_layers,
                is_BN=is_BN
                )
            # self.modulator = SimpleModulator(
            #     dim_in=latent_dim, 
            #     dim_hidden=out_channels_list[0], 
            #     num_layers=self.num_inner_layers
            #     )
            
            # self.shift_modulation_layer = nn.Sequential(
            #     nn.Linear(latent_dim, latent_dim * 2, bias=bias),
            #     nn.LeakyReLU(),
            #     nn.Linear(latent_dim * 2, self.modulation_dims, bias=bias),
            #     # nn.Linear(num_modulation, _out_channels, bias=bias),
            #     # nn.LeakyReLU()
            # )

    def get_bias_size(self):
        parameters_size = OrderedDict()
        for name, parm in self.named_parameters():
            if '.weight' in name and 'SirenLayer' in name:
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
        super(Siren_Modulation, self).train(False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, modulations=None):
        # modulations = None
        if self.is_shift:
            mods = self.modulator(modulations)
            # mods = self.shift_modulation_layer(modulations)
            # mods = torch.split(mods, list(self.modulation_size_dict.values()), dim=0)
        else:
            if self.is_diff_mods:
                mods = torch.split(modulations, list(self.modulation_size_dict.values()), dim=0)
            else:
                mods = cast_tuple(modulations, self.num_inner_layers)

        mods = mods + (None,)

        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            mod = mods[i]
            if self.is_residual and i not in [0, len(self.layers)-1]:
                residual = layer(x, mod)
                x = x + residual
            else:
                if mod is not None:
                    x = layer(x, mod)
                else:
                    x = layer(x)
        x = x + 0.5
        return x

    def train_model_w_b(self, mode=True):
        super(Siren_Modulation, self).train(mode)
        for param in self.parameters():
            param.requires_grad = True
    
    def get_BN_feature(self, x):
        assert self.is_BN
        self.modulator.eval()
        with torch.no_grad():
            x = self.modulator.bn_layer(x)
        return x

